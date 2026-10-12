"""Surface-code memory circuits with SI1000 noise.

SI1000 ("superconducting inspired, 1000 ns cycle") is the noise model of
Gidney, Newman & McEwen, "Benchmarking the planar honeycomb code",
Quantum 6, 813 (2022), as implemented in Gidney's `gen` NoiseModel.si1000:

    CZ                          DEPOLARIZE2(p) after
    single-qubit Clifford       DEPOLARIZE1(p/10) after
    reset (R)                   X_ERROR(2p) after
    measurement (M, MR)         result flipped with probability 5p
    idle qubit, any layer       DEPOLARIZE1(p/10)
    idle qubit, layer with M/R  additional DEPOLARIZE1(2p)

The noiseless rotated memory circuit comes from `stim.Circuit.generated`.
Its CX gates are compiled to H.CZ.H (SI1000 assumes CZ as the native
two-qubit gate), Hadamards that meet on the same qubit cancel, and adjacent
single-qubit layers are merged before noise is applied.
"""

from __future__ import annotations

import stim

_ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}
_COLLAPSING = {"M", "MR", "R"}


class _Op:
    __slots__ = ("name", "qubits")

    def __init__(self, name: str, qubits: list[int]):
        self.name = name
        self.qubits = qubits


# A layer is an ordered list of items: _Op (gates) or stim.CircuitInstruction
# (annotations). Order matters: detectors refer to measurements by offset.
Layer = list


def _gates(layer: Layer) -> list[_Op]:
    return [x for x in layer if isinstance(x, _Op)]


def _only_h(layer: Layer) -> bool:
    return len(layer) == 1 and isinstance(layer[0], _Op) and layer[0].name == "H"


def _layers(circuit: stim.Circuit) -> list[Layer]:
    out: list[Layer] = [[]]
    for inst in circuit.flattened():
        if inst.name == "TICK":
            out.append([])
        elif inst.name in _ANNOTATIONS:
            if inst.name != "QUBIT_COORDS":
                out[-1].append(inst)
        else:
            if inst.gate_args_copy():
                raise ValueError(f"expected a noiseless circuit, got {inst}")
            out[-1].append(_Op(inst.name, [t.value for t in inst.targets_copy()]))
    return [m for m in out if m]


def _compile_cx_to_cz(layers: list[Layer]) -> list[Layer]:
    out = []
    for layer in layers:
        gates = _gates(layer)
        cx = [g for g in gates if g.name in ("CX", "CNOT")]
        if not cx:
            out.append(layer)
            continue
        if len(cx) != len(gates):
            raise ValueError("CX mixed with other gates in one layer")
        pairs = [q for g in cx for q in g.qubits]
        tgts = pairs[1::2]
        notes = [x for x in layer if not isinstance(x, _Op)]
        out.append([_Op("H", tgts)])
        out.append([_Op("CZ", pairs)] + notes)
        out.append([_Op("H", list(tgts))])
    return out


def _cancel_hadamards(layers: list[Layer]) -> list[Layer]:
    """Remove H pairs on a qubit that have nothing else acting between them."""
    hsets = [set(m[0].qubits) if _only_h(m) else None for m in layers]
    pending: dict[int, int] = {}  # qubit -> index of a layer holding an uncancelled H
    for k, layer in enumerate(layers):
        if hsets[k] is not None:
            for q in sorted(hsets[k]):
                if q in pending:
                    hsets[pending.pop(q)].discard(q)
                    hsets[k].discard(q)
                else:
                    pending[q] = k
        else:
            for g in _gates(layer):
                for q in g.qubits:
                    pending.pop(q, None)
    out = []
    for k, layer in enumerate(layers):
        if hsets[k] is not None:
            layer = [_Op("H", sorted(hsets[k]))] if hsets[k] else []
        if layer:
            out.append(layer)
    return out


def _merge_single_qubit_layers(layers: list[Layer]) -> list[Layer]:
    out: list[Layer] = []
    for layer in layers:
        if out and _only_h(layer) and _only_h(out[-1]):
            a, b = out[-1][0].qubits, layer[0].qubits
            if not set(a) & set(b):
                out[-1] = [_Op("H", sorted(a + b))]
                continue
        out.append(layer)
    return out


def add_si1000_noise(noiseless: stim.Circuit, p: float) -> stim.Circuit:
    """Compile CX->CZ and add SI1000 noise of strength p to a noiseless circuit."""
    layers = _merge_single_qubit_layers(_cancel_hadamards(_compile_cx_to_cz(_layers(noiseless))))
    all_qubits = sorted({q for layer in layers for g in _gates(layer) for q in g.qubits})
    out = stim.Circuit()
    for inst in noiseless.flattened():
        if inst.name == "QUBIT_COORDS":
            out.append(inst)
    for k, layer in enumerate(layers):
        if k:
            out.append("TICK")
        touched: list[int] = []
        collapse = False
        for item in layer:
            if not isinstance(item, _Op):
                out.append(item)
                continue
            g = item
            touched += g.qubits
            if g.name == "H":
                out.append("H", g.qubits)
                out.append("DEPOLARIZE1", g.qubits, p / 10)
            elif g.name == "CZ":
                out.append("CZ", g.qubits)
                out.append("DEPOLARIZE2", g.qubits, p)
            elif g.name == "R":
                collapse = True
                out.append("R", g.qubits)
                out.append("X_ERROR", g.qubits, 2 * p)
            elif g.name == "M":
                collapse = True
                out.append("M", g.qubits, 5 * p)
            elif g.name == "MR":
                collapse = True
                out.append("MR", g.qubits, 5 * p)
                out.append("X_ERROR", g.qubits, 2 * p)
            else:
                raise ValueError(f"unsupported gate {g.name}")
        if len(set(touched)) != len(touched):
            raise ValueError(f"qubit used twice in layer {k}")
        idle = sorted(set(all_qubits) - set(touched))
        if idle:
            out.append("DEPOLARIZE1", idle, p / 10)
            if collapse:
                out.append("DEPOLARIZE1", idle, 2 * p)
    return out


def surface_code_si1000(distance: int, p: float, rounds: int | None = None, basis: str = "z") -> stim.Circuit:
    """Rotated surface-code memory experiment (d rounds by default) under SI1000 noise."""
    noiseless = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=distance if rounds is None else rounds,
    )
    return add_si1000_noise(noiseless, p)
