"""Sinter-compatible evidence-conditioned correlated matching."""

from __future__ import annotations

import numpy as np
import sinter
import stim

from evmatch.dem import DemGraph, build

MODES = {
    "match": 0,  # plain matching (PyMatching weights, "fewest" decomposition)
    "correlated": 1,  # PyMatching-style two-pass correlated matching
    "evidence": 2,  # one pass, evidence-conditioned weights
    "evidence-correlated": 3,  # pass 1 full weights, pass 2 evidence + correlations
    "evidence-both": 4,  # both passes evidence-conditioned (default)
    "single-correlated": 5,  # ablation: hyperedge-free graph in both passes, no evidence
}


def _native_args(g: DemGraph) -> dict:
    single = np.zeros(len(g.e_nodes))
    mech_ptr = [0]
    mech_edges: list[int] = []
    mech_p: list[float] = []
    for h, es in enumerate(g.h_edges):
        p = float(g.h_prior[h])
        if len(es) == 1:
            e = es[0]
            single[e] = single[e] * (1 - p) + p * (1 - single[e])
        else:
            mech_edges.extend(es)
            mech_ptr.append(len(mech_edges))
            mech_p.append(p)
    return dict(
        num_detectors=g.num_dets,
        num_observables=g.num_obs,
        edge_u=np.ascontiguousarray(g.e_nodes[:, 0], dtype=np.int64),
        edge_v=np.ascontiguousarray(g.e_nodes[:, 1], dtype=np.int64),
        edge_obs=g.e_obs.astype(np.uint64),
        q_full=g.e_prior.astype(np.float64),
        q_single=single,
        mech_ptr=np.array(mech_ptr, dtype=np.int64),
        mech_edges=np.array(mech_edges, dtype=np.int64),
        mech_p=np.array(mech_p, dtype=np.float64),
    )


class CompiledEvMatch(sinter.CompiledDecoder):
    """An evmatch decoder compiled for one detector error model."""

    def __init__(self, native, num_detectors: int, num_observables: int):
        self._native = native
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self._num_obs_bytes = (num_observables + 7) // 8

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        data = np.ascontiguousarray(bit_packed_detection_event_data, dtype=np.uint8)
        return self._native.decode_batch(data, self._num_obs_bytes)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """Decodes unpacked shots, shape (shots, num_detectors), to observable flips (shots, num_observables)."""
        dets = np.asarray(detection_events, dtype=np.uint8)
        packed = np.packbits(dets, axis=1, bitorder="little")
        out = self.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
        return np.unpackbits(out, axis=1, bitorder="little", count=self.num_observables).astype(bool)


class EvMatchDecoder(sinter.Decoder):
    """Evidence-conditioned correlated matching, as a sinter decoder.

    Args:
        mode: one of MODES; "evidence-both" is the method of the paper, the
            others are ablations.
        decomposition: how to split multi-detector error mechanisms into
            edges: "fewest" (default) or "first" (stim's first decomposition).
        cap: upper bound on the implied probability in the correlated pass.
    """

    def __init__(self, mode: str = "evidence-both", decomposition: str = "fewest", cap: float = 0.5,
                 partners_only: bool = False):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
        self.mode = mode
        self.decomposition = decomposition
        self.cap = cap
        self.partners_only = partners_only

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> CompiledEvMatch:
        from evmatch import _native

        g = build(dem, self.decomposition)
        native = _native.Decoder(
            **_native_args(g), mode=MODES[self.mode], cap=self.cap, partners_only=self.partners_only
        )
        return CompiledEvMatch(native, dem.num_detectors, dem.num_observables)


def sinter_decoders() -> dict[str, sinter.Decoder]:
    """For `sinter collect --custom_decoders_module_function "evmatch:sinter_decoders"`."""
    return {"evmatch": EvMatchDecoder()}
