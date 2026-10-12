"""Every decoder the benchmark scripts know about, by sinter name.

Besides sinter's built-ins (pymatching, pymatching-correlated, ...):
  evmatch                 the method (EvMatchDecoder default)
  nat-<mode>[-first|-po]  evmatch modes/ablations (see evmatch.MODES)
  beliefmatching[-noshortcut]  reference belief-matching (needs `beliefmatching`)
  proto-*                 slow Python prototypes (bench/proto.py)
"""

from __future__ import annotations

import sinter

from evmatch import MODES, EvMatchDecoder


def all_decoders() -> dict[str, sinter.Decoder]:
    from sinter._decoding._decoding_all_built_in_decoders import BUILT_IN_DECODERS

    out: dict[str, sinter.Decoder] = dict(BUILT_IN_DECODERS)
    out["evmatch"] = EvMatchDecoder()
    for mode in MODES:
        out[f"nat-{mode}"] = EvMatchDecoder(mode)
        out[f"nat-{mode}-first"] = EvMatchDecoder(mode, decomposition="first")
        out[f"nat-{mode}-po"] = EvMatchDecoder(mode, partners_only=True)
    try:
        import beliefmatching  # noqa: F401

        from reference import BeliefMatchingDecoder

        out["beliefmatching"] = BeliefMatchingDecoder()
        out["beliefmatching-noshortcut"] = BeliefMatchingDecoder(shortcut=False)
        out["beliefmatching-30"] = BeliefMatchingDecoder(max_bp_iters=30)
    except ImportError:
        pass
    try:
        from proto import PROTOS, ProtoDecoder

        for name, kw in PROTOS.items():
            out[name] = ProtoDecoder(**kw)
    except ImportError:  # scipy / ldpc not installed
        pass
    return out
