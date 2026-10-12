"""Decoders available to the benchmark scripts, by sinter name."""

from __future__ import annotations

import sinter


def all_decoders() -> dict[str, sinter.Decoder]:
    from sinter._decoding._decoding_all_built_in_decoders import BUILT_IN_DECODERS

    out: dict[str, sinter.Decoder] = dict(BUILT_IN_DECODERS)
    try:
        import beliefmatching  # noqa: F401

        from surfdec.reference import BeliefMatchingDecoder

        out["beliefmatching"] = BeliefMatchingDecoder()
        out["beliefmatching-noshortcut"] = BeliefMatchingDecoder(shortcut=False)
    except ImportError:
        pass
    from surfdec.native import MODES, NativeDecoder

    out["evmatch"] = NativeDecoder("evidence-both")
    for mode in MODES:
        out[f"nat-{mode}"] = NativeDecoder(mode)
        out[f"nat-{mode}-first"] = NativeDecoder(mode, decomposition="first")
        out[f"nat-{mode}-po"] = NativeDecoder(mode, partners_only=True)

    from surfdec.proto import PROTOS, ProtoDecoder

    for name, kw in PROTOS.items():
        out[name] = ProtoDecoder(**kw)
    return out
