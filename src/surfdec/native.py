"""Sinter decoders backed by the native extension."""

from __future__ import annotations

import numpy as np
import sinter
import stim

from surfdec.dem import DemGraph, build

MODES = {
    "match": 0,
    "correlated": 1,
    "evidence": 2,
    "evidence-correlated": 3,
    "evidence-both": 4,
    "single-correlated": 5,
}


def native_args(g: DemGraph):
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
        edge_u=g.e_nodes[:, 0].astype(np.int64),
        edge_v=g.e_nodes[:, 1].astype(np.int64),
        edge_obs=g.e_obs.astype(np.uint64),
        q_full=g.e_prior.astype(np.float64),
        q_single=single,
        mech_ptr=np.array(mech_ptr, dtype=np.int64),
        mech_edges=np.array(mech_edges, dtype=np.int64),
        mech_p=np.array(mech_p, dtype=np.float64),
    )


class _Compiled(sinter.CompiledDecoder):
    def __init__(self, dec, num_obs: int):
        self.dec = dec
        self.num_obs_bytes = (num_obs + 7) // 8

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        data = np.ascontiguousarray(bit_packed_detection_event_data, dtype=np.uint8)
        return self.dec.decode_batch(data, self.num_obs_bytes)


class NativeDecoder(sinter.Decoder):
    def __init__(
        self,
        mode: str = "evidence-correlated",
        decomposition: str = "fewest",
        cap: float = 0.5,
        partners_only: bool = False,
    ):
        self.mode = mode
        self.decomposition = decomposition
        self.cap = cap
        self.partners_only = partners_only

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        from surfdec import _native

        g = build(dem, self.decomposition)
        dec = _native.Decoder(
            **native_args(g), mode=MODES[self.mode], cap=self.cap, partners_only=self.partners_only
        )
        return _Compiled(dec, dem.num_observables)
