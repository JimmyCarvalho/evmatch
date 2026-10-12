"""Sinter adapter for the reference belief-matching implementation (benchmark only).

`beliefmatching.BeliefMatchingSinterDecoder` only implements sinter's file
interface; this exposes `compile_decoder_for_dem` so it can be timed on the
same in-memory shots as the other decoders.
"""

from __future__ import annotations

import numpy as np
import sinter
import stim


def _decode_without_shortcut(bm, syndrome: np.ndarray) -> np.ndarray:
    """BeliefMatching.decode, but always matching on the BP marginals.

    The package returns BP's own hard decision whenever BP converges; this
    variant ignores convergence and always reweights and matches.
    """
    import pymatching

    bm._bpd.decode(syndrome)
    ps_h = 1 / (1 + np.exp(bm._bpd.log_prob_ratios))
    ps_e = bm._matrices.hyperedge_to_edge_matrix @ ps_h
    ps_e = np.clip(ps_e, 1e-14, 1 - 1e-14)
    matching = pymatching.Matching.from_check_matrix(
        bm._matrices.edge_check_matrix,
        weights=-np.log(ps_e),
        faults_matrix=bm._matrices.edge_observables_matrix,
        use_virtual_boundary_node=True,
    )
    return matching.decode(syndrome)


class _CompiledBeliefMatching(sinter.CompiledDecoder):
    def __init__(self, bm, num_dets: int, num_obs: int, shortcut: bool):
        self.bm = bm
        self.num_dets = num_dets
        self.num_obs = num_obs
        self.shortcut = shortcut

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        dets = np.unpackbits(bit_packed_detection_event_data, axis=1, bitorder="little")[:, : self.num_dets]
        if self.shortcut:
            pred = self.bm.decode_batch(dets)
        else:
            pred = np.array([_decode_without_shortcut(self.bm, row) for row in dets], dtype=np.uint8)
        return np.packbits(pred.astype(np.uint8), axis=1, bitorder="little")


class BeliefMatchingDecoder(sinter.Decoder):
    def __init__(self, max_bp_iters: int = 20, shortcut: bool = True):
        self.max_bp_iters = max_bp_iters
        self.shortcut = shortcut

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        from beliefmatching import BeliefMatching

        bm = BeliefMatching(dem, max_bp_iters=self.max_bp_iters)
        return _CompiledBeliefMatching(bm, dem.num_detectors, dem.num_observables, self.shortcut)
