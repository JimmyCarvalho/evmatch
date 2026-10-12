"""Sinter adapter for the reference belief-matching implementation.

`beliefmatching.BeliefMatchingSinterDecoder` only implements sinter's file
interface; this exposes `compile_decoder_for_dem` so it can be timed on the
same in-memory shots as the other decoders.
"""

from __future__ import annotations

import numpy as np
import sinter
import stim


class _CompiledBeliefMatching(sinter.CompiledDecoder):
    def __init__(self, bm, num_dets: int, num_obs: int):
        self.bm = bm
        self.num_dets = num_dets
        self.num_obs = num_obs

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        dets = np.unpackbits(bit_packed_detection_event_data, axis=1, bitorder="little")[:, : self.num_dets]
        pred = self.bm.decode_batch(dets)
        return np.packbits(pred.astype(np.uint8), axis=1, bitorder="little")


class BeliefMatchingDecoder(sinter.Decoder):
    def __init__(self, max_bp_iters: int = 20):
        self.max_bp_iters = max_bp_iters

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        from beliefmatching import BeliefMatching

        bm = BeliefMatching(dem, max_bp_iters=self.max_bp_iters)
        return _CompiledBeliefMatching(bm, dem.num_detectors, dem.num_observables)
