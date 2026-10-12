"""Slow Python prototypes, used only to measure logical error rates of ideas.

Each shot rebuilds a pymatching graph with shot-specific weights, so these
are 100-1000x slower than PyMatching; speed is the job of the native code.

Pipeline (each stage optional):
  1. edge probabilities: priors, or belief-propagation posteriors on the full
     hypergraph (as in belief-matching, Higgott et al. PRX 13, 031007);
  2. first matching pass;
  3. correlated second pass: edges that share an error mechanism with a
     matched edge get probability min(0.5, P(e1 and e2) / P(e1)), as in
     PyMatching's enable_correlations, optionally from a base graph that
     drops mechanisms whose other components show no evidence ("negative
     evidence").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymatching
import sinter
import stim

from surfdec.dem import DemGraph, build


@dataclass(frozen=True)
class ProtoConfig:
    bp_iters: int = 0  # 0 = no BP, use priors
    bp_method: str = "product_sum"
    bp_schedule: str = "parallel"
    ms_scaling: float = 1.0
    bp_converge_shortcut: bool = True  # belief-matching returns the BP solution when it converges
    corr_pass: bool = False
    neg: str = "none"  # none | touch | matched
    neg_alpha: float = 0.0  # weight kept by mechanisms without evidence (neg != none)
    neg_pass1: bool = False  # also apply "touch" negative evidence to the first pass
    touch_rule: str = "any"  # any: a component counts as touched if any of its detectors fired; all: every one
    cap: float = 0.5
    decomposition: str = "fewest"


def _w(p):
    p = np.clip(p, 1e-14, 0.5)
    return np.log((1 - p) / p)


class _Compiled(sinter.CompiledDecoder):
    def __init__(self, g: DemGraph, cfg: ProtoConfig):
        self.g, self.cfg = g, cfg
        self.bpd = None
        if cfg.bp_iters > 0:
            from ldpc import BpDecoder

            self.bpd = BpDecoder(
                g.H, error_channel=list(g.h_prior), max_iter=cfg.bp_iters, bp_method=cfg.bp_method,
                schedule=cfg.bp_schedule, ms_scaling_factor=cfg.ms_scaling,
            )
        # Multi-component mechanisms: for each edge, (other edge, mechanism) pairs.
        self.partners: list[list[tuple[int, int]]] = [[] for _ in range(len(g.e_nodes))]
        self.multi = np.zeros(len(g.h_dets), dtype=bool)
        for h, es in enumerate(g.h_edges):
            if len(es) > 1:
                self.multi[h] = True
                for a in es:
                    for b in es:
                        if a != b:
                            self.partners[a].append((b, h))
        self.single_h = [h for h, es in enumerate(g.h_edges) if len(es) == 1]
        self.Hobs = None

    def _matching(self, w):
        g = self.g
        return pymatching.Matching.from_check_matrix(
            g.E, weights=w, faults_matrix=g.EO, use_virtual_boundary_node=True
        )

    def _edge_probs(self, post_h, mask=None):
        """Independent combination of the mechanisms in each edge's decomposition."""
        g = self.g
        q = np.zeros(len(g.e_nodes))
        ph = post_h if mask is None else post_h * mask
        # log(1 - 2p) trick: P(odd) = (1 - prod(1 - 2 p_i)) / 2
        lg = np.log(np.clip(1 - 2 * ph, 1e-300, None))
        acc = g.H2E @ lg
        q = (1 - np.exp(acc)) / 2
        return q

    def _touch_keep(self, syn: np.ndarray) -> np.ndarray:
        g, cfg = self.g, self.cfg
        flipped = syn.astype(bool)
        u, v = g.e_nodes[:, 0], g.e_nodes[:, 1]
        fu = flipped[u]
        fv = np.where(v >= 0, flipped[np.maximum(v, 0)], cfg.touch_rule == "all")
        ev = (fu | fv) if cfg.touch_rule == "any" else (fu & fv)
        keep = np.ones(len(g.h_dets))
        for h in np.flatnonzero(self.multi):
            if not all(ev[e] for e in g.h_edges[h]):
                keep[h] = cfg.neg_alpha
        return keep

    def decode_one(self, syn: np.ndarray) -> np.ndarray:
        g, cfg = self.g, self.cfg
        if self.bpd is not None:
            hard = self.bpd.decode(syn)
            if cfg.bp_converge_shortcut and self.bpd.converge:
                obs = 0
                for h in np.flatnonzero(hard):
                    obs ^= int(g.h_obs[h])
                return np.array([(obs >> o) & 1 for o in range(g.num_obs)], dtype=np.uint8)
            post_h = 1 / (1 + np.exp(self.bpd.log_prob_ratios))
        else:
            post_h = g.h_prior
        touch_keep = None
        if cfg.neg == "touch" or cfg.neg_pass1:
            touch_keep = self._touch_keep(syn)
        e_p = self._edge_probs(post_h, touch_keep if cfg.neg_pass1 else None)
        w1 = _w(e_p)
        m1 = self._matching(w1)
        if not cfg.corr_pass:
            return m1.decode(syn)
        matched = m1.decode_to_edges_array(syn)
        matched_ids = []
        for u, v in matched:
            u, v = int(u), int(v)
            key = (min(u, v), max(u, v)) if v >= 0 and u >= 0 else (max(u, v), -1)
            matched_ids.append(g.edge_index[key])
        if cfg.neg == "none":
            w2 = w1.copy()
        elif cfg.neg == "touch":
            w2 = _w(self._edge_probs(post_h, touch_keep))
        else:
            ev = np.zeros(len(g.e_nodes), dtype=bool)
            ev[matched_ids] = True
            keep = np.ones(len(g.h_dets))
            for h in np.flatnonzero(self.multi):
                if not all(ev[e] for e in g.h_edges[h]):
                    keep[h] = cfg.neg_alpha
            w2 = _w(self._edge_probs(post_h, keep))
        for e1 in matched_ids:
            joint: dict[int, float] = {}
            for e2, h in self.partners[e1]:
                pj = joint.get(e2, 0.0)
                joint[e2] = pj * (1 - post_h[h]) + post_h[h] * (1 - pj)
            for e2, pj in joint.items():
                q = min(cfg.cap, pj / max(e_p[e1], 1e-14))
                if q > 0:
                    w2[e2] = min(w2[e2], _w(q))
        return self._matching(w2).decode(syn)

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        dets = np.unpackbits(bit_packed_detection_event_data, axis=1, bitorder="little")[:, : self.g.num_dets]
        out = np.zeros((dets.shape[0], self.g.num_obs), dtype=np.uint8)
        for i in range(dets.shape[0]):
            out[i] = self.decode_one(dets[i])
        return np.packbits(out, axis=1, bitorder="little")


class ProtoDecoder(sinter.Decoder):
    def __init__(self, **kw):
        self.cfg = ProtoConfig(**kw)

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _Compiled(build(dem, self.cfg.decomposition), self.cfg)


PROTOS = {
    "proto-mwpm": dict(),
    "proto-mwpm-first": dict(decomposition="first"),
    "proto-mwpm-direct": dict(decomposition="direct"),
    "proto-corr-first": dict(corr_pass=True, decomposition="first"),
    "proto-corr": dict(corr_pass=True),
    "proto-corr-neg-touch": dict(corr_pass=True, neg="touch"),
    "proto-corr-neg-matched": dict(corr_pass=True, neg="matched"),
    "proto-neg1": dict(neg_pass1=True),
    "proto-neg-both": dict(corr_pass=True, neg="touch", neg_pass1=True),
    "proto-corr-neg-a10": dict(corr_pass=True, neg="touch", neg_alpha=0.1),
    "proto-corr-neg-a30": dict(corr_pass=True, neg="touch", neg_alpha=0.3),
    "proto-corr-neg-all": dict(corr_pass=True, neg="touch", touch_rule="all"),
    "proto-bm20": dict(bp_iters=20),
    "proto-bm20-noshort": dict(bp_iters=20, bp_converge_shortcut=False),
    "proto-bm1": dict(bp_iters=1, bp_converge_shortcut=False),
    "proto-bm2": dict(bp_iters=2, bp_converge_shortcut=False),
    "proto-bm3": dict(bp_iters=3, bp_converge_shortcut=False),
    "proto-bm5": dict(bp_iters=5, bp_converge_shortcut=False),
    "proto-bm20-corr": dict(bp_iters=20, corr_pass=True),
    "proto-bm3-corr": dict(bp_iters=3, corr_pass=True, bp_converge_shortcut=False),
}
