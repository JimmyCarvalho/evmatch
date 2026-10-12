"""Detector error model -> error mechanisms and matching-graph edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass
class DemGraph:
    num_dets: int
    num_obs: int
    # Error mechanisms ("hyperedges"), merged when they flip the same detectors.
    h_dets: list[tuple[int, ...]]
    h_obs: np.ndarray  # (num_h,) uint64 observable mask
    h_prior: np.ndarray  # (num_h,)
    h_edges: list[tuple[int, ...]]  # decomposition into graph edges (edge ids)
    # Graph edges: (u, v) with v == -1 for a boundary edge.
    e_nodes: np.ndarray  # (num_e, 2) int64
    e_obs: np.ndarray  # (num_e,) uint64
    e_prior: np.ndarray  # (num_e,) independent combination of mechanisms containing e
    edge_index: dict[tuple[int, int], int]


def _xor_all(parts):
    acc: set[int] = set()
    for part in parts:
        acc ^= set(part)
    return tuple(sorted(acc))


def build(dem: stim.DetectorErrorModel, decomposition: str = "fewest") -> DemGraph:
    """Merge error mechanisms by symptom (detector set) and decompose into edges.

    decomposition:
      "fewest": of the decompositions stim emitted for the symptom, the one
          with the fewest components (ties: the one carrying the most
          probability). Stim splits Y-type errors into X and Z parts even when
          another mechanism with the same symptom is a single edge; applying
          every decomposition (as PyMatching does) leaks probability onto
          edges the mechanism never flips alone, often boundary edges.
      "direct": like "fewest", but a symptom of <= 2 detectors is always its
          own edge, even if stim never offered it (this can join the X and Z
          matching graphs).
      "first": the first decomposition stim emitted for the symptom.
    """
    by_symptom: dict[tuple[int, ...], list] = {}
    order: list[tuple[int, ...]] = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        if p == 0:
            continue
        parts_d: list[list[int]] = [[]]
        parts_o: list[list[int]] = [[]]
        for t in inst.targets_copy():
            if t.is_separator():
                parts_d.append([])
                parts_o.append([])
            elif t.is_relative_detector_id():
                parts_d[-1].append(t.val)
            elif t.is_logical_observable_id():
                parts_o[-1].append(t.val)
        dets = _xor_all(parts_d)
        if not dets:
            continue  # undetectable
        obs_mask = 0
        for o in _xor_all(parts_o):
            obs_mask ^= 1 << o
        comps = []
        for pd, po in zip(parts_d, parts_o):
            if not pd:
                continue
            if len(pd) > 2:
                raise ValueError("undecomposed hyperedge; use decompose_errors=True")
            m = 0
            for o in po:
                m ^= 1 << o
            comps.append((tuple(sorted(pd)), m))
        if dets not in by_symptom:
            by_symptom[dets] = []
            order.append(dets)
        by_symptom[dets].append((p, obs_mask, comps))

    h_dets, h_obs, h_prior, h_edges = [], [], [], []
    edge_index: dict[tuple[int, int], int] = {}
    e_nodes, e_obs = [], []

    def edge_id(ds, m):
        key = (ds[0], ds[1]) if len(ds) == 2 else (ds[0], -1)
        if key not in edge_index:
            edge_index[key] = len(e_nodes)
            e_nodes.append(key)
            e_obs.append(m)
        return edge_index[key]

    for dets in order:
        mechs = by_symptom[dets]
        prior = 0.0
        for p, _, _ in mechs:
            prior = prior * (1 - p) + p * (1 - prior)
        obs_mask = mechs[-1][1]
        if decomposition == "first":
            comps = mechs[0][2]
        elif decomposition == "direct" and len(dets) <= 2:
            comps = [(dets, obs_mask)]
        else:
            mass: dict[tuple, float] = {}
            for p, _, cs in mechs:
                key = tuple(sorted(cs))
                mass[key] = mass.get(key, 0.0) + p
            comps = list(min(mass, key=lambda k: (len(k), -mass[k])))
        h_dets.append(dets)
        h_obs.append(obs_mask)
        h_prior.append(prior)
        h_edges.append(tuple(edge_id(ds, m) for ds, m in comps))

    ne = len(e_nodes)
    h_prior_arr = np.array(h_prior)
    # Independent combination of all mechanisms containing each edge.
    e_prior = np.zeros(ne)
    for k, es in enumerate(h_edges):
        for e in es:
            e_prior[e] = e_prior[e] * (1 - h_prior_arr[k]) + h_prior_arr[k] * (1 - e_prior[e])

    return DemGraph(
        num_dets=dem.num_detectors, num_obs=dem.num_observables,
        h_dets=h_dets, h_obs=np.array(h_obs, dtype=np.uint64), h_prior=h_prior_arr, h_edges=h_edges,
        e_nodes=np.array(e_nodes, dtype=np.int64).reshape(-1, 2), e_obs=np.array(e_obs, dtype=np.uint64),
        e_prior=e_prior, edge_index=edge_index,
    )
