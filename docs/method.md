# Evidence-conditioned correlated matching

Working notes on the method. Numbers here are development measurements
(paired tests, `bench/paired.py`); headline numbers come from
`bench/collect.py` (sinter).

## Setting

A detector error model (DEM) from Stim lists independent error mechanisms
`h`, each with a probability `p_h` and a symptom (the set of detectors it
flips). With `decompose_errors=True`, each symptom with more than two
detectors is split into "components", each one an edge of a matching graph:
a Y error on a data qubit becomes an X-graph edge plus a Z-graph edge, some
two-qubit gate errors become two edges in the same graph, and so on.

Matching decoders put each mechanism's probability on every one of its
component edges, as if the components were independent, then look for the
most likely set of edges explaining the detection events.

## Two problems with the standard graph

**1. Decomposition leakage.** Stim decomposes a mechanism into its X and Z
parts whenever it can, even when another mechanism with the same symptom is
a single edge. PyMatching applies every decomposition it is given, so a
mechanism whose symptom is `{a, b}` can add its probability to the boundary
edges of `a` and of `b` separately, edges it never flips alone. We merge
mechanisms by symptom and use the decomposition with the fewest components
("fewest"). Effect alone: about 1–3% lower LER than PyMatching, small but
consistent.

**2. Probability that the syndrome rules out.** A multi-component
mechanism can only have occurred if each of its components is "explained" by
the syndrome. If a Y error's Z part would flip X-type detectors and none of
them fired, that Y error essentially did not happen (it would need a second
error cancelling exactly those detection events, which is second order).
Yet its probability still lowers the weight of its X part in the matching
graph, making paths through it look more likely than they are. Correlated
matching (Fowler 2013; PyMatching `enable_correlations`) adds *positive*
evidence after a first pass, never this *negative* evidence.

## The rule

Precompute two graphs on the same edges:

- `full`: each edge carries every mechanism containing it (PyMatching's
  weights, with the "fewest" decomposition);
- `single`: each edge carries only single-component mechanisms (edges with
  none get a floor probability, 1e-9).

Per shot, with detection events `D`:

1. **Evidence.** A component edge is *touched* if one of its endpoints is in
   `D`. Every multi-component mechanism whose components are all touched is
   *active*; restore its probability on its component edges of `single`
   (independent XOR combination). Active mechanisms are found by walking the
   edges incident to detection events, so this is O(|D|) work.
2. **First pass.** Match on `single` + active mechanisms and trace the
   matched paths to edges (PyMatching's search flooder).
3. **Correlations.** For each edge `e1` on the first-pass paths, every edge
   `e2` sharing a mechanism with it gets weight
   `min(w(e2), log((1 - q)/q))`, `q = min(1/2, P(e1 and e2) / P(e1))`, which
   is PyMatching's rule with its precomputed joint probabilities.
4. **Second pass.** Match again; report the observable flips.
5. Restore the overwritten weights (undo log).

Variants measured: evidence only in the second pass, with a first pass on
`full` ("evidence-correlated"); evidence in both passes ("evidence-both",
the default); evidence alone, one pass ("evidence"); requiring evidence only
for the *other* components ("partners only", no measurable difference).

## Why not belief propagation?

Belief-matching gets its gain from BP marginals, which also encode the
negative evidence above. BP however also lowers the probability of edges in
the interior of a chain of errors (their detectors did not fire, because the
chain flips them twice), and needs many iterations over the whole graph:
with 3 iterations instead of 20 the gain over correlated matching
disappeared in our tests. The touch rule keeps exactly the part of the
conditioning that matters for matching and costs O(|D|).

## Cost

Two matching passes plus path tracing, as in PyMatching's correlated mode,
plus the O(|D|) evidence walk and the weight writes. Measured at 4–5x
PyMatching (uncorrelated) per shot, i.e. about 1.5–2x PyMatching's
correlated mode.
