**Title:** Correlated decoding: syndrome-gated base weights give 7–33% fewer logical errors (d=5–11) at ~2x the cost

Hi! Thanks for PyMatching and for shipping correlated matching in 2.3.

While benchmarking decoders for the rotated surface code under SI1000 noise
(Stim `rotated_memory_z`, CX compiled to H·CZ·H, d = 5, 7, 9, p = 0.1–0.5%),
I found two changes to how edge weights are formed that reduce the logical
error rate of `enable_correlations=True` by a consistent margin. Both fit in
the existing two-pass structure, and I wanted to ask whether you would be
interested in a PR.

## 1. Syndrome-gated base weights (the main effect)

Today every decomposed component of a multi-edge error mechanism carries the
mechanism's probability in the base graph, and the second pass can only
*lower* weights. If a Y error's Z-part edge touches no detection event, that
Y error essentially did not happen, but its probability still lowers the
weight of its X-part edge in both passes.

The proposed change has three steps:

- **Base graph:** keep only single-component mechanisms in the base graph.
  This is the existing `include_decomposed_error_components_in_edge_weights = false`
  path in `user_graph.h`, which Python does not currently expose.
- **Per-shot restore:** for each mechanism whose component edges **all touch
  a detection event**, add its probability back to those edges. This is
  found by walking the edges incident to detection events, so it costs
  O(#detection events). It goes through the same `previous_weights` undo
  log as `reweight_for_edges`.
- **Passes:** run the first pass on these weights, then apply the existing
  implied-weight reweighting, then run the second pass.

Paired results against PyMatching 2.4.0 `enable_correlations=True`, on
identical shots (McNemar test on discordant shots):

| d | p | shots | ratio of logical errors (±2σ) | McNemar p | time vs uncorrelated |
|---|---|---|---|---|---|
| 5 | 0.3% | 400k | 0.924 ± 0.017 | 5e-20 | 4.9x |
| 5 | 0.5% | 400k | 0.934 ± 0.008 | 2e-61 | 4.6x |
| 7 | 0.3% | 400k | 0.886 ± 0.026 | 9e-19 | 5.1x |
| 7 | 0.5% | 400k | 0.920 ± 0.010 | 4e-62 | 4.8x |
| 9 | 0.2% | 2M | 0.859 ± 0.051 | 3e-8 | 6.6x |
| 9 | 0.3% | 400k | 0.835 ± 0.040 | 2e-16 | 5.9x |
| 9 | 0.5% | 400k | 0.887 ± 0.011 | 5e-97 | 4.8x |

For reference, `enable_correlations=True` itself is 2.5–3.3x uncorrelated on
the same machine.

The gain is not specific to that circuit. On Stim's stock `surface_code:rotated_memory_z` (plain CX gates,
uniform depolarizing circuit noise p on every location), paired the same way:

| d | p | shots | ratio of logical errors (±2σ) | McNemar p |
|---|---|---|---|---|
| 5 | 0.5% | 200k | 0.891 ± 0.028 | 7e-15 |
| 7 | 0.5% | 200k | 0.860 ± 0.041 | 5e-12 |
| 9 | 0.5% | 200k | 0.808 ± 0.060 | 1e-10 |
| 9 | 0.3% | 400k | 0.671 ± 0.152 | 1e-5 |

In the X basis (d = 7, p = 0.5%) the ratio is 0.822 ± 0.039, and on the SI1000 circuit at d = 11 it is
0.777 ± 0.063 (p = 0.3%) and 0.864 ± 0.017 (p = 0.5%). The full table is in `results/robustness.md`.

At d = 9, p = 0.3%, an ablation separates the pieces:

| configuration | errors |
|---|---|
| correlated (current) | 1210 |
| hyperedge-free base, no gate | 1141 |
| gate in second pass only | 1054 |
| gate in both passes | 1010 |

## 2. One decomposition per symptom (small effect, about 2%)

Stim can emit several DEM instructions with the same detector set but
different decompositions. For example, it emits `error(0.038) D1 D29`
alongside `error(0.0013) D29 ^ D1`, a Y-type error split into its X and Z
parts. Each decomposition is applied separately, so the second instruction
adds probability to the boundary edges of D1 and D29, edges it never flips
alone. Merging instructions by symptom and using the decomposition with the
fewest components reduced logical errors by about 2% in both uncorrelated and
correlated modes. That is p ≈ 1e-6 at 400k shots, d = 5–9, p = 0.5%.

## Code

A standalone implementation on top of your sparse-blossom core (vendored
unmodified, Apache-2.0, with attribution) is at
https://github.com/JimmyCarvalho/evmatch, together with the circuits,
benchmark scripts and raw data. If this direction interests you, I can open
a PR that adds it as an option of the correlated mode, for example
`enable_correlations="gated"` or a separate flag, with tests. Happy to adapt
to whatever API you prefer.
