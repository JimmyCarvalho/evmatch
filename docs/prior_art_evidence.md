# Prior art: syndrome-conditioned ("negative evidence") edge weights

Checked 2026-10-12. Everything below was read in the full text or source, and quotes are verbatim. **No published work applies our exact rule**: a multi-component mechanism counts toward an edge only if a partner component touches a detection event, decided locally before matching, with no BP. Every element of the rule exists somewhere, though.

## Closest matches

- **DGR, Wang et al., arXiv:2311.16214.** The only explicit two-sided rule we found: *"if one edge is not selected, the probability of the occurrence of the other correlated edges should be diminished, leading to an increase in weights."* The update is w̃_j = w_j − Σ_{i∈M} p(e_i,e_j)/p(e_i) + Σ_{i∉M} p(e_i,e_j)/p(e_i).
  - **How ours differs:** DGR conditions on the first-pass **matching**, not on detection events. Its correlations are counted across earlier shots instead of taken from the DEM. The update is an additive heuristic, and it costs an extra MWPM pass.
- **Harmony's correlated matching, arXiv:2401.12434.** The first pass has no hyperedges, the same as our base graph: *"we distribute the probabilities of edge-like error mechanisms to the corresponding X- and Z-type error graphs, ignoring the hyperedges."* In the second pass, a hyperedge contributes only when a partner edge was **matched**, using the conditional probability (max over candidates).
  - **How ours differs:** we gate on syndrome contact, add the prior p (not the conditional ≈0.5), and also apply the gate in the first pass.
- **PyMatching ≥2.3 (commit 6f63b2b).** The second pass only ever lowers weights: *"Only reweight if the new weight is lower than the current weight"* (`flooder/graph.h`). The implied probability is min(0.5, p_joint/p_marginal). The base graph includes the components of decomposed errors. The source also has a flag that Python does not expose: *"If include_decomposed_error_components_in_edge_weights is False, then only add the edge into the graph if it is not a component in a decomposed error with more than one component"* (`user_graph.h`). That flag removes those components statically, not per shot.
- **Google's real-time "preweight correlations", arXiv:2408.13687 (supplement).** Local, driven by detection events, with no first matching. It picks edges that have *"a detection event at both of its endpoints"*, and then *"we lower the edge weight corresponding to a Z error at the same circuit location."* It is positive-only.
- **Paler & Fowler, arXiv:2205.09828 (also patent US12026589B1).** Each detection event is paired with its mutual nearest neighbour ("virtual pre-matching"), and the correlated edges receive *"pf = pe + pc"*. Local and positive-only.
- **Grans-Samuelsson et al., arXiv:2310.12981, §7.2.** The edges of a split fault *"are assigned weight zero whenever j is present in the observed syndrome"*, where j is the partner's detector. This is per shot and gated by the syndrome, so the mechanism is close to ours. But it is positive-only: when the partner detector is silent, the edges keep their static weight.
- **Belief-matching, arXiv:2203.04948.** *"the posterior marginal probability of each hyperedge is added to the marginal probabilities of the edges in its decomposition."* BP posteriors include negative evidence implicitly, so this is the principled version of our idea. It is global and iterative, which makes it costlier, and convergence is not guaranteed.

## Positive-only and conditioned on matching

- **Fowler 2013, arXiv:1310.0863.** *"If we choose to believe with certainty that one of these errors occurred, we can uniformly scale up the probability of each of these errors so that they total to 1."* The base weights include *"two-qubit correlated errors"*. It never raises weights when the partner was not matched.
- **Delfosse & Tillich, arXiv:1401.6975.** The X-decoded qubits become erasures for the Z decoder. The paper does write the negative conditional P(E_Z=Z | E_X=I) = (p/3)/(1−2p/3), but that value is the same for every non-erased qubit, so it has no effect on the matching. Code-capacity setting.
- **Yuan & Lu, arXiv:2202.11239, and IRMWPM, arXiv:2509.06756.** Both alternate between the X and Z lattices, conditioning each on the other's matching. IRMWPM: *"an edge conditioned on the presence of an error in the dual lattice is reweighted to 0."* It takes 2–4 MWPM passes.
- **Tesseract multi-pass (commit 248e6f7).** Errors predicted in pass 1 raise the pass-2 priors to p_paired/p_marginal (capped at 0.499), and only *"if (reweight_probability > rule.original_probability)"*. Priors only go up. The base priors are XOR marginals over all mechanisms.

## Checked and unrelated

- **Libra, arXiv:2408.12135.** Random Lognorm(0, σ²) perturbation of the DEM probabilities.
- **Harmony's ensemble.** Uniform random perturbation of the priors.
- **Decoder switching, Toshio et al., arXiv:2510.25222.** Its claims check out: MWPM or UF with a complementary-gap output as the weak decoder, belief-matching as the strong one, and P_weak(e|g) < P_strong(e|g) at about 20–30 dB. It does no reweighting.
- **Majaniemi & Matekole, arXiv:2504.03504.** Per-shot weights only on measurement-error edges, taken from analog readout.
- **Iolius et al. review, arXiv:2307.14989.** Only generic one-way reweighting.
- **NMWPM, arXiv:2601.00242.** Learned per-syndrome weights, tested on the toric code under depolarizing noise.

## Assessment

Already published:

- **A hyperedge-free base graph:** Harmony, and PyMatching's hidden flag.
- **Local reweighting driven by detection events:** Google, Paler–Fowler, and Grans-Samuelsson. All of them are positive-only.
- **Negative evidence:** only DGR (conditioned on the matching) and belief-matching (BP).

**Defensible novelty:** a deterministic gate on syndrome contact, applied before the first pass, costing O(#detection events), and built from the DEM alone.

**Expected baselines:**

- belief-matching;
- correlated matching with a hyperedge-free first pass (Harmony-style);
- DGR's two-sided update.
