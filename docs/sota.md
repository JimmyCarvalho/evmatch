# Decoder survey: rotated surface code, circuit-level noise (as of 2026-10-12)

Goal: an open decoder that beats PyMatching 2 in logical error rate (LER), approaches belief-matching, runs at no more than 10x PyMatching time, and works with sinter. Target setting: SI1000, d = 5/7/9, p = 0.1–0.5%.

Each claim below was checked against the cited abstract, full text, README or PyPI page. "Not found" means I searched and found nothing. It does not prove the thing doesn't exist.

## 1. AlphaQubit and its successor (Google DeepMind / Google Quantum AI)

- **AlphaQubit 1.** Bausch et al., *Nature* 635, 834 (2024), doi:10.1038/s41586-024-08148-8, arXiv:2310.05900.
  - **Method:** a recurrent transformer decoder.
  - **Accuracy:** on Sycamore d=5 hardware data it made 6% fewer errors than a tensor-network decoder and 30% fewer than correlated matching ([Google blog](https://blog.google/technology/google-deepmind/alphaqubit-quantum-error-correction/)). It keeps its advantage in simulation up to d=11.
  - **Speed:** not real-time; Toshio et al. (arXiv:2510.25222) put its decoding time above 10 µs at d=3.
- **AlphaQubit 2 (AQ2) and its real-time variant AQ2-RT.** Senior, …, Higgott, …, Bausch, arXiv:2512.07737 (v2, Mar 2026).
  - **Setup:** surface-code results use SI1000 at p=0.15% with 120-cycle memory experiments.
  - **Accuracy:** at d=23, AQ2 reaches 7.3e-11 logical error per cycle. That is "close to" Libra and "substantially" better than both correlated and uncorrelated PyMatching; the text gives no fold factor. At d=11, AQ2-RT is 3.73e-6 against 1.89e-6 for full AQ2.
  - **Speed:** AQ2-RT runs under 1 µs per cycle up to d=11 on Trillium TPUs. Full AQ2 runs under 100 µs per cycle up to d=27, about 9.6x faster than AQ1 at d=11.
  - **Availability:** the paper lists PyMatching 2.3.1 and Tesseract as baselines and calls Libra "in-house". I found no statement that AQ2 code or weights are released.

## 2. Correlated matching

- **Fowler 2013,** arXiv:1310.0863. Exploits correlations in the error model "in an asymptotically optimal manner", parallelizable to O(1) per unit area. The abstract gives no threshold number. PyMatching's docstring cites this paper for its two-pass algorithm.
- **Paler & Fowler,** *Quantum* 7, 1205 (2023), arXiv:2205.09828. A pipelined version with a simpler reweighting step.
- **PyMatching ≥ 2.3 ships correlated matching** (README; release notes; docstrings checked in the installed 2.4.0).
  - **Release history (PyPI):** v2.3.0 (2025-08-10), v2.3.1 (2025-09-25, fixes a correlated-mode segfault and incomplete reweight undo), v2.4.0 (2026-05-22). The commits between 2.3.1 and 2.4.0 visible on GitHub touch only packaging and CI (Python 3.14 wheels, PyPI automation); the code diff did not load, so code changes are not ruled out.
  - **Method:** "two-pass": sparse blossom runs once, edges correlated with the predicted edges are reweighted, then sparse blossom runs again. It needs a DEM built with `decompose_errors=True`. Primary contributor: Sid Madhuk. I found no paper on it.
  - **API:** `Matching.from_detector_error_model(dem, enable_correlations=True)` (also on `from_stim_circuit` and the constructor), then `decode`/`decode_batch`/`decode_to_edges_array(..., enable_correlations=True)`. The CLI flag is `--enable_correlations`.
  - **Sinter:** sinter 1.16 has a built-in `pymatching-correlated` decoder and requires PyMatching ≥ 2.3.1.
  - **Reported cost:** "around 2x–3x slower than uncorrelated" (v2.3.0 notes). NVIDIA's tables (arXiv:2604.12841) at d=13 give 3.38 → 9.14 µs/round at p=0.3% and 9.97 → 21.51 µs/round at p=0.6%.
  - **Reported accuracy:** the README shows 8 vs 3 failures in 1000 shots at d=5. There is no systematic published benchmark.
- **My local check (not literature; uniform depolarizing via `stim.Circuit.generated`, not SI1000, small samples):**

  | d | p | PyMatching vs correlated: LER ratio | Slowdown | Failures (PM / corr / belief-matching) |
  |---|---|---|---|---|
  | 5 | 0.3% | 1.39x | 2.1x | 240 / 168 / 161 in 60k shots |
  | 7 | 0.3% | 2.17x | 2.4x | 78 / 27 / 37 in 40k shots |
  | 9 | 0.3% | 2.9x | 3.0x | 61 vs 21 in 100k shots (correlated only) |

  On these runs, correlated PyMatching was as accurate as `beliefmatching`, which was about 1000x slower than PyMatching. **This must be repeated under SI1000.**

## 3. Ensembles

- **Harmony.** Shutty, Newman, Villalonga, arXiv:2401.12434.
  - **Method:** an ensemble of correlated-matching decoders, each given perturbed priors, combined by "harmonization".
  - **Accuracy:** an ensemble of 3 already beats correlated matching; about 100 members captures most of the gain. A layered scheme runs N₁≈4 first and a 100-member ensemble only when the first pass disagrees.
  - **Setup:** a Stim noise model at p=0.4% (not SI1000).
  - **Availability:** no code found.
- **Libra / matching synthesis.** C. Jones, arXiv:2408.12135.
  - **Method:** combines the locally best parts of many perturbed correlated-matching solutions.
  - **Accuracy:** about 10% higher Λ; the gain saturates around 60 members.
  - **Availability:** no code found ("in-house" per AQ2).
- **Willow paper.** "QEC below the surface code threshold", *Nature* 638, 920 (2025), arXiv:2408.13687.
  - **Offline decoders:** neural network, Λ=2.14; ensembled matching synthesis (Libra, 51 members), Λ=2.04; Harmony (101 members), with 1.14x lower LER than correlated matching with an SI1000 prior at d=5.
  - **Real-time decoder:** a specialized sparse-blossom variant, Λ=2.0, with 63 µs average latency at d=5.

## 4. Tesseract (Google)

- Aghababaie Beni, Higgott, Shutty, arXiv:2503.10988.
  - **Method:** most-likely-error decoding by A* search with pruning and beam search, run on the full hypergraph DEM.
  - **Accuracy and speed vs integer programming:** at p ≤ 0.1% it is about as accurate as an integer-programming decoder and about 5x faster. Surface-code runs use SI1000 at p ∈ {0.05, 0.1, 0.2}%.
  - **Accuracy vs PyMatching:** the paper has no direct comparison.
- Grbic et al., arXiv:2602.02985: low-level optimizations give about 2x speedup.
- **Availability:** [quantumlib/tesseract-decoder](https://github.com/quantumlib/tesseract-decoder), Apache-2.0. PyPI `tesseract-decoder` publishes dev builds only (latest 0.1.1.dev20261012). Sinter integration exists (`TesseractSinterDecoder`, `make_tesseract_sinter_decoders_dict`). The README also documents "multi-pass graph shattering", an X/Z split with two-pass prior reweighting.
- **Independent data (low confidence):** arXiv:2609.04557 (Sep 2026; p not stated; 5k shots) reports these SI1000 per-cycle LERs at d=5 and d=7, using uncorrelated MWPM only:

  | Decoder | d=5 | d=7 |
  |---|---|---|
  | Tesseract | 1.2e-4 | 1e-5 |
  | BeliefMatching | 1.6e-4 | 3e-5 |
  | MWPM | 2.8e-4 | 5e-5 |

  Median Willow latency at d=5 was 49.8 ms (Tesseract) vs 0.007 ms (PyMatching), roughly 7000x.

## 5. BP-based decoders

- **Belief-matching.** Higgott et al., *PRX* 13, 031007 (2023), arXiv:2203.04948.
  - **Method:** BP on the full circuit-level Tanner graph (`m_iter`=30); the posterior marginals set the matching weights.
  - **Threshold:** 0.817% (MWPM) → 0.940% (1.15x) under *standard circuit-level depolarizing* noise, not SI1000. Belief-find (BP then weighted union-find) reaches 0.937%.
  - **Below threshold:** I found no below-threshold LER ratio for the CSS surface code in the text.
  - **Data:** no data or circuit release (no Zenodo or GitHub data link) found in the arXiv text.
  - **Package:** PyPI `beliefmatching` 0.2.0 (2025-06), sinter-compatible (`BeliefMatchingSinterDecoder`). The README says it is ">100x slower" than PyMatching because it is unoptimized and uses parallel-schedule BP from `ldpc`.
- **ldpc** (Roffe), PyPI 2.4.1 (2025-12).
  - **Decoders:** BP+OSD, BP+LSD (arXiv:2406.18655), union-find, and BeliefFind.
  - **Sinter:** wrappers in `ldpc.sinter_decoders` for BP+OSD, LSD and BeliefFind (checked in the installed package).
- **Relay-BP** (IBM), Müller et al., arXiv:2506.01779; PyPI `relay-bp` 0.2.2 (Rust, Stim/sinter support).
  - **Surface code:** tested on the rotated surface code at d=11 under uniform circuit noise. It is "comparable to Matching" (PyMatching) and slower than PyMatching on CPU. The authors "would not advocate Relay-BP for real-time surface code decoding."
- **Other BP variants:**
  - Ambiguity Clustering (Riverlane, arXiv:2406.14527): tested on BB codes only, no surface-code results.
  - BP+OTF (arXiv:2409.01440): qLDPC.
  - Standalone BP on the decoding graph (arXiv:2603.05381): thresholds "close to MWPM".
  - Soft-information belief-matching (arXiv:2504.03504): checked at abstract level only.
- **Local Clustering Decoder** (Riverlane), *Nat. Commun.* 16, 11048 (2025), arXiv:2411.10343: an FPGA decoder (under 1 µs per round) that adapts to heralded leakage. The abstract makes no LER-vs-MWPM claim. I did not check whether it is open-source.

## 6. Other neural decoders (2025–26)

- **NVIDIA "Ising" pre-decoders.** Chamberland et al., arXiv:2604.12841; code at [nvidia/ising-decoding](https://github.com/nvidia/ising-decoding).
  - **Method:** a 3D CNN cleans up local errors, then PyMatching decodes what remains.
  - **Setup:** uniform depolarizing noise (not SI1000).
  - **Accuracy:** 0.9–2.4x lower LER than uncorrelated PyMatching at d=5–13 (p=0.3–0.6%), depending on the model. A larger model beats correlated PyMatching up to d=13; no fold factors are given.
  - **Speed:** on a GB300 GPU, the end-to-end speedup over correlated PyMatching (CPU) at d=13 is 0.66x at p=0.3% (i.e. slower) and 1.38x at p=0.6%.
  - **Software:** shipped in CUDA-Q QEC 0.6.
  - **Independent evaluation:** arXiv:2609.04557 found no advantage on Willow data, but with mismatched training; low confidence.
- **GNN decoders.**
  - Lange et al., *PRR* 7, 023181 (2025), arXiv:2307.01241: beats MWPM under circuit noise.
  - POLYMECHANON (arXiv:2610.01683): up to 25% fewer failures than correlated MWPM, at a cost of milliseconds per shot on a GPU.
- **NMWPM** (arXiv:2601.00242, ICML 2026): a GNN plus transformer predicts MWPM edge weights. Toric code with code-capacity noise only.
- **Not found:** a CPU neural decoder that beats correlated matching under SI1000 at d=5–9 within 10x of PyMatching's cost.

## 7. Speed-oriented matching and "between MWPM and belief-matching" results

- **Fusion Blossom** (arXiv:2305.08307): parallel MWPM; `fusion-blossom` on PyPI and in sinter.
- **Micro Blossom** (arXiv:2502.14787, ASPLOS'25): FPGA MWPM, 0.8 µs at d=13, p=0.1%.
- **Promatch** (arXiv:2404.03136), **Astrea** (ISCA'23) and **Clique** (arXiv:2208.08547, ASPLOS'23): real-time predecoders and approximations of MWPM. They target speed; none claims accuracy above MWPM.
- **HyperBlossom / MWPF** (Wu et al., arXiv:2508.04969; PyPI `mwpf`, sinter-compatible): hypergraph matching. 4.8x lower LER than MWPM at d=11, but under *code-capacity* noise; the circuit-level LER comparison is not in the abstract.
- **Decoder switching** (Toshio et al., arXiv:2510.25222): this is the closest existing "between MWPM and belief-matching at low cost" result.
  - **Method:** PyMatching runs first; shots with a small complementary gap go to belief-matching.
  - **Result:** LER within about 10% of belief-matching, with average time "on par with" MWPM.
  - **Setup:** uniform noise at p=0.1%, d=5–11. No code found.
- **K minimum-weight matchings** (Lin, *PRA* 112, 042436 (2025), arXiv:2510.06531): approaches maximum-likelihood decoding as K grows; no circuit-level SI1000 benchmark.

## 8. Reference numbers

- **Sparse blossom** (*Quantum* 9, 1600 (2025), arXiv:2303.15933).
  - **Noise model:** *standard circuit-level depolarizing* at p=0.1% (threshold about 0.71%), not SI1000. Single core of an Apple M1 Max.
  - **d=17:** 0.62 µs per round on average (97.4% of shots under 1 µs).
  - **d=29:** 3.5 µs per round.
  - **Release:** code is in PyMatching. I found no separate circuit or data release.
- **Belief-matching:** see §5 (0.817% → 0.940% threshold; standard depolarizing noise; no data release found).
- **SI1000 definition:** Gidney et al., "A Fault-Tolerant Honeycomb Memory", *Quantum* 5, 605 (2021), arXiv:2108.10457. Neither of the two reference papers above uses SI1000.

## Gap analysis

1. **The real baseline is `pymatching-correlated`, not plain PyMatching.**
   - It ships in PyMatching ≥ 2.3.1 and is built into sinter. It costs 2–3x PyMatching.
   - In my uniform-noise check it was already about as accurate as belief-matching at d=5 and d=7.
   - So "beats PyMatching, approaches belief-matching" is largely met by correlated PyMatching already. A credible contribution has to beat **correlated PyMatching under SI1000** within the 10x budget, which is only about 3–5x correlated PyMatching's time.
   - SI1000 changes the mix of error types (heavy measurement and reset noise, light single-qubit noise), so the correlated-vs-belief-matching ordering has to be measured, not assumed.
2. **Ideas that fit ≤10x on CPU and are known to beat correlated matching:**
   - **Small Harmony-style ensembles of correlated PyMatching.** 3–4 members beat correlated matching; the cost is about 6–12x PyMatching, borderline, but layering helps.
   - **Gap-gated escalation**, as in decoder switching. Escalate to a correlated-PyMatching ensemble or to Tesseract instead of belief-matching.
   Both ideas are published, but I found no open, sinter-compatible implementation of either. Releasing one is an engineering contribution, not a new scientific result.
3. **What would genuinely be novel** (I found no prior work):
   - A fast belief-matching-quality decoder in C++ or Rust: a few serial-BP iterations, or Relay-BP marginals, feeding *correlated* matching instead of plain matching.
   - Multi-pass correlated reweighting driven by soft information (gap, ensemble disagreement) rather than a single hard first pass.
   - An open SI1000 benchmark at d=5/7/9 covering PyMatching, correlated PyMatching, belief-matching, Tesseract and an ensemble, with LER and time. No such public leaderboard was found.
4. **Out of budget or out of scope:**
   - Tesseract and Libra: thousands of times slower than PyMatching, or closed.
   - AQ2: TPUs, no release found.
   - GPU neural pre-decoders: they need a GPU, and at d=13 they are only 0.66–1.38x as fast as correlated PyMatching.
   - Relay-BP alone: only matches PyMatching on the surface code.
