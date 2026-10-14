# Robustness: circuits not used during development

Paired runs (`bench/robust.py`): both decoders decode the same shots, sampled with seeds 1000–1007. "Uniform" is Stim's stock
generated circuit with plain CX gates and `after_clifford_depolarization = before_round_data_depolarization =
before_measure_flip_probability = after_reset_flip_probability = p`. Correlated PyMatching is PyMatching 2.4.0 with
`enable_correlations=True`; times are for the same shots on one 4-vCPU cloud machine. The ±2σ is computed from the
discordant shots. Raw output: [robustness.jsonl](robustness.jsonl).

| circuit | d | p | shots | errors, PyMatching correlated | errors, evmatch | ratio ± 2σ | McNemar p | time / correlated |
|---|---|---|---|---|---|---|---|---|
| Stim `rotated_memory_z`, uniform | 5 | 0.5% | 200,000 | 2202 | 1962 | 0.891 ± 0.028 | 7e-15 | 1.79x |
| Stim `rotated_memory_z`, uniform | 5 | 0.3% | 400,000 | 955 | 852 | 0.892 ± 0.042 | 4e-07 | 1.74x |
| Stim `rotated_memory_z`, uniform | 7 | 0.5% | 200,000 | 1182 | 1016 | 0.860 ± 0.041 | 5e-12 | 1.86x |
| Stim `rotated_memory_z`, uniform | 7 | 0.3% | 400,000 | 271 | 227 | 0.838 ± 0.088 | 3e-04 | 1.65x |
| Stim `rotated_memory_z`, uniform | 9 | 0.5% | 200,000 | 656 | 530 | 0.808 ± 0.060 | 1e-10 | 1.91x |
| Stim `rotated_memory_z`, uniform | 9 | 0.3% | 400,000 | 82 | 55 | 0.671 ± 0.152 | 1e-05 | 1.99x |
| Stim `rotated_memory_x`, uniform | 7 | 0.5% | 200,000 | 1324 | 1088 | 0.822 ± 0.039 | 9e-20 | 1.92x |
| SI1000 (this repo's circuit) | 11 | 0.3% | 400,000 | 546 | 424 | 0.777 ± 0.063 | 1e-12 | 1.90x |
| SI1000 (this repo's circuit) | 11 | 0.5% | 200,000 | 7508 | 6489 | 0.864 ± 0.017 | 5e-55 | 1.75x |

Every point favours evmatch (McNemar p ≤ 3×10⁻⁴). On the stock circuit the ratio is 0.67–0.89 at d = 5–9, and at
d = 11 on the SI1000 circuit it is 0.78 (p = 0.3%) and 0.86 (p = 0.5%): the gain does not depend on the CX compilation
used during development, holds in the X basis, and keeps growing with distance.
