"""Table of logical error rates with confidence intervals from sinter CSVs.

    python bench/table.py results/baseline_pm.csv results/main.csv

Intervals are sinter.fit_binomial with max_likelihood_factor=1000 by default
(hypotheses up to 1000x less likely than the max-likelihood estimate), the
convention used in sinter's plots; pass --factor to change it.
"""

from __future__ import annotations

import argparse
import collections

import sinter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--factor", type=float, default=1000)
    ap.add_argument("--ref", default="pymatching-correlated", help="ratios are relative to this decoder")
    args = ap.parse_args()

    stats = sinter.read_stats_from_csv_files(*args.csv)
    by_point = collections.defaultdict(dict)
    for s in stats:
        m = s.json_metadata
        by_point[(m["d"], m["p"])][s.decoder] = s
    decoders = sorted({s.decoder for s in stats})
    print(f"| d | p | decoder | shots | errors | LER | CI low | CI high | LER / {args.ref} |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (d, p) in sorted(by_point):
        row = by_point[(d, p)]
        ref = row.get(args.ref)
        ref_ler = ref.errors / ref.shots if ref and ref.errors else None
        for dec in decoders:
            s = row.get(dec)
            if s is None:
                continue
            fit = sinter.fit_binomial(num_shots=s.shots - s.discards, num_hits=s.errors, max_likelihood_factor=args.factor)
            ler = s.errors / s.shots
            ratio = f"{ler / ref_ler:.3f}" if ref_ler else "—"
            print(f"| {d} | {p:g} | {dec} | {s.shots} | {s.errors} | {ler:.3e} | {fit.low:.2e} | {fit.high:.2e} | {ratio} |")


if __name__ == "__main__":
    main()
