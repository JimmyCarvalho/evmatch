"""Summarize paired runs: LER ratio vs a reference decoder with +-2 sigma from discordant shots.

On identical shots, errors_B = errors_A - better + worse, so
ratio = errors_B / errors_A and Var(errors_B - errors_A) ~ better + worse.

    python bench/paired_summary.py results/ablation.jsonl results/lowp.jsonl --decoder nat-evidence-both evmatch
"""

from __future__ import annotations

import argparse
import csv
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--decoder", nargs="+", required=True)
    ap.add_argument("--timing", default="results/timing.csv")
    args = ap.parse_args()

    times = {}
    try:
        with open(args.timing) as f:
            for r in csv.DictReader(f):
                times[(r["decoder"], int(r["d"]), float(r["p"]))] = float(r["us_per_shot"])
    except FileNotFoundError:
        pass

    print("| d | p | shots | errors, PyMatching correlated | errors, evmatch | ratio ± 2σ | McNemar p | time / PyMatching |")
    print("|---|---|---|---|---|---|---|---|")
    for path in args.jsonl:
        rows = [json.loads(line) for line in open(path)]
        for r in rows:
            if r["decoder"] not in args.decoder:
                continue
            ref = next(x for x in rows if x["d"] == r["d"] and x["p"] == r["p"] and x["better_than_ref"] == 0
                       and x["worse_than_ref"] == 0 and x["shots"] == r["shots"])
            n_ref = ref["errors"]
            ratio = r["errors"] / n_ref
            err = 2 * math.sqrt(r["better_than_ref"] + r["worse_than_ref"]) / n_ref
            t = times.get(("nat-evidence-both", r["d"], r["p"])) or times.get(("evmatch", r["d"], r["p"]))
            t_pm = times.get(("pymatching", r["d"], r["p"]))
            tr = f"{t / t_pm:.1f}x" if t and t_pm else "—"
            print(f"| {r['d']} | {r['p'] * 100:g}% | {r['shots']:,} | {n_ref} | {r['errors']} | "
                  f"{ratio:.3f} ± {err:.3f} | {r['mcnemar_p']:.0e} | {tr} |")


if __name__ == "__main__":
    main()
