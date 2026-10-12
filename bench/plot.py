"""Logical error rate vs p, one panel per distance, from sinter CSVs.

    python bench/plot.py results/baseline_pm.csv results/main.csv --out results/ler.png
"""

from __future__ import annotations

import argparse
import collections

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sinter  # noqa: E402

# Fixed categorical order (validated reference palette, light mode).
STYLE = {
    "pymatching": ("PyMatching", "#2a78d6", "o"),
    "pymatching-correlated": ("PyMatching (correlated)", "#eb6834", "s"),
    "evmatch": ("This work (evidence-conditioned)", "#1baf7a", "D"),
    "beliefmatching": ("Belief-matching (reference)", "#eda100", "^"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=float, default=1000)
    args = ap.parse_args()

    stats = sinter.read_stats_from_csv_files(*args.csv)
    series = collections.defaultdict(list)
    for s in stats:
        if s.decoder in STYLE and s.errors > 0:
            series[(s.json_metadata["d"], s.decoder)].append(s)
    distances = sorted({d for d, _ in series})

    fig, axes = plt.subplots(1, len(distances), figsize=(4.2 * len(distances), 4.0), sharey=True)
    for ax, d in zip(axes, distances):
        for dec, (label, color, marker) in STYLE.items():
            pts = sorted(series.get((d, dec), []), key=lambda s: s.json_metadata["p"])
            if not pts:
                continue
            xs = [s.json_metadata["p"] for s in pts]
            fits = [sinter.fit_binomial(num_shots=s.shots, num_hits=s.errors, max_likelihood_factor=args.factor) for s in pts]
            ax.plot(xs, [f.best for f in fits], color=color, marker=marker, markersize=5, linewidth=2, label=label)
            ax.fill_between(xs, [f.low for f in fits], [f.high for f in fits], color=color, alpha=0.18, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"d = {d}", color="#0b0b0b")
        ax.set_xlabel("physical error rate p (SI1000)", color="#52514e")
        ax.grid(True, which="major", color="#e5e4df", linewidth=0.6)
        ax.grid(True, which="minor", color="#f0efec", linewidth=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("logical error rate per shot (d rounds)", color="#52514e")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
