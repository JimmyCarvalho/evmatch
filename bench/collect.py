"""Logical error rates for SI1000 surface-code memory, via sinter.

    python bench/collect.py --decoders pymatching pymatching-correlated \
        --distances 5 7 9 --ps 0.001 0.002 0.003 0.004 0.005 \
        --max-shots 1000000 --save results/stats.csv

Results are appended to the --save CSV (sinter resumes from it), so the run
can be interrupted and restarted. Every task has the same circuit for a
given (d, p), whatever the decoder.
"""

from __future__ import annotations

import argparse
import os
import sys

import sinter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surfdec.circuits import surface_code_si1000  # noqa: E402
from surfdec.registry import all_decoders  # noqa: E402


def tasks(distances, ps):
    for d in distances:
        for p in ps:
            yield sinter.Task(
                circuit=surface_code_si1000(d, p),
                json_metadata={"d": d, "p": p, "r": d, "noise": "SI1000", "b": "Z"},
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoders", nargs="+", required=True)
    ap.add_argument("--distances", nargs="+", type=int, default=[5, 7, 9])
    ap.add_argument("--ps", nargs="+", type=float, default=[0.001, 0.002, 0.003, 0.004, 0.005])
    ap.add_argument("--max-shots", type=int, default=1_000_000)
    ap.add_argument("--max-errors", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--save", required=True)
    args = ap.parse_args()

    sinter.collect(
        num_workers=args.workers,
        tasks=list(tasks(args.distances, args.ps)),
        decoders=args.decoders,
        custom_decoders=all_decoders(),
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        save_resume_filepath=args.save,
        print_progress=True,
    )


if __name__ == "__main__":
    main()
