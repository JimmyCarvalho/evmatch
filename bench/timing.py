"""Single-threaded decoding time per shot, on identical pre-sampled syndromes.

sinter's `seconds` mixes sampling and decoding across workers, so speed is
measured here instead: sample once, then time each decoder's
`decode_shots_bit_packed` on the same detection events, one thread.

    python bench/timing.py --decoders pymatching pymatching-correlated \
        --distances 5 7 9 --ps 0.001 0.003 0.005 --shots 20000
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from decoders import all_decoders
from evmatch.circuits import surface_code_si1000


def time_decoder(decoder, dem, dets_packed, obs_packed, repeats: int):
    t0 = time.perf_counter()
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    t_compile = time.perf_counter() - t0
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        pred = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets_packed)
        best = min(best, time.perf_counter() - t0)
    errors = int(np.any(pred != obs_packed, axis=1).sum())
    return best, t_compile, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoders", nargs="+", required=True)
    ap.add_argument("--distances", nargs="+", type=int, default=[5, 7, 9])
    ap.add_argument("--ps", nargs="+", type=float, default=[0.001, 0.003, 0.005])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    decoders = all_decoders()
    rows = []
    for d in args.distances:
        for p in args.ps:
            circuit = surface_code_si1000(d, p)
            dem = circuit.detector_error_model(decompose_errors=True)
            sampler = circuit.compile_detector_sampler(seed=args.seed)
            dets, obs = sampler.sample(args.shots, separate_observables=True, bit_packed=True)
            for name in args.decoders:
                secs, t_compile, errors = time_decoder(decoders[name], dem, dets, obs, args.repeats)
                row = dict(
                    decoder=name, d=d, p=p, shots=args.shots,
                    us_per_shot=1e6 * secs / args.shots,
                    us_per_round=1e6 * secs / args.shots / d,
                    compile_s=t_compile, errors=errors,
                )
                rows.append(row)
                print(
                    f"{name:28s} d={d} p={p:.4f} {row['us_per_shot']:10.2f} us/shot "
                    f"errors={errors}/{args.shots} compile={t_compile:.2f}s",
                    flush=True,
                )
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
