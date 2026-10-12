"""Paired comparison of decoders on identical shots (development tool).

Every decoder decodes the same sampled shots. Besides each decoder's logical
error rate, the report counts discordant shots against the first decoder
(the reference): `worse` = shots this decoder gets wrong and the reference
gets right, `better` = the reverse. Under the null hypothesis of equal error
rates, better ~ Binomial(better + worse, 1/2) (McNemar's exact test), which
resolves much smaller differences than two independent confidence intervals.

The headline results are produced with bench/collect.py (sinter); this is
for fast iteration.

    python bench/paired.py --decoders pymatching-correlated pymatching \
        --d 7 --p 0.004 --shots 100000
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evmatch.circuits import surface_code_si1000  # noqa: E402
from decoders import all_decoders  # noqa: E402


def _binom_two_sided_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = min(k, n - k)
    # P(X <= k) for Binomial(n, 1/2), doubled.
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(x - m) for x in logs)
    return min(1.0, 2 * tail)


def _worker(job):
    d, p, shots, seed, names = job
    circuit = surface_code_si1000(d, p)
    dem = circuit.detector_error_model(decompose_errors=True)
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True, bit_packed=True)
    decs = all_decoders()
    fails, secs = {}, {}
    for name in names:
        compiled = decs[name].compile_decoder_for_dem(dem=dem)
        t0 = time.perf_counter()
        pred = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
        secs[name] = time.perf_counter() - t0
        fails[name] = np.any(pred != obs, axis=1)
    return fails, secs


def run(names, d, p, shots, chunk, seed, workers):
    jobs = []
    k = 0
    while k * chunk < shots:
        jobs.append((d, p, min(chunk, shots - k * chunk), seed + k, names))
        k += 1
    fails = {n: [] for n in names}
    secs = {n: 0.0 for n in names}
    with mp.get_context("fork").Pool(workers) as pool:
        for f, s in pool.imap_unordered(_worker, jobs):
            for n in names:
                fails[n].append(f[n])
                secs[n] += s[n]
    fails = {n: np.concatenate(v) for n, v in fails.items()}
    ref = names[0]
    report = []
    for n in names:
        f = fails[n]
        worse = int(np.sum(f & ~fails[ref]))
        better = int(np.sum(~f & fails[ref]))
        report.append(dict(
            decoder=n, d=d, p=p, shots=int(f.size), errors=int(f.sum()),
            ler=float(f.mean()), us_per_shot=1e6 * secs[n] / f.size,
            better_than_ref=better, worse_than_ref=worse,
            mcnemar_p=_binom_two_sided_p(better, better + worse),
        ))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoders", nargs="+", required=True, help="first one is the reference")
    ap.add_argument("--d", type=int, nargs="+", default=[7])
    ap.add_argument("--p", type=float, nargs="+", default=[0.004])
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--jsonl", default=None, help="append results here")
    args = ap.parse_args()
    for d in args.d:
        for p in args.p:
            t0 = time.time()
            rep = run(args.decoders, d, p, args.shots, args.chunk, args.seed, args.workers)
            print(f"--- d={d} p={p} shots={args.shots} wall={time.time() - t0:.0f}s")
            for r in rep:
                print(
                    f"{r['decoder']:34s} LER={r['ler']:.3e} ({r['errors']:6d})  "
                    f"{r['us_per_shot']:9.1f} us/shot  better={r['better_than_ref']:5d} "
                    f"worse={r['worse_than_ref']:5d}  p={r['mcnemar_p']:.2g}",
                    flush=True,
                )
            if args.jsonl:
                with open(args.jsonl, "a") as fh:
                    for r in rep:
                        fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
