"""Paired robustness check of evmatch against correlated PyMatching on circuits not used during development.

    python bench/robust.py uniform 7 0.005 200000 8

kind d p shots [chunks]
kind: si1000 (evmatch's own circuit) | uniform (Stim's stock rotated_memory_z, CX gates, uniform depolarizing
circuit noise) | uniform_x (same, X basis)"""
import json, math, sys, time
import multiprocessing as mp
import numpy as np
import stim, pymatching, evmatch
from evmatch.circuits import surface_code_si1000

def circuit(kind, d, p):
    if kind == "si1000":
        return surface_code_si1000(d, p)
    task = "surface_code:rotated_memory_x" if kind == "uniform_x" else "surface_code:rotated_memory_z"
    return stim.Circuit.generated(task, distance=d, rounds=d, after_clifford_depolarization=p,
                                  before_round_data_depolarization=p, before_measure_flip_probability=p,
                                  after_reset_flip_probability=p)

def work(args):
    kind, d, p, shots, seed = args
    c = circuit(kind, d, p)
    dem = c.detector_error_model(decompose_errors=True)
    det, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    pm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    t = time.time(); a = pm.decode_batch(det, enable_correlations=True); ta = time.time() - t
    ev = evmatch.EvMatchDecoder().compile_decoder_for_dem(dem=dem)
    t = time.time(); b = ev.decode_batch(det); tb = time.time() - t
    ea = np.any(a != obs, axis=1); eb = np.any(np.asarray(b, dtype=bool) != obs, axis=1)
    return int(ea.sum()), int(eb.sum()), int((ea & ~eb).sum()), int((eb & ~ea).sum()), ta, tb

def binom_p(k, n):
    if n == 0: return 1.0
    k = min(k, n - k)
    logs = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2) for i in range(k+1)]
    m = max(logs); return min(1.0, 2*math.exp(m)*sum(math.exp(x-m) for x in logs))

if __name__ == "__main__":
    kind, d, p, shots = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    chunks = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    per = shots // chunks
    with mp.Pool(min(chunks, 4)) as pool:
        res = pool.map(work, [(kind, d, p, per, 1000 + i) for i in range(chunks)])
    A = sum(r[0] for r in res); B = sum(r[1] for r in res); better = sum(r[2] for r in res); worse = sum(r[3] for r in res)
    ta = sum(r[4] for r in res); tb = sum(r[5] for r in res)
    n = better + worse
    ratio = B / A if A else float("nan")
    # 2-sigma on the ratio from the discordant counts (delta method on B/A with A fixed approx)
    sig = 2 * math.sqrt(better + worse) / A if A else float("nan")
    print(json.dumps({"kind": kind, "d": d, "p": p, "shots": per * chunks, "pm_corr": A, "evmatch": B,
                      "ratio": round(ratio, 3), "pm2sigma": round(sig, 3), "evm_better": better, "evm_worse": worse,
                      "mcnemar_p": binom_p(min(better, worse), n), "time_ratio_vs_pmcorr": round(tb / ta, 2)}), flush=True)
