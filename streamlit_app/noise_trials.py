"""1000 Gaussian-noise trials per (t, sigma) cell, for one fixed code.

    uv run --group sage --group app python streamlit_app/noise_trials.py

Deliberately narrow, unlike sweep.py: one code (p=3, m=3, K=3, N=27), the same
deterministic codewords every trial, and only the noise redrawn. That isolates the
effect of the noise instead of mixing it with a change of code.

The noise is e ~ CN(0, sigma^2 I) over all N^2 entries -- white, so the covariance is
a scalar times the identity -- added to the bundle before Stage A.

Cost: a full t-sweep is ~18s of decoding, so 8 sigma x 1000 trials is ~40 core-hours
with the naive decoder. Two things make it tractable:

  * ell bracket   unbundle walks ell upward from 2 and succeeds at ell = t, so the
                  first t-2 attempts are wasted. Passing a bracket around t is 5-6x
                  faster and gives identical results (checked over t=1..17 x 5 sigma).
  * processes     one worker per core, each with its own Sage context.
"""

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core

OUT = Path(__file__).resolve().parent / "data"
P = int(os.environ.get("P", 3))
M = int(os.environ.get("M", 3))
K = int(os.environ.get("K", 3))
D = int(os.environ.get("D", 1))              # Guruswami-Sudan multiplicity
# the default code keeps the plain filename; others get their own
STEM = os.environ.get("STEM", "noise_trials" if (P, M, K, D) == (3, 3, 3, 1)
                      else f"noise_trials_{P}_{M}_{K}"
                           + (f"_d{D}" if D != 1 else ""))
TRIALS = int(os.environ.get("TRIALS", 1000))
CHUNK = int(os.environ.get("CHUNK", 50))
SIGMAS = [float(v) for v in os.environ["SIGMAS"].split(",")] \
    if os.environ.get("SIGMAS") else \
    [0.0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]

_CTX = None


def _init():
    global _CTX
    _CTX = core.build_context(P, M, K)


def _run_chunk(task):
    """One (t, sigma) cell over a slice of trial seeds."""
    t, si, sigma, lo, hi = task
    ctx = _CTX
    msgs = core.message_list(ctx, t)
    clean = core.bundle_of(ctx, msgs)
    h0 = core.histograms(ctx, clean)
    want = sorted(msgs)
    n_a = n_exact = n_corr = 0
    t0 = time.time()
    for seed in range(lo, hi):
        e = core.gaussian_noise(ctx, sigma, seed=seed * 1000 + si) if sigma else 0
        res = core.decode(ctx, clean + e, d=D,
                          ell_min=max(2, t - 3), ell_max=t + 5)
        n_a += bool(np.array_equal(res["hist"], h0))
        n_exact += res["messages"] == want
        n_corr += sum(1 for u in res["messages"] if u in want)
    return {"t": t, "sigma_i": si, "sigma": sigma, "n": hi - lo,
            "stage_a_ok": n_a, "exact": n_exact, "n_correct": n_corr,
            "sec": time.time() - t0}


def main():
    t_max = int(os.environ.get("T_MAX", 17))
    workers = int(os.environ.get("WORKERS", max(1, mp.cpu_count() - 1)))
    # Resume: a cell already carrying its full TRIALS is skipped, so a run that was
    # cut short (the grid takes hours, and t runs ascending so the cliff region is
    # last) can be continued rather than restarted.
    agg = {}
    prev = OUT / f"{STEM}.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        if old.get("trials") == TRIALS:
            # Match per cell on the sigma VALUE, not on the sigma list as a whole --
            # otherwise extending the sigma range throws away every trial already
            # computed, which at 120,000 decodes is hours of work.
            for r in old["rows"]:
                if r["sigma"] not in SIGMAS:
                    continue                  # a sigma that is no longer swept
                si = SIGMAS.index(r["sigma"])
                agg[(r["t"], si)] = {"t": r["t"], "sigma": r["sigma"], "n": r["n"],
                                     "stage_a_ok": r["stage_a_ok"],
                                     "exact": r["exact"],
                                     "n_correct": r["n_correct"]}
            done = {k for k, v in agg.items() if v["n"] >= TRIALS}
            print(f"resuming: {len(done)} cells already complete "
                  f"({sum(v['n'] for v in agg.values()):,} trials kept)", flush=True)

    tasks = [(t, si, sg, lo, min(lo + CHUNK, TRIALS))
             for t in range(1, t_max + 1)
             for si, sg in enumerate(SIGMAS)
             for lo in range(0, TRIALS, CHUNK)
             if agg.get((t, si), {}).get("n", 0) < TRIALS]
    # cheapest first, so partial results still cover the whole sigma range
    tasks.sort(key=lambda x: x[0])
    print(f"p={P} m={M} K={K} d={D} N={P**M}, t=1..{t_max}, {len(SIGMAS)} sigma, "
          f"{TRIALS} trials -> {len(tasks)} chunks on {workers} workers", flush=True)

    t_start = time.time()
    OUT.mkdir(exist_ok=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_chunk, tasks), 1):
            key = (r["t"], r["sigma_i"])
            a = agg.setdefault(key, {"t": r["t"], "sigma": r["sigma"], "n": 0,
                                     "stage_a_ok": 0, "exact": 0, "n_correct": 0})
            for f in ("n", "stage_a_ok", "exact", "n_correct"):
                a[f] += r[f]
            if i % 40 == 0 or i == len(tasks):
                done = sum(v["n"] for v in agg.values())
                print(f"  {i}/{len(tasks)} chunks, {done:,} trials, "
                      f"{time.time() - t_start:.0f}s", flush=True)
                (OUT / f"{STEM}.json").write_text(json.dumps(
                    {"p": P, "m": M, "K": K, "N": P ** M, "d": D,
                     "u_max": (P ** M - 1) // (K - 1), "trials": TRIALS,
                     "sigmas": SIGMAS, "t_max": t_max,
                     "complete": i == len(tasks),
                     "rows": sorted(agg.values(),
                                    key=lambda v: (v["t"], v["sigma"]))}))
    print(f"DONE in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
