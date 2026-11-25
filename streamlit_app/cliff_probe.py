"""The bundling cliff as a function of multiplicity: t = 12, 13, 14 at d = 2..6.

    uv run --group sage --group app python streamlit_app/cliff_probe.py

noise_trials.py sweeps the whole t range, which is wasted effort here -- everything
below t = 12 is already known to be flat in d. This probes only the three rows where
the ceiling actually bites, and only at the sigmas where recovery is partial:

    sigma = 0     the bundling cliff on its own, no noise. Deterministic, so one
                  trial is the whole answer and the rest would be copies.
    sigma = 1.2   recall is partial but high -- where d showed a gain at t = 13.
    sigma = 1.4   recall is partial and low.

Rows are appended to data/cliff_3_3_3.jsonl as they finish, so a run that is cut
short (d = 6 at t = 14 can take an hour for a single decode) still leaves usable
results, and a rerun skips what is already there.

Cost is steep and very uneven -- measured at t = 13, sigma = 1.2, one decode:
d=2 ~8s, d=3 ~57s, d=4 ~154s, d=5 ~310s. Tasks are therefore ordered cheapest
first, so the grid fills in from the affordable corner.
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

P = int(os.environ.get("P", 3))
M = int(os.environ.get("M", 3))
K = int(os.environ.get("K", 3))
# one file per code: rows from different (p, m, K) are not comparable and must not
# be aggregated together
OUT = Path(__file__).resolve().parent / "data" / f"cliff_{P}_{M}_{K}.jsonl"
D_LIST = [int(v) for v in os.environ.get("D_LIST", "2,3,4,5,6").split(",")]
T_LIST = [int(v) for v in os.environ.get("T_LIST", "12,13,14").split(",")]
SIGMAS = [float(v) for v in os.environ.get("SIGMAS", "0,1.2,1.4").split(",")]
TRIALS = int(os.environ.get("TRIALS", 5))
# Wall-clock budget per t row. A single d=6 decode can outrun any sensible grid, so
# each row gets a fixed slice of time and reports whatever it managed; the JSONL is
# append-only, so a later run with a bigger budget simply fills in more cells.
BUDGET = float(os.environ.get("BUDGET", 600))


def _by_d(name, default):
    """Parse a "d:value,d:value" env override into {d: value}."""
    return {int(k): int(v) for k, v in
            (pair.split(":") for pair in
             os.environ.get(name, default).split(",") if pair)}


# Noisy trials per cell, per d. A d=5 decode costs ~356s against ~8s at d=2, so a
# flat trial count would spend the whole budget on the two most expensive rows.
TRIALS_BY_D = _by_d("TRIALS_BY_D", "2:3,3:3,4:2,5:1,6:1")
# Hard cap on new decodes for the expensive multiplicities, applied after ordering
# by what each decode actually tells us (see _priority).
MAX_BY_D = _by_d("MAX_BY_D", "5:6,6:6")


def _priority(task):
    """Cheapest-to-interpret first, for when the cap bites.

    sigma = 0 before any noisy cell -- it answers "does this d reach this t at all",
    which is the precondition for the noisy cells meaning anything. Then t
    descending, because t = 14 is the open question and t = 12 is nearly known.
    """
    d, t, si, sg, seed = task
    return (sg != 0, sg, -t, seed)

_CTX = None


def _init():
    global _CTX
    _CTX = core.build_context(P, M, K)


def _run_one(task):
    """One decode. Returns the row to append, including how long it took."""
    d, t, si, sigma, seed = task
    ctx = _CTX
    msgs = core.message_list(ctx, t)
    clean = core.bundle_of(ctx, msgs)
    h0 = core.histograms(ctx, clean)
    want = sorted(msgs)
    e = core.gaussian_noise(ctx, sigma, seed=seed * 1000 + si) if sigma else 0
    N = ctx["N"]
    t0 = time.time()
    try:
        # One interpolation, scored but not yet filtered. The threshold is free
        # once the curve exists, so both rules are evaluated from the same decode:
        #   strict   score >= d*N        the noiseless rule
        #   relaxed  score >= d*(N - s)  histo.pdf IV-B2, s estimated from the
        #                                histogram column sums
        hist = core.histograms(ctx, clean + e)
        Y = core.support_matrix(hist)
        s_est, _, _ = core.corrupted_columns(ctx, hist)
        cands, _, ell = core.unbundle_candidates(
            ctx, Y, d=d, ell_min=max(2, t - 3), ell_max=t + 5)
        strict = [u for u, sc in cands if sc >= d * N]
        relaxed = [u for u, sc in cands if sc >= d * max(1, N - s_est)]
        err = None
        stage_a = bool(np.array_equal(hist, h0))
    except Exception as exc:                 # no interpolation polynomial at any ell
        strict, relaxed, ell, err = [], [], None, str(exc)[:120]
        s_est, stage_a = 0, False
    return {"d": d, "t": t, "sigma": sigma, "seed": seed,
            "stage_a_ok": stage_a, "exact": sorted(strict) == want,
            "n_correct": sum(1 for u in strict if u in want),
            "n_returned": len(strict),
            # the relaxed rule, from the same interpolation
            "s_est": int(s_est),
            "exact_relaxed": sorted(relaxed) == want,
            "n_correct_relaxed": sum(1 for u in relaxed if u in want),
            "n_spurious_relaxed": sum(1 for u in relaxed if u not in want),
            "ell": ell, "error": err,
            "sec": round(time.time() - t0, 1)}


def main():
    workers = int(os.environ.get("WORKERS", 8))
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["d"], r["t"], r["sigma"], r["seed"]))

    by_t = {}
    for d in D_LIST:
        cand = []
        for t in T_LIST:
            for si, sg in enumerate(SIGMAS):
                # sigma = 0 is deterministic: one trial is the whole answer
                n_seed = 1 if sg == 0 else TRIALS_BY_D.get(d, TRIALS)
                for seed in range(n_seed):
                    if (d, t, sg, seed) not in done:
                        cand.append((d, t, si, sg, seed))
        cap = MAX_BY_D.get(d)
        if cap is not None and len(cand) > cap:
            cand.sort(key=_priority)
            print(f"d={d}: capping {len(cand)} candidate decodes at {cap}",
                  flush=True)
            cand = cand[:cap]
        for task in cand:
            by_t.setdefault(task[1], []).append(task)

    total = sum(len(v) for v in by_t.values())
    print(f"{total} decodes queued ({len(done)} already on disk), "
          f"{workers} workers, {BUDGET:.0f}s budget per t row", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    t_start = time.time()
    with OUT.open("a") as fh:
        for t in T_LIST:
            tasks = by_t.get(t, [])
            if not tasks:
                continue
            # cheapest d first, so a row that runs out of time still has its
            # affordable half rather than a scatter of whatever finished
            tasks.sort(key=lambda x: x[0])
            print(f"\n--- t = {t}: {len(tasks)} decodes, {BUDGET:.0f}s budget ---",
                  flush=True)
            row_start = time.time()
            n_done = 0
            # a fresh pool per row, so terminating on the budget cannot strand
            # workers that the next row needs
            pool = mp.get_context("spawn").Pool(workers, initializer=_init)
            try:
                for r in pool.imap_unordered(_run_one, tasks):
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    n_done += 1
                    print(f"  [{n_done}/{len(tasks)}] d={r['d']} t={r['t']} "
                          f"sigma={r['sigma']} seed={r['seed']}: "
                          f"{r['n_correct']}/{r['t']} back, ell={r['ell']}, "
                          f"{r['sec']}s   (row {time.time() - row_start:.0f}s)",
                          flush=True)
                    if time.time() - row_start > BUDGET:
                        print(f"  budget spent -- stopping t = {t} at "
                              f"{n_done}/{len(tasks)}", flush=True)
                        break
            finally:
                pool.terminate()
                pool.join()
    print(f"\nDONE in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
