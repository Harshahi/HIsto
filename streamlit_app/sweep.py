"""Precompute the Gaussian-noise recovery sweep over (p, m, K, d, t, sigma).

    uv run --group sage --group app python streamlit_app/sweep.py

Writes data/stage_a.json and data/recovery.jsonl, and is resumable: re-running picks
up where it stopped. Two halves, because they cost wildly different amounts.

Stage A depends only on (N, sigma) and has a closed form, so it is tabulated for every
prime power q <= 100 at no cost at all.

Recovery needs a real Stage-B decode per cell, and the cost is driven by t, which runs
up to u_max = (q-1)/(K-1). Measured, q=25 K=2 t=24 takes 10s and q=49 K=2 t=24 takes
115s, so the large-q small-K corner is out of reach and cells are run cheapest first
under a budget.

Groups run in a PERSISTENT child process fed over stdin. Sage's Alekhnovich root finder
segfaults outright on some inputs -- a SIGSEGV, not an exception -- which killed an
earlier single-process version mid-run, so the work has to be isolated. One subprocess
per group also works but pays Sage's import every time, which dominated the run at
~1300 groups; a long-lived child amortises that and is simply restarted when it dies,
with the group that killed it recorded and skipped.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core

OUT = Path(__file__).resolve().parent / "data"
SIGMA_FRAC = [0.0, 0.5, 0.8, 1.0, 1.2]      # multiples of sigma_half(q)
D_VALUES = [1]
Q_LIMIT = 100


def prime_powers(limit=Q_LIMIT):
    """Every q = p^m <= limit, with its (p, m) -- primes and the gaps between them."""
    out = []
    for p in range(2, limit + 1):
        if any(p % f == 0 for f in range(2, int(p ** 0.5) + 1)):
            continue
        m, q = 1, p
        while q <= limit:
            out.append((p, m, q))
            m, q = m + 1, p ** (m + 1)
    return sorted(out, key=lambda r: r[2])


def t_values(u_max):
    """A few t per code: trivial, halfway, at the bound, and just past it."""
    return sorted({1, max(1, u_max // 2), u_max, u_max + 1})


# --------------------------------------------------------------------------------
# child: one (p, m, K, d) group, JSON Lines to stdout
# --------------------------------------------------------------------------------

def serve():
    """Read one group spec per line from stdin, emit its rows, repeat.

    A bare "." line marks the end of a group so the parent knows the child survived
    it and can attribute a later crash to the right place.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        spec = json.loads(line)
        run_group(spec["p"], spec["m"], spec["K"], spec["d"], spec["sec"])
        print(".", flush=True)


def run_group(p, m, K, d, per_cell_s):
    from cysignals.alarm import alarm, cancel_alarm
    from cysignals.signals import AlarmInterrupt

    q = p ** m
    ctx = core.build_context(p, m, K)
    u_max = ctx["u_max"]
    sh = core.sigma_half(q)
    for t in t_values(u_max):
        try:
            msgs = core.message_list(ctx, t)
        except ValueError:
            continue                          # more codewords than the space holds
        s0 = core.bundle_of(ctx, msgs)
        h0 = core.histograms(ctx, s0)
        want = sorted(msgs)
        for frac in SIGMA_FRAC:
            sigma = frac * sh
            try:
                alarm(per_cell_s)
                t0 = time.time()
                e = core.gaussian_noise(ctx, sigma, seed=0) if sigma else 0
                res = core.decode(ctx, s0 + e, d=d, ell_max=min(40, u_max + 4))
                cancel_alarm()
                row = {"p": p, "m": m, "q": q, "K": K, "d": d, "t": t,
                       "u_max": u_max, "sigma": sigma, "sigma_frac": frac,
                       "stage_a_ok": bool(np.array_equal(res["hist"], h0)),
                       "n_rec": len(res["messages"]),
                       "n_correct": sum(1 for u in res["messages"] if u in want),
                       "exact": res["messages"] == want,
                       "sec": time.time() - t0}
            except AlarmInterrupt:
                cancel_alarm()
                print(json.dumps({"p": p, "m": m, "q": q, "K": K, "d": d, "t": t,
                                  "u_max": u_max, "timeout": True}), flush=True)
                return                        # larger t only gets slower
            except Exception as exc:
                cancel_alarm()
                print(json.dumps({"p": p, "m": m, "q": q, "K": K, "d": d, "t": t,
                                  "err": type(exc).__name__}), flush=True)
                break
            print(json.dumps(row), flush=True)


# --------------------------------------------------------------------------------
# parent
# --------------------------------------------------------------------------------

def write_stage_a():
    """The closed form, complete over every prime power q <= 100."""
    pps = prime_powers()
    rows = []
    for p, m, q in pps:
        sh = core.sigma_half(q)
        for frac in [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]:
            rows.append({"p": p, "m": m, "q": q, "sigma": frac * sh,
                         "sigma_frac": frac,
                         "P_exact": core.stage_a_probability(q, frac * sh)})
    OUT.mkdir(exist_ok=True)
    (OUT / "stage_a.json").write_text(json.dumps(
        {"sigma_half": {str(q): core.sigma_half(q) for _, _, q in pps},
         "rows": rows}))
    return len(pps), len(rows)


def main():
    budget_s = float(os.environ.get("BUDGET_S", 5400))
    per_cell_s = int(os.environ.get("CELL_TIMEO", 45))
    n_pp, n_rows = write_stage_a()
    print(f"stage_a.json: {n_pp} prime powers q<={Q_LIMIT}, {n_rows} rows", flush=True)

    groups = []
    for p, m, q in prime_powers():
        for K in range(2, min(q - 1, 100) + 1):
            u_max = (q - 1) // (K - 1)
            if u_max < 1:
                continue
            for d in D_VALUES:
                # proxy for the dominant cost: interpolation points x monomials
                cost = max(t_values(u_max)) * q * (q * q / max(1, K - 1))
                groups.append((cost, p, m, K, d))
    groups.sort()
    print(f"{len(groups)} (q, K, d) groups queued", flush=True)

    rec_path, fail_path = OUT / "recovery.jsonl", OUT / "failed.json"
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["p"], r["m"], r["K"], r["d"]))
            except json.JSONDecodeError:
                pass
    failed = json.loads(fail_path.read_text()) if fail_path.exists() else []
    done |= {tuple(f) for f in failed}
    print(f"resuming: {len(done)} groups already done or failed", flush=True)

    fh = open(rec_path, "a")
    t_start = time.time()
    n_rows = n_crash = 0
    child = None

    def start_child():
        return subprocess.Popen(
            [sys.executable, __file__, "--serve"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    pending = [g for g in groups if (g[1], g[2], g[3], g[4]) not in done]
    for cost, p, m, K, d in pending:
        if time.time() - t_start > budget_s:
            print("BUDGET REACHED", flush=True)
            break
        if child is None or child.poll() is not None:
            child = start_child()
        try:
            child.stdin.write(json.dumps(
                {"p": p, "m": m, "K": K, "d": d, "sec": per_cell_s}) + "\n")
            child.stdin.flush()
        except (BrokenPipeError, ValueError):
            child = start_child()
            continue
        while True:                           # read until the group's end marker
            line = child.stdout.readline()
            if line == "":                    # the child died on this group
                n_crash += 1
                failed.append([p, m, K, d])
                fail_path.write_text(json.dumps(failed))
                child = None
                break
            line = line.strip()
            if line == ".":
                break
            if line.startswith("{"):
                fh.write(line + "\n")
                n_rows += 1
        fh.flush()
        if n_rows and n_rows % 1000 < len(SIGMA_FRAC):
            print(f"  {n_rows} rows, {n_crash} crashed groups, "
                  f"{time.time() - t_start:.0f}s (at q={p ** m} K={K})", flush=True)
    if child is not None and child.poll() is None:
        child.terminate()
    fh.close()
    print(f"DONE: {n_rows} rows, {n_crash} crashed groups, "
          f"{time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve()
    elif len(sys.argv) > 1 and sys.argv[1] == "--cell":
        _p, _m, _K, _d, _sec = (int(v) for v in sys.argv[2:7])
        run_group(_p, _m, _K, _d, _sec)
    else:
        main()
