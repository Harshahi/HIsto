"""Every call into Sage happens here, in its own process.

Streamlit executes the app script on a ScriptRunner thread, and Sage's PARI/NTL
backends are not thread-safe: calling the Guruswami-Sudan root finder off the main
thread segfaults the whole server, taking the session with it. A bigger thread stack
does not help. So the app does all the numpy work in-process -- encoding, bundling,
and the Stage A histogram are pure numpy -- and shells out to this module for anything
that needs Sage.

Protocol: one JSON job on stdin, JSON Lines on stdout.

    {"event": "progress", "i": 3, "n": 17}
    {"event": "result",   ...}
    {"event": "error",    "message": "..."}
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core


def emit(**kw):
    sys.stdout.write(json.dumps(kw) + "\n")
    sys.stdout.flush()


def run(job):
    ctx = core.build_context(job["p"], job["m"], job["K"])
    d, ell_max = job.get("d", 1), job.get("ell_max", 40)
    op = job["op"]

    if op == "decode":
        msgs = job["messages"]
        if job.get("sigma"):                 # AWGN: e ~ CN(0, sigma^2 I)
            s = core.bundle_of(ctx, msgs) + core.gaussian_noise(
                ctx, job["sigma"], seed=job.get("seed", 0))
        else:
            s = core.noisy_bundle(ctx, msgs, alpha=job.get("alpha", 0.0),
                                  kind=job.get("kind", "isotropic"),
                                  seed=job.get("seed", 0))
        res = core.decode(ctx, s, d=d, ell_max=ell_max)
        return {"messages": res["messages"], "polys": res["polys"],
                "n_P": res["n_P"], "ell": res["ell"], "ms": res["ms"],
                "error": res["error"]}

    if op == "sweep":
        rows, msgs = core.sweep(ctx, job["t_limit"], d=d, ell_max=ell_max,
                                progress=lambda i, n: emit(event="progress", i=i, n=n))
        return {"rows": rows, "messages": msgs}

    if op == "peel":
        multiset = job["multiset"]
        hist = core.histograms(ctx, core.bundle_of(ctx, multiset))
        recovered, rounds = core.unbundle_with_multiplicities(
            ctx, hist, d=d, ell_max=ell_max)
        return {"recovered": recovered,
                "rounds": [{"residual": c, "found": ms} for c, ms in rounds]}

    if op == "noise_sweep":
        rows = core.noise_sweep(
            ctx, job["messages"], job["alphas"], kind=job.get("kind", "isotropic"),
            seed=job.get("seed", 0), d=d, ell_max=ell_max,
            progress=lambda i, n: emit(event="progress", i=i, n=n))
        return {"rows": rows}

    if op == "noise_grid":
        rows = core.noise_grid(
            ctx, job["t_values"], job["alphas"], job["kinds"], d=d, ell_max=ell_max,
            progress=lambda i, n: emit(event="progress", i=i, n=n))
        return {"rows": rows}

    if op == "trials":
        # Replay individual trials out of the noise_trials.py grid. That file keeps
        # only per-cell counts, so the detail has to be regenerated -- which is exact
        # rather than approximate, because the seed is a pure function of (trial index,
        # sigma index) and the decode parameters below are the ones it used.
        t, sigma, si = job["t"], job["sigma"], job["sigma_i"]
        msgs = core.message_list(ctx, t)
        clean = core.bundle_of(ctx, msgs)
        h0 = core.histograms(ctx, clean)
        want = sorted(msgs)
        syms = {tuple(u): core.rs_encode(ctx, u) for u in msgs}
        out = []
        for seed in range(job["lo"], job["hi"]):
            e = core.gaussian_noise(ctx, sigma, seed=seed * 1000 + si) if sigma else 0
            res = core.decode(ctx, clean + e, d=d,
                              ell_min=max(2, t - 3), ell_max=t + 5)
            Y, got = res["Y"], res["messages"]
            words = []
            for u in msgs:
                # the lemma needs agreement at EVERY position, so the positions where
                # this codeword's symbol fell out of the histogram are exactly what
                # cost it the guarantee
                lost = [j for j, a in enumerate(syms[tuple(u)]) if not Y[a, j]]
                words.append({"u": u, "lost": lost, "recovered": u in got})
            out.append({
                "seed": seed,
                "stage_a_ok": bool(np.array_equal(res["hist"], h0)),
                "bad_cells": int((res["hist"] != h0).sum()),
                "bad_cols": int(sum(1 for j in range(ctx["N"])
                                    if not np.array_equal(res["hist"][:, j], h0[:, j]))),
                "words": words,
                "spurious": [u for u in got if u not in want],
                "n_correct": sum(1 for u in got if u in want),
                "ell": int(res["ell"]),
            })
        return {"t": t, "sigma": sigma, "rows": out}

    if op == "attempt":
        msgs = job["messages"]
        Y = core.support_matrix(core.histograms(
            ctx, core.bundle_of(ctx, msgs)))
        rows = []
        for budget, ell in job["probes"]:
            r = core.attempt(ctx, Y, budget, ell, d=d)
            rows.append({"budget": budget, "ell": ell, "monomials": r["monomials"],
                         "kernel dim": r["kernel"], "roots": r["roots"],
                         "matched the bundle": sum(1 for u in r["messages"]
                                                   if u in msgs)})
        return {"rows": rows}

    raise ValueError(f"unknown op {op!r}")


if __name__ == "__main__":
    try:
        emit(event="result", **run(json.loads(sys.stdin.read())))
    except Exception as exc:
        emit(event="error", message=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
