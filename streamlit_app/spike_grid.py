"""The spiking-noise analogue of the Gaussian grid: sweep (mu, nu) instead of sigma.

    uv run --group sage --group app python streamlit_app/spike_grid.py

Proposition 2 is stated per block: recovery is guaranteed when at most s blocks have
||e_i||_2 >= sqrt(N)/2, those s blocks lie in E_r, and (r+1)s < N. That is a spiking
model written in the paper's own variables -- mu drives s, nu drives r -- so the right
experiment is a (mu, nu) grid, not another sigma sweep.

Each cell records what the paper's bound predicts AND what actually happened, so the
two can be compared rather than conflated:

    s_pred   blocks failing Corollary 2            (the paper's s)
    in_E     whether those blocks are in E_r at all (Prop 2's hypothesis (b))
    r        worst ell_1 histogram error among them (the paper's r)
    rs_ok    whether (r+1)s < N
    recall   fraction of bundled codewords actually returned

Rows are appended to data/spikes_<p>_<m>_<K>.jsonl.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core

P = int(os.environ.get("P", 3))
M = int(os.environ.get("M", 3))
K = int(os.environ.get("K", 3))
OUT = Path(__file__).resolve().parent / "data" / f"spikes_{P}_{M}_{K}.jsonl"
D = int(os.environ.get("D", 1))
T_LIST = [int(v) for v in os.environ.get("T_LIST", "5,10").split(",")]
MU = [int(v) for v in os.environ.get("MU", "1,2,3,5,8,13,20").split(",")]
NU = [float(v) for v in os.environ.get("NU", "2,5,10,14,20,40").split(",")]
WIDTH = [int(v) for v in os.environ.get("WIDTH", "1").split(",")]
SIGMA = float(os.environ.get("SIGMA", 0))     # background noise under the spikes
TRIALS = int(os.environ.get("TRIALS", 5))


def main():
    ctx = core.build_context(P, M, K)
    N = ctx["N"]
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["t"], r["mu"], r["nu"], r["width"], r["sigma"], r["seed"]))

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as fh:
        for t in T_LIST:
            msgs = core.message_list(ctx, t)
            clean = core.bundle_of(ctx, msgs)
            want = sorted(msgs)
            for w in WIDTH:
                for mu in MU:
                    for nu in NU:
                        for seed in range(TRIALS):
                            key = (t, mu, nu, w, SIGMA, seed)
                            if key in done:
                                continue
                            e = core.spiking_noise(ctx, sigma=SIGMA, mu=mu, nu=nu,
                                                   width=w, seed=seed)
                            rep = core.noise_report(ctx, e, clean)
                            try:
                                res = core.decode(ctx, clean + e, d=D,
                                                  ell_min=2, ell_max=40)
                                got = res["messages"]
                                ell = res.get("ell")
                            except Exception:
                                got, ell = [], None
                            row = {
                                "t": t, "mu": mu, "nu": nu, "width": w,
                                "sigma": SIGMA, "seed": seed, "d": D,
                                # how many blocks a spike actually landed in --
                                # mu counts spikes, which can share a block
                                "blocks_hit": int((rep["block_norms"] > 1e-9).sum()),
                                "s_pred": rep["s"], "r": rep["r"],
                                "outside_E": rep["outside_E"],
                                "prop2": bool(rep["prop2_applies"]),
                                "rs_ok": bool(rep["rs_ok"]),
                                "s_true": rep.get("s_true", 0),
                                "r_true": rep.get("r_true", 0),
                                "n_correct": sum(1 for u in got if u in want),
                                "exact": sorted(got) == want, "ell": ell}
                            fh.write(json.dumps(row) + "\n")
                            fh.flush()
                    print(f"  t={t} w={w} mu={mu} done", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
