"""Fold cliff_3_3_3.jsonl into per-d grids the Gaussian tab can display.

    uv run --group app python streamlit_app/cliff_to_grid.py

cliff_probe.py writes one row per decode. The app's heatmap wants one row per
(t, sigma) cell in the shape noise_trials.py emits, so this aggregates and writes
data/noise_trials_3_3_3_cliff_d<N>.json -- one grid per multiplicity, each covering
only t = 12..14. available_grids() keys on d, so they all appear in the selector
side by side instead of overwriting each other.

Cells are included with however many trials actually finished, since the probe is
time-boxed per t row and the expensive corners are deliberately thin. Each cell
carries its own n, and the app already divides by it.
"""

import json
from collections import defaultdict
from pathlib import Path

import os

DATA = Path(__file__).resolve().parent / "data"
P = int(os.environ.get("P", 3))
M = int(os.environ.get("M", 3))
K = int(os.environ.get("K", 3))
SRC = DATA / f"cliff_{P}_{M}_{K}.jsonl"
N = P ** M


def main():
    if not SRC.exists():
        raise SystemExit(f"no {SRC} -- run cliff_probe.py first")

    cells = defaultdict(lambda: {"n": 0, "stage_a_ok": 0, "exact": 0,
                                 "n_correct": 0, "sec": 0.0})
    seen = set()
    for line in SRC.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["d"], r["t"], r["sigma"], r["seed"])
        if key in seen:            # append-only file, so a rerun can duplicate
            continue
        seen.add(key)
        c = cells[(r["d"], r["t"], r["sigma"])]
        c["n"] += 1
        c["stage_a_ok"] += bool(r["stage_a_ok"])
        c["exact"] += bool(r["exact"])
        c["n_correct"] += r["n_correct"]
        c["sec"] += r["sec"]

    by_d = defaultdict(list)
    for (d, t, sigma), c in cells.items():
        by_d[d].append({"t": t, "sigma": sigma, "n": c["n"],
                        "stage_a_ok": c["stage_a_ok"], "exact": c["exact"],
                        "n_correct": c["n_correct"],
                        "sec": round(c["sec"], 1)})

    for d, rows in sorted(by_d.items()):
        rows.sort(key=lambda r: (r["t"], r["sigma"]))
        sigmas = sorted({r["sigma"] for r in rows})
        out = DATA / f"noise_trials_{P}_{M}_{K}_cliff_d{d}.json"
        out.write_text(json.dumps({
            "p": P, "m": M, "K": K, "N": N, "d": d,
            "u_max": (N - 1) // (K - 1),
            # the selector label shows this; the probe is time-boxed, so cells
            # differ and the per-cell n is what actually counts
            "trials": max(r["n"] for r in rows),
            "sigmas": sigmas,
            "t_max": max(r["t"] for r in rows),
            "complete": False,
            "rows": rows}, indent=1))
        tot = sum(r["n"] for r in rows)
        print(f"d={d}: {len(rows)} cells, {tot} decodes, "
              f"{sum(r['sec'] for r in rows) / 3600:.2f} core-hr -> {out.name}")


if __name__ == "__main__":
    main()
