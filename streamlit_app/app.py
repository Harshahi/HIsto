"""histogram_recovery.ipynb with the parameters on sliders.

The notebook fixes p = 3, m = 3, K = 3 and walks through encode -> bundle -> decode
once. This is the same four parts with (p, m, K, d) moved into the sidebar, so you can
drag the parameters and watch the cliff move.

    uv run --group app --group sage streamlit run streamlit_app/app.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import report

st.set_page_config(page_title="Histogram recovery", page_icon="◐", layout="wide")

# --------------------------------------------------------------------------------
# palette -- the dataviz reference instance, light and dark columns
# --------------------------------------------------------------------------------

LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#dcdbd6",
         "s1": "#2a78d6", "s2": "#eb6834",
         # sequential ramp, single hue, light -> dark. Starts at pure white so an
         # absolute zero reads as empty rather than as "a little bit of something".
         "heat": ["#ffffff", "#bcd7f2", "#5a9ae0", "#2a78d6", "#14417a"]}
DARK = {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#3a3a38",
        "s1": "#3987e5", "s2": "#d95926",
        "heat": ["#ffffff", "#bcd7f2", "#5a9ae0", "#3987e5", "#1b4d8f"]}


def palette():
    mode = getattr(getattr(st, "context", None), "theme", None)
    return DARK if getattr(mode, "type", "light") == "dark" else LIGHT


C = palette()
IN_LABEL, OUT_LABEL = "codewords bundled in", "codewords decoded out"


def styled(chart, height=380):
    """Recessive grid and axes, no chart junk, text in ink tokens not series color.

    A faceted chart carries its size on the inner spec, and setting `height` on the
    outer FacetChart is a schema error -- so pass height=None for those.
    """
    props = {"background": C["surface"]}
    if height is not None:
        props["height"] = height
    return (chart
            .properties(**props)
            .configure_view(stroke=None)
            .configure_axis(domainColor=C["grid"], tickColor=C["grid"],
                            gridColor=C["grid"], gridOpacity=0.9, gridWidth=0.6,
                            labelColor=C["ink2"], titleColor=C["ink2"],
                            labelFontSize=11, titleFontSize=12)
            .configure_legend(labelColor=C["ink"], titleColor=C["ink2"],
                              symbolType="stroke", symbolStrokeWidth=3)
            .configure_title(color=C["ink"], fontSize=13, anchor="start"))


def two_series(df, y_title, title, series_order, rule_at=None, rule_label=None,
               x_title="number of codewords bundled,  t"):
    """One y-axis, two categorical hues in fixed order, a legend, hover tooltips, and
    direct labels where they fit. Used for the norm chart and the in-versus-out one."""
    color = alt.Color("series:N",
                      scale=alt.Scale(domain=series_order, range=[C["s1"], C["s2"]]),
                      legend=alt.Legend(title=None, orient="top-left"))
    base = alt.Chart(df).encode(
        x=alt.X("t:Q", title=x_title, scale=alt.Scale(nice=False, padding=18)),
        y=alt.Y("value:Q", title=y_title))
    line = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=55, filled=True)
                          ).encode(color=color)
    hover = base.mark_circle(size=170, opacity=0).encode(
        color=color,
        tooltip=[alt.Tooltip("t:Q", title="t"), alt.Tooltip("series:N", title=""),
                 alt.Tooltip("value:Q", title="value", format=".4~g")])
    layers = [line, hover]

    # Direct labels supplement the legend, but only when they are certain to fit. They
    # ride the last point, right-aligned so they grow inward and cannot overflow the
    # right edge, and are dropped entirely when the two series end too close together
    # to label without collision -- the legend, tooltips and table carry it instead.
    ends = df.sort_values("t").groupby("series", as_index=False).tail(1)
    span = df["value"].max() - df["value"].min()
    if len(ends) == 2 and span > 0 and abs(ends["value"].diff().iloc[-1]) > 0.25 * span:
        top = ends["value"].max()
        for _, row in ends.iterrows():
            layers.append(alt.Chart(pd.DataFrame([row]))
                          .mark_text(align="right", dx=-6, fontSize=11,
                                     dy=16 if row["value"] == top else -12)
                          .encode(x="t:Q", y="value:Q", text="series:N", color=color))
    if rule_at is not None:
        ref = pd.DataFrame({"t": [rule_at], "lab": [rule_label or ""]})
        layers += [
            alt.Chart(ref).mark_rule(strokeDash=[4, 3], strokeWidth=1.4,
                                     color=C["ink2"]).encode(x="t:Q"),
            alt.Chart(ref).mark_text(align="left", dx=5, dy=-6, baseline="bottom",
                                     fontSize=11, color=C["ink2"]
                                     ).encode(x="t:Q", y=alt.value(6), text="lab:N")]
    return styled(alt.layer(*layers).properties(title=title).resolve_scale(color="shared"))


# --------------------------------------------------------------------------------
# cached setup
# --------------------------------------------------------------------------------

@st.cache_resource(show_spinner="reading encode_m.ipynb ...")
def get_encoder():
    return core.encoder()


@st.cache_resource(show_spinner="building the Hadamard matrix ...")
def get_context(p, m, K):
    return core.numpy_context(p, m, K, enc=get_encoder())


REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data(show_spinner="loading the trial grid ...")
def _load_trials_cached(path, mtime, size):
    """Cached on (path, mtime, size) so the grid refreshes when the file is rebuilt.

    Keying on the filename alone meant a session opened while a sweep was still
    running kept serving that partial snapshot forever -- the grid appeared to stop
    at whatever t the run had reached when the page first loaded.
    """
    return json.loads(Path(path).read_text())


def load_trials(stem="noise_trials"):
    """The trials grid from noise_trials.py, if it has been built."""
    path = DATA_DIR / f"{stem}.json"
    if not path.exists():
        return None
    st_ = path.stat()
    return _load_trials_cached(str(path), st_.st_mtime, st_.st_size)


def available_grids():
    """Every (p, m, K) grid on disk, newest data first."""
    out = {}
    for f in sorted(DATA_DIR.glob("noise_trials*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        # key on d as well -- two grids for the same code at different
        # multiplicities are different experiments, and a (p, m, K) key would
        # silently drop one of them
        out[f"p={d['p']}, m={d['m']}, K={d['K']}, d={d.get('d', 1)}"
            f"  (N={d['N']}, u_max={d['u_max']}, {d['trials']} trials)"] = f.stem
    return out


def cliff_path(p_, m_, K_):
    """cliff_probe.py writes one JSONL per code; timings from a different code
    would make the cost estimates meaningless."""
    return DATA_DIR / f"cliff_{p_}_{m_}_{K_}.jsonl"
CLIFF_PROBE = Path(__file__).resolve().parent / "cliff_probe.py"


@st.cache_data(show_spinner=False)
def decode_costs(mtime, p_, m_, K_):
    """Median seconds per decode for each (d, t) actually measured.

    cliff_probe.py records `sec` on every row, so the estimate shown before a run
    is this machine's own timings rather than a guess. Costs differ by two orders
    of magnitude across d, which is the whole reason the grid is sparse.
    """
    import statistics
    seen = {}
    src = cliff_path(p_, m_, K_)
    if src.exists():
        for line in src.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen.setdefault((r["d"], r["t"]), []).append(r.get("sec", 0.0))
    return {k: statistics.median(v) for k, v in seen.items() if v}


def estimate_sec(costs, d, t):
    """Seconds for one decode at (d, t), and how that number was arrived at.

    Exact measurement if we have one. Otherwise the nearest t at the same d,
    scaled by (t/t_ref)^3 -- the interpolation matrix grows in both dimensions
    with t, and the kernel is cubic in its size. Failing that, scale across d by
    d^3.3, fitted to the measured d=2..6 ladder (8s, 57s, 154s, 356s, 530s).
    """
    if (d, t) in costs:
        return costs[(d, t)], "measured"
    same_d = [(abs(tt - t), tt) for (dd, tt) in costs if dd == d]
    if same_d:
        _, tref = min(same_d)
        return costs[(d, tref)] * (max(t, 1) / max(tref, 1)) ** 3, f"scaled from t={tref}"
    any_d = [(abs(dd - d), dd, tt) for (dd, tt) in costs if abs(tt - t) <= 2]
    if any_d:
        _, dref, tref = min(any_d)
        return costs[(dref, tref)] * (d / max(dref, 1)) ** 3.3, f"scaled from d={dref}"
    return max(0.2, 0.3 * (d ** 3.3) * max(t / 10, 0.1) ** 3), "rough model"


def human_time(sec):
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.0f} min"
    return f"{sec / 3600:.1f} hr"


def spikes_path(p_, m_, K_):
    """spike_grid.py writes one JSONL per code."""
    return DATA_DIR / f"spikes_{p_}_{m_}_{K_}.jsonl"


@st.cache_data(show_spinner=False)
def spike_cells(mtime, p_, m_, K_):
    """One row per (t, width, sigma, mu, nu) cell, aggregated over seeds.

    Carries the Proposition 2 bookkeeping beside the outcome -- s_pred (blocks past
    Corollary 2), how many of those are genuinely in E_r, and whether (r+1)s < N --
    so the guarantee and the behaviour can be read against each other instead of
    one standing in for the other.
    """
    src = spikes_path(p_, m_, K_)
    if not src.exists():
        return pd.DataFrame()
    rows = []
    seen = set()
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (r["t"], r["mu"], r["nu"], r["width"], r["sigma"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["t", "width", "sigma", "mu", "nu"], as_index=False).agg(
        n=("seed", "count"), n_correct=("n_correct", "sum"),
        exact=("exact", "sum"), s_pred=("s_pred", "mean"),
        s_true=("s_true", "mean"), r=("r", "mean"),
        outside_E=("outside_E", "sum"), rs_ok=("rs_ok", "mean"))
    g["mean recall"] = g["n_correct"] / (g["n"] * g["t"])
    g["recovery exact"] = g["exact"] / g["n"]
    return g


def _grid_mtime():
    """Newest mtime across every data file the layers are built from, so the cache
    drops as soon as a running probe appends a decode."""
    files = list(DATA_DIR.glob("noise_trials*.json")) + list(DATA_DIR.glob("cliff_*.jsonl"))
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_data(show_spinner=False)
def all_d_layers(mtime, p_, m_, K_):
    """Every grid on disk folded into one long frame, one row per (d, t, sigma).

    The per-d grids come from different runs with wildly different trial counts --
    1000 at d=1, 3 at d=2, 1 at d=5 -- so `n` is carried through per row and shown
    rather than averaged away. A cell measured once is an anecdote; the layered
    chart marks those explicitly instead of colouring them like the rest.
    """
    frames = []
    for f in sorted(DATA_DIR.glob("noise_trials*.json")):
        try:
            g = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        rows = pd.DataFrame(g["rows"])
        if rows.empty:
            continue
        if (g.get("p"), g.get("m"), g.get("K")) != (p_, m_, K_):
            continue            # a grid for a different code is not comparable
        rows = rows[rows["n"] > 0].copy()
        rows["d"] = g.get("d", 1)
        # cliff_to_grid records total seconds per cell; per-decode is what a
        # reader wants when deciding whether to run more
        rows["sec per decode"] = (rows["sec"] / rows["n"]) if "sec" in rows \
            else float("nan")
        rows["mean recall"] = rows["n_correct"] / (rows["n"] * rows["t"])
        rows["recovery exact"] = rows["exact"] / rows["n"]
        rows["Stage A histogram exact"] = rows["stage_a_ok"] / rows["n"]
        frames.append(rows)
    # cliff_probe.py appends one row per decode and never writes a grid file, so
    # read its JSONL directly rather than waiting for cliff_to_grid.py to be run by
    # hand. Otherwise a probe can finish and the tab still shows nothing.
    src = cliff_path(p_, m_, K_)
    if src.exists():
        seen, cells = set(), {}
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                        # a half-written final line
            key = (r["d"], r["t"], r["sigma"], r["seed"])
            if key in seen:
                continue                        # append-only: reruns can duplicate
            seen.add(key)
            c = cells.setdefault((r["d"], r["t"], r["sigma"]),
                                 {"n": 0, "stage_a_ok": 0, "exact": 0,
                                  "n_correct": 0, "sec": 0.0, "n_rel": 0,
                                  "n_correct_relaxed": 0, "exact_relaxed": 0,
                                  "spurious_relaxed": 0, "s_est": 0})
            c["n"] += 1
            c["stage_a_ok"] += bool(r["stage_a_ok"])
            c["exact"] += bool(r["exact"])
            c["n_correct"] += r["n_correct"]
            c["sec"] += r.get("sec", 0.0)
            # Rows written before the dual-threshold change have no relaxed
            # fields. Count how many rows DO, so a cell that is only partly
            # covered reports a rate over the rows that actually measured it
            # rather than silently averaging in zeros.
            if "n_correct_relaxed" in r:
                c["n_rel"] += 1
                c["n_correct_relaxed"] += r["n_correct_relaxed"]
                c["exact_relaxed"] += bool(r["exact_relaxed"])
                c["spurious_relaxed"] += r.get("n_spurious_relaxed", 0)
                c["s_est"] += r.get("s_est", 0)
        if cells:
            cf = pd.DataFrame([{"d": d, "t": t, "sigma": sg, **c}
                               for (d, t, sg), c in cells.items()])
            cf["mean recall"] = cf["n_correct"] / (cf["n"] * cf["t"])
            cf["recovery exact"] = cf["exact"] / cf["n"]
            cf["Stage A histogram exact"] = cf["stage_a_ok"] / cf["n"]
            cf["sec per decode"] = cf["sec"] / cf["n"]
            rel = cf["n_rel"].replace(0, float("nan"))
            cf["mean recall (relaxed)"] = cf["n_correct_relaxed"] / (rel * cf["t"])
            cf["recovery exact (relaxed)"] = cf["exact_relaxed"] / rel
            cf["spurious per decode"] = cf["spurious_relaxed"] / rel
            cf["mean s"] = cf["s_est"] / rel
            frames.append(cf)

    if not frames:
        # a shaped empty frame, so callers can select columns without a KeyError
        return pd.DataFrame(columns=["t", "sigma", "n", "stage_a_ok", "exact",
                                     "n_correct", "d", "mean recall",
                                     "recovery exact", "Stage A histogram exact",
                                     "sec per decode"])
    out = pd.concat(frames, ignore_index=True)
    # one run per (d, t, sigma): prefer whichever carries more trials
    out = out.sort_values("n").drop_duplicates(["d", "t", "sigma"], keep="last")
    return out.reset_index(drop=True)


WORKER = Path(__file__).resolve().parent / "worker.py"


@st.cache_data(show_spinner=False)
def monomial_count(budget, ell, w):
    """How many monomials x^i y^j fit under the interpolation budget.

    Sage's bound is exclusive -- deg < budget -- so i runs 0 .. budget-1-w*j, which is
    `budget - w*j` values. Matches the measured counts exactly (196 at d=1, 756 at d=2).
    """
    return sum(max(0, budget - w * j) for j in range(ell + 1))


@st.cache_data(show_spinner=False)
def observed_points(p_, m_, K_, t_max):
    """n_P per t: the number of distinct (position, symbol) pairs actually observed.

    Below t*N whenever two bundled codewords share a symbol somewhere, and it is the
    real constraint count, so the counting table uses it rather than t*N. Pure numpy,
    so this stays in-process -- no Sage, no worker.
    """
    c = core.numpy_context(p_, m_, K_, enc=get_encoder())
    msgs = core.message_list(c, t_max)
    acc = np.zeros(c["dim"], dtype=complex)
    out = []
    for t in range(1, t_max + 1):
        acc = acc + core.encode(c, msgs[t - 1])
        out.append(int(core.support_matrix(core.histograms(c, acc)).sum()))
    return out


@st.cache_data(show_spinner=False)
def measured_kernels(p_, m_, K_, mtime):
    """Whatever gs_bound.ipynb has already measured, keyed (d, t) -> row."""
    f = DATA_DIR / f"gs_bound_{p_}_{m_}_{K_}.jsonl"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[f"{r['d']},{r['t']}"] = r
    return out


def d_sigma_index(trials, sigma):
    """noise_trials.py seeds each draw with `trial * 1000 + sigma_index`, so replaying
    a cell faithfully needs that index, not the sigma value."""
    for i, v in enumerate(trials["sigmas"]):
        if abs(v - sigma) < 1e-9:
            return i
    return 0


def int_slider(label, lo, hi, default, **kw):
    """A slider that degrades to a caption when there is nothing to choose.

    At the small end of the parameter space these ranges collapse -- p = 2, m = 1
    gives u_max = 1 -- and Streamlit raises outright when a slider's min equals its
    max. Returning the single legal value keeps those corners usable.
    """
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        st.caption(f"{label}: **{lo}** (nothing to vary here at these parameters)")
        return lo
    return st.slider(label, lo, hi, int(min(max(default, lo), hi)), **kw)



def run_job(job, on_progress=None):
    """Hand one job to worker.py and read JSON Lines back.

    Sage cannot run on Streamlit's script thread -- PARI is not thread-safe and the
    Guruswami-Sudan root finder segfaults the server outright. So decoding lives in a
    subprocess. Encoding, bundling and the Stage A histogram are pure numpy and stay
    in-process, which is why dragging the sliders still feels immediate.
    """
    proc = subprocess.Popen([sys.executable, str(WORKER)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc.stdin.write(json.dumps(job))
    proc.stdin.close()

    result = None
    for line in proc.stdout:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue                              # Sage chatters on stdout; ignore it
        if msg.get("event") == "progress" and on_progress:
            on_progress(msg["i"], msg["n"])
        elif msg.get("event") == "error":
            proc.wait()
            raise RuntimeError(msg["message"])
        elif msg.get("event") == "result":
            result = msg
    stderr = proc.stderr.read()
    proc.wait()
    if result is None:
        raise RuntimeError(stderr.strip()[-900:] or f"worker exited {proc.returncode}")
    return result


# --------------------------------------------------------------------------------
# sidebar -- the parameters
# --------------------------------------------------------------------------------

st.sidebar.title("Parameters")
st.sidebar.caption("Drag these. Everything on the right re-derives from them.")

# Each slider's range is derived from the ones above it, so an unusable combination is
# simply unreachable. The earlier version used fixed ranges plus one blunt guard that
# st.stop()'d the whole page -- and since the default p = 3, m = 3 sat exactly on that
# guard's corner, raising either slider blanked every tab.

DIM_CAP = 500_000          # numpy memory: dim = p^2m complex numbers per hypervector
M_CAP = 4


def max_m_for(prime):
    """Largest m whose hypervector is still a reasonable size at this p."""
    m = min_m_for(prime)
    while m < M_CAP and prime ** (2 * (m + 1)) <= DIM_CAP:
        m += 1
    return m


def min_m_for(prime):
    """Smallest m that leaves room for a non-degenerate code.

    K has to be at least 2 to be a message and at most N - 1 for the Reed-Solomon
    code to have d_min >= 2, so N = p^m must be at least 3. Only p = 2 is affected:
    N = 2 admits no usable K at all, and Sage refuses it outright ("the decoding
    radius must be less than the Johnson radius, which is 0.59").
    """
    return 1 if prime >= 3 else 2


p = st.sidebar.select_slider("**p** — the prime", options=core.PRIMES, value=3, key="p",
                             help="How many distinct phase values one coordinate takes. "
                                  "phi sends F_p to the p-th roots of unity.")

m_lo, m_cap = min_m_for(p), max_m_for(p)
st.session_state["m"] = min(max(st.session_state.get("m", 3), m_lo), m_cap)
m = st.sidebar.slider("**m** — extension degree", m_lo, m_cap, key="m",
                      help=f"The alphabet is F_(p^m). Raising m grows the hypervector "
                           f"exponentially; every bound keeps its form. Capped at "
                           f"{m_cap} for p = {p}, to keep dim = p^2m under "
                           f"{DIM_CAP:,}.")

N, DIM = p ** m, p ** (2 * m)

K_cap = min(8, N - 1)             # K = N gives d_min = 1: no code at all
if K_cap <= 2:
    K = 2
    st.sidebar.caption(f"**K** — pinned at 2; N = {N} leaves no room for more.")
else:
    st.session_state["K"] = min(st.session_state.get("K", 3), K_cap)
    K = st.sidebar.slider("**K** — Reed-Solomon dimension", 2, K_cap, key="K",
                          help="Symbols per message. The only parameter where more is "
                               "worse for bundling: vocabulary N^K goes up "
                               "exponentially but u_max = (N-1)/(K-1) comes down.")

# This d drives LIVE decodes and the cost is steep -- measured at t=13: 2s at d=1,
# 8s at d=2, 57s at d=3, 154s at d=4, 356s at d=5, ~530s at d=6. It is not capped at
# a small number any more, because the interesting behaviour (t past u_max) only
# appears at d >= 4; instead the cost is stated next to the slider.
D_CAP = 6
d = st.sidebar.slider("**d** — Guruswami-Sudan multiplicity", 1, D_CAP, 1, key="d",
                      help="Interpolate through each point with multiplicity d. "
                           "Raising it lifts the ceiling on how many codewords Q can "
                           "carry (deg_y Q <= dN/(K-1)) but costs roughly d^3.3 per "
                           "decode. t > u_max needs d >= 4.")

# Everything below depends on d, so it is derived here rather than hard-coded.
ELL_SAT = d * N // (K - 1)        # past this, raising ell adds no monomials at all
_COST = {1: 2, 2: 8, 3: 57, 4: 154, 5: 356, 6: 530}
st.sidebar.caption(
    f"`ell_sat` = dN/(K-1) = **{ELL_SAT}** · one decode at t≈13 ≈ "
    f"**{_COST.get(d, 530)}s**"
    + ("" if d >= 4 else f" · t > u_max needs d ≥ 4"))

# ell used to be a slider. It is not a choice: unbundle walks ell upward and stops
# at the first value that admits a curve, and `ell_sat` is always tried as a last
# rung, so any ceiling above ell_sat is equivalent and any ceiling below it only
# hides solutions. Pinned here rather than exposed.
ell_max = ELL_SAT + 1

extra = st.sidebar.slider("**bundle past u_max by**", 0, 15, 4, key="extra",
                          help="How far beyond u_max the t sliders may go, so you "
                               "can see where it breaks. t itself is set per tab.")

ctx = get_context(p, m, K)
B = core.bounds(ctx)
U_MAX = ctx["u_max"]

# You cannot bundle more distinct codewords than the message space holds -- at tiny
# (p, m, K) that ceiling bites before u_max does.
T_LIMIT = min(U_MAX + extra, B["size"])

st.sidebar.divider()
sc1, sc2 = st.sidebar.columns(2)
sc1.metric("N = p^m", N, help="symbols in the alphabet, and codeword length")
sc2.metric("dim = p^2m", f"{DIM:,}", help="length of the hypervector")
sc1.metric("size = N^K", f"{B['size']:,}", help="how many distinct messages")
sc2.metric("u_max", U_MAX, help="(N-1)/(K-1) — distinct codewords IV-B can unbundle")

# --------------------------------------------------------------------------------

st.title("Histogram recovery, end to end")
st.caption(f"p = {p}, m = {m}, K = {K}, d = {d} &nbsp;·&nbsp; "
           f"Deng and Raviv, *Efficient Vector Symbolic Architectures from Histogram "
           f"Recovery* (arXiv:2511.01838) &nbsp;·&nbsp; mirrors `histogram_recovery.ipynb`")

def _grid_section(p, m, K, d, ell_max, N, U_MAX, T_LIMIT, ctx):
    """Both failure axes at once: t up the side, noise level across the bottom.

    u_max is SS IV-B's limit and Corollary 2's alpha = 1 is SS IV-A's; they are
    independent, which is exactly what the grid makes visible.
    """
    st.divider()
    st.markdown("##### Both cliffs at once")
    st.markdown(
        "The sweep above fixes $t$ and varies $\\alpha$; the **In versus out** tab "
        "fixes $\\alpha = 0$ and varies $t$. They are two slices of one picture. "
        "Here is the whole grid, for all three noise directions — "
        f"$u_{{max}} = {U_MAX}$ is §IV-B's limit, $\\alpha = 1$ is §IV-A's, and they "
        "are independent.")

    gmax = int(min(T_LIMIT, U_MAX + 2))
    ALPHAS = [0.0, 0.5, 0.9, 1.0, 1.25, 2.0, 3.0, 5.0]
    KINDS = ["real", "isotropic", "imaginary"]
    t_values = list(range(1, gmax + 1))
    n_cells = len(t_values) * len(ALPHAS) * len(KINDS)
    st.caption(f"{n_cells} full decodes ({len(t_values)} values of t x "
               f"{len(ALPHAS)} noise levels x {len(KINDS)} directions). "
               f"Minutes, not seconds — the result is cached per parameter set.")

    gkey = (p, m, K, d, ell_max, gmax)
    gstore = st.session_state.setdefault("noise_grids", {})
    if not (st.button("Build the grid", key="grid_btn") or gkey in gstore):
        return
    if gkey not in gstore:
        bar = st.progress(0.0, text="decoding ...")
        t0 = time.time()
        try:
            out = run_job({"op": "noise_grid", "p": p, "m": m, "K": K,
                           "t_values": t_values, "alphas": ALPHAS, "kinds": KINDS,
                           "d": d, "ell_max": ell_max},
                          on_progress=lambda i, n: bar.progress(
                              i / n, text=f"{i} of {n} decodes — "
                                          f"{time.time() - t0:.0f}s"))
        except RuntimeError as exc:
            bar.empty()
            st.error(f"**The grid failed.**\n\n```\n{exc}\n```")
            return
        bar.empty()
        gstore[gkey] = out["rows"]
    grows = gstore[gkey]

    gdf = pd.DataFrame(grows)
    # put the two boundaries on the tick labels -- Altair cannot draw a rule
    # between two ordinal categories, and the axis is the honest place for them
    gdf["alpha_lab"] = [f"{a:g}" + ("  ◂ Cor. 2" if a == 1.0 else "")
                        for a in gdf["alpha"]]
    gdf["t_lab"] = [f"{t}" + ("  ◂ u_max" if t == U_MAX else "") for t in gdf["t"]]
    a_order = [f"{a:g}" + ("  ◂ Cor. 2" if a == 1.0 else "") for a in ALPHAS]
    t_order = [f"{t}" + ("  ◂ u_max" if t == U_MAX else "") for t in t_values]

    heat = alt.Chart(gdf).mark_rect(stroke=C["surface"], strokeWidth=2).encode(
        x=alt.X("alpha_lab:O", title="noise level  α   (multiples of √N/2)",
                sort=a_order),
        y=alt.Y("t_lab:O", title="codewords bundled,  t", sort=list(reversed(t_order))),
        color=alt.Color("frac:Q", title="fraction recovered",
                        scale=alt.Scale(range=C["heat"], domain=[0, 1])),
        tooltip=[alt.Tooltip("kind:N", title="noise"),
                 alt.Tooltip("t:Q", title="t"),
                 alt.Tooltip("alpha:Q", title="α"),
                 alt.Tooltip("n_correct:Q", title="correctly returned"),
                 alt.Tooltip("n_rec:Q", title="returned in total"),
                 alt.Tooltip("exact:N", title="exact"),
                 alt.Tooltip("stage_a_ok:N", title="Stage A exact")])

    st.altair_chart(
        styled(heat.properties(width=170, height=max(220, 22 * len(t_values)))
               .facet(column=alt.Column("kind:N", title=None,
                                        sort=KINDS,
                                        header=alt.Header(labelFontSize=12,
                                                          labelColor=C["ink"],
                                                          labelFontWeight="bold"))),
               height=None),
        width="stretch", theme=None)

    lines = [f"{'kind':>11} {'t<=u_max, a<1':>14} {'t<=u_max, a>=1':>15} "
             f"{'t>u_max':>9}", "-" * 54]
    for kd in KINDS:
        sub = [r for r in grows if r["kind"] == kd]
        def frac(sel):
            s = [r for r in sub if sel(r)]
            return f"{sum(r['exact'] for r in s)}/{len(s)}" if s else "-"
        lines.append(f"{kd:>11} "
                     f"{frac(lambda r: r['t'] <= U_MAX and r['alpha'] < 1):>14} "
                     f"{frac(lambda r: r['t'] <= U_MAX and r['alpha'] >= 1):>15} "
                     f"{frac(lambda r: r['t'] > U_MAX):>9}")
    lines += ["", "cells exactly recovered, out of each region.",
              "the left column is where both papers' guarantees hold."]
    st.code("\n".join(lines), language="text")


tabs = st.tabs(["1 · Encode", "2 · Bundle", "3 · Decode", "4 · Noise",
                "5 · Gaussian noise", "6 · Spiking noise", "7 · In versus out",
                "8 · Repeats (§IV-B3)", "Bounds"])

# ================================================================================
# 1 -- Encode
# ================================================================================
with tabs[0]:
    st.markdown("The paper's eq. (1), one layer at a time: a message of "
                f"$K = {K}$ symbols becomes a hypervector of ${DIM}$ complex numbers.")

    cols = st.columns(min(K, 6))
    msg = []
    for i in range(K):
        with cols[i % len(cols)]:
            msg.append(st.slider(f"u[{i}]", 0, N - 1, min((i * 2 + 1) % N, N - 1),
                                 key=f"msg{i}"))

    codeword = core.rs_encode(ctx, msg)
    hv = core.encode(ctx, msg)
    st.code(report.encode_steps(ctx, msg, codeword, hv), language="text")
    assert np.allclose(np.abs(hv), 1.0)

    st.markdown("""
##### The same codeword, three ways

The pipeline writes one codeword as three different tables, and the differences between
them are the whole algorithm:

| | each cell holds | what it is for |
|---|---|---|
| **Hadamard** grid (STEP 2 above) | a phase exponent in $0 \\ldots p-1$ | this is what gets bundled |
| **Kautz-Singleton** $W(c)$ | $0$ or $1$ --- *did* this symbol occur | what Algorithm 2 decodes |
| **histogram** $h$ | a count --- *how many* carried it | what Stage A hands back |

SS IV-B never uses the phases. And for a *single* codeword the last two coincide, which
is the cleanest way to see what bundling actually does to the data.
""")
    W = core.ks_matrix(ctx, codeword)
    st.code(report.ks_section(ctx, msg, codeword, W), language="text")
    assert W.sum() == N and (W.sum(axis=0) == 1).all()

    h1 = core.histograms(ctx, hv)
    st.code(report.histogram_section(ctx, h1, W), language="text")
    assert np.array_equal(h1, W), "Stage A on one codeword must give its one-hot table"

# ================================================================================
# 2 -- Bundle
# ================================================================================
with tabs[1]:
    st.markdown("Bundling is **addition**: lay the hypervectors on top of each other "
                "and add, entry by entry. No randomness — a fixed list of messages, "
                f"bundled $t = 1 \\ldots {T_LIMIT}$, each bundle containing the one "
                "before.")

    msgs = core.message_list(ctx, T_LIMIT)
    acc = np.zeros(DIM, dtype=complex)
    brows = []
    for t in range(1, T_LIMIT + 1):
        acc = acc + core.encode(ctx, msgs[t - 1])   # one at a time; see core.bundle
        brows.append({"t": t, "abs_min": float(np.abs(acc).min()),
                      "abs_max": float(np.abs(acc).max()),
                      "norm": float(np.linalg.norm(acc))})

    st.code(report.bundle_section(ctx, brows, T_LIMIT, U_MAX), language="text")

    lines = [f"the fixed list, first {min(6, T_LIMIT)} of {T_LIMIT}:",
             f"   {'i':>3} {'message':>14}   codeword (first {min(10, N)} of {N})"]
    for i in range(min(6, T_LIMIT)):
        lines.append(f"   {i:>3} {str(msgs[i]):>14}   "
                     f"{core.rs_encode(ctx, msgs[i])[:min(10, N)]}")
    st.code("\n".join(lines), language="text")

    norm_df = pd.concat([
        pd.DataFrame(brows)[["t", "norm"]].rename(columns={"norm": "value"})
            .assign(series="measured norm"),
        pd.DataFrame({"t": [r["t"] for r in brows],
                      "value": [np.sqrt(r["t"] * DIM) for r in brows],
                      "series": "sqrt(t·dim)"})])
    st.altair_chart(two_series(norm_df, "norm of the bundle",
                               "The bundle grows like sqrt(t), not like t",
                               ["measured norm", "sqrt(t·dim)"]),
                    width="stretch", theme=None)

# ================================================================================
# 3 -- Decode
# ================================================================================
with tabs[2]:
    st.markdown("""
Unbundling has two stages, and the histogram is the thing in between.

**Stage A (§IV-A).** Undo $\\phi$ and the Hadamard layer. The bundle splits into $N$
blocks of $N$ entries; block $j$ is $Hh_j + e_j$, so one matrix multiply and a rounding
inverts it — the paper's **Corollary 1**:

$$h_j = \\operatorname{round}\\!\\left(\\tfrac{1}{N}\\,\\Re\\!\\left(H^\\dagger s_j\\right)\\right)$$

**Stage B (§IV-B).** Throw the counts away, keep only *which* symbols appeared ($Y$).
Interpolate a curve through every observed $(j, a)$ pair, factor out $y - f(x)$, and
keep the candidates that score the maximum.

This is the notebook's **by hand** walkthrough — pick $t$ and every step is printed.
""")

    t = int_slider("**t** — how many codewords to bundle", 1, T_LIMIT, min(3, U_MAX),
                   key="decode_t")
    msgs = core.message_list(ctx, T_LIMIT)
    bundle = core.bundle_of(ctx, msgs[:t])
    hist = core.histograms(ctx, bundle)
    Y = core.support_matrix(hist)

    if st.button("Run the walkthrough", type="primary", key="decode_btn"):
        try:
            with st.spinner(f"interpolating through {int(Y.sum())} points ..."):
                res = run_job({"op": "decode", "p": p, "m": m, "K": K,
                               "messages": msgs[:t], "d": d, "ell_max": ell_max})
        except RuntimeError as exc:
            st.error(f"**The decoder failed.**\n\n```\n{exc}\n```")
            st.stop()
        res["codewords"] = [core.rs_encode(ctx, u) for u in msgs[:t]]
        res["bundle_head"] = bundle[:6]
        res["abs_min"] = float(np.abs(bundle).min())
        res["abs_max"] = float(np.abs(bundle).max())
        heads = [core.encode(ctx, u) for u in msgs[:t]]
        st.code(report.by_hand(ctx, msgs, t, hist, Y, res, hvs_head=heads),
                language="text")
        if res["messages"] == sorted(msgs[:t]):
            st.success(f"**Exact.** All {t} codewords came back, and nothing else.")
        elif res["messages"]:
            st.warning(f"Returned {len(res['messages'])}, not the {t} that went in.")
        else:
            st.error(f"**Nothing came back.** t = {t} is past u_max = {U_MAX}.")
    else:
        st.code(report.show_histogram(ctx, hist,
                                      "STAGE A -- histogram h (counts):"),
                language="text")
        st.caption("Press **Run the walkthrough** to interpolate and factor.")

# ================================================================================
# 4 -- Noise
# ================================================================================
with tabs[3]:
    st.markdown(f"""
The notebook is noiseless. The paper is not: §IV-A writes block $j$ of the received
vector as $s_j = Hh_j^\\top + e_j^\\top$, and asks when the rounding still returns
$h_j$.

**Corollary 2** gives the usable answer: $\\lVert e_j \\rVert_2 < \\sqrt{{N}}/2$ suffices
— here $\\sqrt{{{N}}}/2 = {np.sqrt(N) / 2:.3f}$, per block of ${N}$ entries.
**Corollary 1** is sharper and says the rounding succeeds whenever
$e_j \\in H(-0.5,0.5)^N + i\\,H\\mathbb{{R}}^N$, which the paper notes holds *"even for
some unbounded error vectors"*.

Below, $\\alpha$ is the multiple of the permitted level: $\\alpha < 1$ is inside
Corollary 2's guarantee, $\\alpha \\ge 1$ is outside it. How far outside you can go
depends entirely on **which direction** the noise points, which is what the three
kinds are for.

| kind | $e_j$ | what it tests |
|---|---|---|
| **real** | $Ha^\\top$, one signed coordinate | the direction Lemma 1 actually constrains — tight |
| **isotropic** | complex Gaussian | generic; half the energy is harmless imaginary |
| **imaginary** | $i\\,Hb^\\top$ | the free direction of the Voronoi cell |
""")

    nc1, nc2, nc3 = st.columns([1, 1, 2])
    with nc1:
        noise_t = int_slider("**t** — codewords bundled", 1, T_LIMIT, min(3, U_MAX),
                             key="noise_t")
    with nc2:
        kind = st.radio("**noise direction**", ["real", "isotropic", "imaginary"],
                        key="noise_kind",
                        help="real is the worst case, imaginary the free one")
    with nc3:
        alpha = st.slider("**α** — multiple of the permitted level", 0.0, 5.0, 1.0,
                          step=0.05, key="noise_alpha",
                          help="α < 1 is inside Corollary 2; α ≥ 1 is outside it")

    nmsgs = core.message_list(ctx, T_LIMIT)[:noise_t]
    clean_s = core.bundle_of(ctx, nmsgs)
    clean_hist = core.histograms(ctx, clean_s)
    e = (core.noise_for(ctx, alpha, kind=kind, seed=0) if alpha
         else np.zeros(DIM, dtype=complex))
    noisy_hist = core.histograms(ctx, clean_s + e)
    noisy_Y = core.support_matrix(noisy_hist)

    st.code(report.noise_header(ctx, alpha, kind, core.block_norms(ctx, e)),
            language="text")

    with st.expander("the matrices, before and after φ  —  bundle, noise, and the sum"):
        # Only ever materialise a couple of these: each is an N x N matrix, and at
        # N = 625 with a large t the full list would be gigabytes.
        shown = nmsgs[:min(len(nmsgs), 3)]
        exps = [ctx["L"][:, core.rs_encode(ctx, u)].T for u in shown]
        hvs = [core.encode(ctx, u) for u in shown]
        # row j of the reshaped hypervector is block j, and it is phi applied to row j
        # of the exponent grid -- the two views below really are the same object
        assert np.allclose(hvs[0].reshape(N, N),
                           ctx["enc"]["phi"](exps[0], p)), "phi should line these up"
        st.code(report.phi_sections(ctx, shown, exps, hvs, clean_s, e,
                                    clean_s + e, alpha, kind, total=len(nmsgs)),
                language="text")
    st.code(report.show_histogram(
        ctx, noisy_hist, "STAGE A -- histogram h recovered from the NOISY bundle:",
        max_rows=14, against=clean_hist), language="text")

    if np.array_equal(noisy_hist, clean_hist):
        st.success("**Stage A is exact** — the rounding absorbed the noise entirely, "
                   "so Stage B sees the same Y it would have without it.")
    else:
        st.warning(f"**Stage A is wrong** in "
                   f"{int((noisy_hist != clean_hist).sum())} cells. Whether that "
                   f"breaks recovery is a separate question — Stage B has its own "
                   f"slack, and at small t it often absorbs the damage.")

    if st.button("Decode the noisy bundle", type="primary", key="noise_btn"):
        try:
            with st.spinner("interpolating ..."):
                res = run_job({"op": "decode", "p": p, "m": m, "K": K,
                               "messages": nmsgs, "d": d, "ell_max": ell_max,
                               "alpha": alpha, "kind": kind, "seed": 0})
        except RuntimeError as exc:
            st.error(f"**The decoder failed.**\n\n```\n{exc}\n```")
            st.stop()
        res["codewords"] = [core.rs_encode(ctx, u) for u in nmsgs]
        res["bundle_head"] = (clean_s + e)[:6]
        res["abs_min"] = float(np.abs(clean_s + e).min())
        res["abs_max"] = float(np.abs(clean_s + e).max())
        heads = [core.encode(ctx, u) for u in nmsgs]
        st.code(report.by_hand(ctx, nmsgs, noise_t, noisy_hist, noisy_Y, res,
                               hvs_head=heads), language="text")
        if res["messages"] == sorted(nmsgs):
            st.success(f"**Still exact at α = {alpha:g}.**")
        else:
            st.error(f"**Broken at α = {alpha:g}** — {len(res['messages'])} of "
                     f"{noise_t} came back.")

    st.divider()
    st.markdown("##### Up to and beyond the permitted level")
    st.caption("Decode the same bundle at a ladder of noise levels and watch both "
               "stages. Each α is a full decode.")

    ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
    nkey = (p, m, K, d, ell_max, noise_t, kind)
    nstore = st.session_state.setdefault("noise_sweeps", {})
    if st.button("Run the α ladder", key="noise_sweep_btn") or nkey in nstore:
        if nkey not in nstore:
            bar = st.progress(0.0, text="decoding at each α ...")
            try:
                out = run_job({"op": "noise_sweep", "p": p, "m": m, "K": K,
                               "messages": nmsgs, "alphas": ALPHAS, "kind": kind,
                               "seed": 0, "d": d, "ell_max": ell_max},
                              on_progress=lambda i, n: bar.progress(
                                  i / n, text=f"α {i} of {n}"))
            except RuntimeError as exc:
                bar.empty()
                st.error(f"**The α ladder failed.**\n\n```\n{exc}\n```")
                st.stop()
            bar.empty()
            nstore[nkey] = out["rows"]
        nrows = nstore[nkey]

        lines = [f"noise direction '{kind}',  t = {noise_t},  "
                 f"permitted ||e_j||_2 < {np.sqrt(N) / 2:.3f}", "",
                 f"{'alpha':>7} {'||e_j||':>9} {'stage A':>9} {'bad cols':>9} "
                 f"{'returned':>9} {'exact?':>8}", "-" * 56]
        for r in nrows:
            lines.append(f"{r['alpha']:>7.2f} {r['block_norm']:>9.3f} "
                         f"{('exact' if r['stage_a_ok'] else 'WRONG'):>9} "
                         f"{r['bad_cols']:>9} {r['n_rec']:>9} "
                         f"{str(r['exact']):>8}")
        a_ok = [r["alpha"] for r in nrows if r["stage_a_ok"]]
        b_ok = [r["alpha"] for r in nrows if r["exact"]]
        lines += ["",
                  f"Stage A last exact at alpha = {max(a_ok) if a_ok else 0:g}",
                  f"recovery last exact at alpha = {max(b_ok) if b_ok else 0:g}",
                  f"Corollary 2 only promises alpha < 1."]
        st.code("\n".join(lines), language="text")

        ndf = pd.concat([
            pd.DataFrame({"t": [r["alpha"] for r in nrows],
                          "value": [int(r["stage_a_ok"]) for r in nrows],
                          "series": "Stage A histogram exact"}),
            pd.DataFrame({"t": [r["alpha"] for r in nrows],
                          "value": [int(r["exact"]) for r in nrows],
                          "series": "recovery exact"})])
        st.altair_chart(two_series(
            ndf, "1 = exact, 0 = wrong",
            f"Where '{kind}' noise breaks each stage  (t = {noise_t}, N = {N})",
            ["Stage A histogram exact", "recovery exact"],
            rule_at=1.0, rule_label="α = 1, Corollary 2's limit",
            x_title="noise level α  (multiples of the permitted √N/2)"),
            width="stretch", theme=None)

        if kind == "imaginary" and all(r["exact"] for r in nrows):
            st.info("Nothing breaks, at any α on this ladder. Lemma 1's Voronoi cell "
                    "is $Hh^\\top + H(-0.5,0.5)^N + i\\,H\\mathbb{R}^N$ — **unbounded** "
                    "in the imaginary direction — so noise pointing that way never "
                    "moves the rounding, however large it gets. This is exactly the "
                    "paper's remark that decoding succeeds even for some unbounded "
                    "error vectors, and it is why Corollary 2's norm bound is "
                    "sufficient but nowhere near necessary.")
        elif kind == "real":
            st.info("This is the direction Lemma 1 actually constrains: $a$ has one "
                    "coordinate of size α/2, so $\\lVert a\\rVert_\\infty$ crosses "
                    "0.5 exactly when α reaches 1. **Corollary 2 is tight here** — "
                    "the histogram is perfect at α = 0.99 and wrong at α = 1.")
        else:
            st.info("Generic noise survives well past the guarantee: half its energy "
                    "is in the harmless imaginary part, and the real half spreads "
                    "over all $N$ coordinates, so $\\lVert a\\rVert_\\infty$ stays "
                    "far below $\\lVert a\\rVert_2$. Corollary 2 bounds the 2-norm "
                    "but only the ∞-norm matters.")

    _grid_section(p, m, K, d, ell_max, N, U_MAX, T_LIMIT, ctx)

# ================================================================================
# 5 -- Gaussian noise
# ================================================================================
with tabs[4]:
    st.markdown(r"""
Everything above is noiseless. A real system that recovers a bundled hypervector does
not hand you a clean sum — it perturbs **every entry independently**, and the
perturbation is Gaussian. So before Stage A, add

$$e \sim \mathcal{CN}(0,\ \sigma^2 I), \qquad s \;=\; \sum_{i=1}^{t} c_i \;+\; e$$

The covariance is $\sigma^2$ times the **identity**: white noise, every entry
independent, no preferred direction. Note $\sigma^2$ and not $\sigma$ — it is a
variance — and $\mathcal{CN}$ rather than $\mathcal{N}$, since each entry's real and
imaginary parts carry $\sigma^2/2$ each. $\sigma$ is in units of the signal, because a
codeword entry has $\lvert\text{entry}\rvert = 1$.
""")

    gsig = st.slider("**σ** — noise level per entry", 0.0, 2.0, 1.0, step=0.05,
                     key="g_sigma",
                     help="sigma = 1 is noise as large as the signal it corrupts")
    gt = int_slider("**t** — codewords bundled", 1, T_LIMIT, min(3, U_MAX), key="g_t")

    gmsgs = core.message_list(ctx, gt)
    gclean = core.bundle_of(ctx, gmsgs)
    gh0 = core.histograms(ctx, gclean)
    ge = core.gaussian_noise(ctx, gsig, seed=0) if gsig else np.zeros(DIM, complex)
    gnoisy = gclean + ge
    ghist = core.histograms(ctx, gnoisy)
    gY = core.support_matrix(ghist)

    E_s = (np.array([core.gaussian_noise(ctx, gsig, seed=sd) for sd in range(120)])
           if gsig else np.zeros((1, DIM), complex))
    probe = (core.block_norms(ctx, core.noise_for(ctx, 2 * gsig, kind="isotropic",
                                                  seed=0))
             if gsig else np.zeros(N))
    st.code(report.gaussian_model(ctx, gsig, E_s, core.block_norms(ctx, ge), probe,
                                  2 * gsig), language="text")

    with st.expander("the matrices, before and after φ  —  bundle, noise, and the sum"):
        shown = gmsgs[:min(len(gmsgs), 3)]
        st.code(report.phi_sections(
            ctx, shown, [ctx["L"][:, core.rs_encode(ctx, u)].T for u in shown],
            [core.encode(ctx, u) for u in shown], gclean, ge, gnoisy,
            gsig, "gaussian", total=len(gmsgs)), language="text")

    st.code(report.show_histogram(
        ctx, ghist, f"STAGE A -- histogram from the NOISY bundle (t = {gt}):",
        max_rows=14, against=gh0), language="text")

    if np.array_equal(ghist, gh0):
        st.success("**Stage A is exact** at this σ — the rounding absorbed the noise.")
    else:
        st.warning(f"**Stage A is wrong** in {int((ghist != gh0).sum())} cells. "
                   f"Stage B may still absorb it.")

    if st.button("Decode this noisy bundle", type="primary", key="g_decode_btn"):
        try:
            with st.spinner("interpolating ..."):
                res = run_job({"op": "decode", "p": p, "m": m, "K": K,
                               "messages": gmsgs, "d": d, "ell_max": ell_max,
                               "sigma": gsig, "seed": 0})
        except RuntimeError as exc:
            st.error(f"**The decoder failed.**\n\n```\n{exc}\n```")
            st.stop()
        res["codewords"] = [core.rs_encode(ctx, u) for u in gmsgs]
        res["bundle_head"] = gnoisy[:6]
        res["abs_min"] = float(np.abs(gnoisy).min())
        res["abs_max"] = float(np.abs(gnoisy).max())
        st.code(report.by_hand(ctx, gmsgs, gt, ghist, gY, res,
                               hvs_head=[core.encode(ctx, u) for u in gmsgs],
                               max_rows=14), language="text")
        if res["messages"] == sorted(gmsgs):
            st.success(f"**Exact at σ = {gsig:g}.**")
        else:
            st.error(f"**Broken at σ = {gsig:g}** — {len(res['messages'])} of {gt} back.")

    st.divider()
    st.markdown("##### The grid")

    grids = available_grids()
    if len(grids) > 1:
        pick = st.selectbox("**which code?**", list(grids),
                            index=list(grids.values()).index("noise_trials")
                            if "noise_trials" in grids.values() else 0,
                            key="g_grid",
                            help="grids built by noise_trials.py for other (p, m, K)")
        trials = load_trials(grids[pick])
    else:
        trials = load_trials()
    if trials is None:
        st.info("No precomputed trials yet. Build them with\n\n"
                "```\nuv run --group sage --group app python "
                "streamlit_app/noise_trials.py\n```")
    else:
        same = (trials["p"], trials["m"], trials["K"]) == (p, m, K)
        st.caption(f"p={trials['p']}, m={trials['m']}, K={trials['K']}, "
                   f"N={trials['N']}, d={trials['d']} — {trials['trials']:,} "
                   f"independent noise draws per cell. The codewords are fixed; only "
                   f"the noise is redrawn."
                   + ("" if same else "  **Note: this grid is for a different code "
                                      "than the sidebar's.**")
                   + ("" if trials.get("complete") else "   *(partial run)*"))

        # This grid is built at one fixed d (d = 1 for the shipped one). Past u_max its
        # rows are identically zero, which looks like a noise result and is not -- it is
        # the y-degree ceiling, and it moves with d. Say so here, using whatever
        # gs_bound.ipynb has measured, rather than leaving a blank band.
        gf = DATA_DIR / f"gs_bound_{trials['p']}_{trials['m']}_{trials['K']}.jsonl"
        gmeas = measured_kernels(trials["p"], trials["m"], trials["K"],
                                 gf.stat().st_mtime if gf.exists() else 0)
        if gmeas:
            gW = trials["K"] - 1
            reach = {}
            for key, row in gmeas.items():
                dd, tt = (int(x) for x in key.split(","))
                if row["exact"]:
                    reach[dd] = max(reach.get(dd, 0), tt)
            with st.expander(
                    f"This grid is **d = {trials['d']}**. Why the rows past "
                    f"t = {trials['u_max']} are all zero — and what changes at higher d",
                    expanded=False):
                lines = [f"{'d':>3} {'ell_sat':>8} {'max t decoded':>14}   noiseless",
                         "-" * 48]
                for dd in sorted(reach):
                    es = dd * trials["N"] // gW
                    lines.append(f"{dd:>3} {es:>8} {reach[dd]:>14}   "
                                 + ("past u_max" if reach[dd] > trials["u_max"] else ""))
                st.code("\n".join(lines), language="text")
                st.markdown(
                    f"At **d = {trials['d']}** the budget forces "
                    f"`deg_y Q <= {trials['d'] * trials['N'] // gW}`, so `Q` cannot "
                    f"*represent* more than that many codewords — the kernel is empty "
                    f"at every sigma, including **sigma = 0**. That is the bundling "
                    f"cliff (u_max), not the noise cliff, and the two are independent.\n\n"
                    "Raising `d` lifts the ceiling, but **this grid cannot show it**: "
                    "it was generated at one fixed `d`. A higher-`d` grid is a separate "
                    "run of `noise_trials.py`, and an expensive one — roughly 7s per "
                    "decode at d=2, 55s at d=3, 300s at d=4, against 0.3s at d=1.\n\n"
                    "`core.unbundle` now always tries `ell_sat = dN/(K-1)` as a "
                    "last rung, so a bracket tuned at d=1 no longer hides the "
                    "higher-`d` behaviour.")

        # One d at a time. The grids were built by different runs -- the 1000-trial
        # d=1 sweep, a 30-trial d=2 sweep, and cliff_probe.py's handful of decodes at
        # d=2..6 -- so the rows for a given d can come from several files. They are
        # unioned here, preferring whichever run carries more trials per cell.
        layers_all = all_d_layers(_grid_mtime(), p, m, K)
        if layers_all.empty:
            # Grids are per-code and only (3,3,3) ships with any. Showing another
            # code's numbers here was the old behaviour and it was silently wrong:
            # the heatmap looked populated while describing a different code.
            st.warning(
                f"No trial grids for **p={p}, m={m}, K={K}** yet — the shipped "
                f"grids are for p=3, m=3, K=3. Build some with\n\n"
                f"```\nP={p} M={m} K={K} D_LIST=1 T_LIST=1,2,3 SIGMAS=0,1.2 "
                f"TRIALS=5 \\\n  uv run --group sage --group app python "
                f"streamlit_app/cliff_probe.py\n```\n"
                f"The panels above this one are computed live and are already "
                f"showing p={p}, m={m}, K={K}.")
            st.stop()
        d_opts = sorted(int(v) for v in layers_all["d"].unique())
        gd_pick = st.radio(
            "multiplicity **d**", d_opts, horizontal=True, key="g_dsel",
            index=d_opts.index(trials["d"]) if trials["d"] in d_opts else 0,
            help="Each d is a separate experiment and they cost wildly different "
                 "amounts -- ~0.3s a decode at d=1 against ~350s at d=5 -- so the "
                 "higher layers are sparse. Cells never run are marked, not blank.")
        tdf = layers_all[layers_all["d"] == gd_pick].copy()
        # `exact` is all-or-nothing -- 9 of 10 codewords back scores the same as none.
        # The grids record n_correct as well, so the graded view costs nothing to
        # show, and the two differ sharply in the sigma ~ 1.4-2.0 band where most
        # failures are partial rather than empty. (all_d_layers computes all three.)

        _opts = ["mean recall", "recovery exact", "Stage A histogram exact"]
        if "mean recall (relaxed)" in tdf and tdf["mean recall (relaxed)"].notna().any():
            _opts.insert(1, "mean recall (relaxed)")
        which = st.radio("show", _opts, horizontal=True, key="g_which")
        st.caption({
            "mean recall":
                "Fraction of the bundled codewords returned, averaged over trials — a "
                "trial returning 9 of 10 scores 0.9. Partial recovery is visible here "
                "and nowhere else.",
            "recovery exact":
                "Fraction of trials returning the bundle **exactly**. All-or-nothing: "
                "returning 9 of 10 scores the same as returning none.",
            "Stage A histogram exact":
                "Fraction of trials whose histogram came back exactly right, before "
                "Stage B runs at all.",
            "mean recall (relaxed)":
                "Same decode, but keeping candidates that score at least "
                "**d(N − s)** instead of **d·N** — histo.pdf §IV-B2, with s read "
                "off the histogram column sums. The strict rule discards correct "
                "codewords that lost a symbol in Stage A; this is what the decoder "
                "actually found.",
        }[which])
        # Canonical axes, identical for every d. Deriving them from whichever rows
        # happen to exist made each d a different shape, so the grids could not be
        # compared by eye and a sparse d looked like a small experiment rather than
        # a mostly-unmeasured one. The full rectangle is the honest frame; what is
        # missing is marked.
        t_hi_max = max(15, int(layers_all["t"].max()) if len(layers_all) else 15)
        t_lo, t_hi = st.slider("**t** range", 1, t_hi_max, (1, t_hi_max),
                               key="g_trange")
        s_opts = sorted(layers_all["sigma"].unique()) if len(layers_all) \
            else sorted(tdf["sigma"].unique())
        s_lo, s_hi = st.select_slider(
            "**σ** range", options=[f"{v:g}" for v in s_opts],
            value=(f"{s_opts[0]:g}", f"{s_opts[-1]:g}"), key="g_srange")

        # Complete the rectangle before plotting. A cell with no data is not the
        # same as a cell measured at zero, and leaving it blank makes the two look
        # identical -- which at d >= 4, where most of the grid was never run, would
        # read as "the decoder fails everywhere up here".
        keep_sig = [v for v in s_opts if float(s_lo) <= v <= float(s_hi)]
        grid = pd.MultiIndex.from_product(
            [range(t_lo, t_hi + 1), keep_sig], names=["t", "sigma"]).to_frame(index=False)
        _cols = ["t", "sigma", "n", "mean recall", "recovery exact",
                 "Stage A histogram exact"]
        for _c in ("mean recall (relaxed)", "recovery exact (relaxed)",
                   "spurious per decode", "mean s"):
            if _c in tdf:
                _cols.append(_c)
        if "sec per decode" in tdf:
            _cols.append("sec per decode")
        sel = grid.merge(tdf[_cols], on=["t", "sigma"], how="left")
        for _c in ("sec per decode", "mean recall (relaxed)",
                   "recovery exact (relaxed)", "spurious per decode", "mean s"):
            if _c not in sel:
                sel[_c] = float("nan")
        sel["missing"] = sel["n"].isna()
        sel["n"] = sel["n"].fillna(0).astype(int)
        sel["sigma_lab"] = [f"{v:g}" for v in sel["sigma"]]
        sel["t_lab"] = [f"{v}" + ("  ◂ u_max" if v == trials["u_max"] else "")
                        for v in sel["t"]]
        s_order = [f"{v:g}" for v in s_opts if float(s_lo) <= v <= float(s_hi)]
        t_order = [f"{v}" + ("  ◂ u_max" if v == trials["u_max"] else "")
                   for v in range(t_lo, t_hi + 1)]

        # click-to-inspect: the grid stores only counts, so a selected cell is
        # replayed trial by trial through the worker (see the block below the chart)
        cell_sel = alt.selection_point(fields=["t", "sigma"], name="cell", toggle=False)
        enc_xy = dict(
            x=alt.X("sigma_lab:O", title="noise level  σ", sort=s_order),
            y=alt.Y("t_lab:O", title="codewords bundled,  t",
                    sort=list(reversed(t_order))))
        sel["status"] = ["not run" if mi else f"{int(n)} decodes"
                         for mi, n in zip(sel["missing"], sel["n"])]
        # pre-formatted, because Altair cannot render "n/a" through a numeric format
        sel["avg sec"] = [human_time(v) if v == v and v else "—"
                          for v in sel["sec per decode"]]
        tips = [alt.Tooltip("t:Q", title="t"),
                alt.Tooltip("sigma:Q", title="sigma"),
                alt.Tooltip("status:N", title="status"),
                alt.Tooltip("mean recall:Q", format=".3f"),
                alt.Tooltip("recovery exact:Q", format=".3f"),
                alt.Tooltip("Stage A histogram exact:Q", format=".3f"),
                alt.Tooltip("avg sec:N", title="avg time/decode"),
                alt.Tooltip("mean recall (relaxed):Q", format=".3f"),
                alt.Tooltip("spurious per decode:Q", format=".2f"),
                alt.Tooltip("mean s:Q", title="corrupted columns", format=".1f")]
        done = sel[~sel["missing"]]
        gone = sel[sel["missing"]]
        heat = alt.Chart(done).mark_rect(
            stroke=C["surface"], strokeWidth=2).encode(
            color=alt.Color(f"{which}:Q", title=which,
                            scale=alt.Scale(range=C["heat"], domain=[0, 1])),
            **enc_xy)
        # unmeasured cells: flat surface fill plus a label, so they read as absent
        # rather than as a measured zero
        blank = alt.Chart(gone).mark_rect(
            stroke=C["surface"], strokeWidth=2,
            fill=C.get("surface2", "#2a2a2a"), fillOpacity=0.55).encode(**enc_xy)
        blank_txt = alt.Chart(gone).mark_text(
            fontSize=9, fontStyle="italic",
            color=C.get("muted", "#8a8a8a")).encode(
            text=alt.value("not run"), **enc_xy)
        # A transparent hit layer over EVERY cell carries the click. Putting the
        # selection on the measured layer alone made unmeasured cells unclickable --
        # precisely the ones worth launching a run from.
        hit = alt.Chart(sel).mark_rect(opacity=0).encode(
            tooltip=tips, **enc_xy).add_params(cell_sel)
        gevent = st.altair_chart(
            styled(blank + blank_txt + heat + hit,
                   height=max(260, 26 * (t_hi - t_lo + 1))),
            width="stretch", theme=None, key="g_heat", on_select="rerun")
        if len(gone):
            st.caption(f"**{len(gone)} of {len(sel)} cells not run** at d={gd_pick}. "
                       f"Decodes cost ~0.3s at d=1 and ~350s at d=5, so the higher "
                       f"multiplicities were probed only where the cliff is.")

        n_max = int(tdf["n"].max()) if len(tdf) else 0
        head = (f"d = {gd_pick} — {which}"
                + (", correct codewords / t" if which == "mean recall" else "")
                + f"  (up to {n_max:,} decodes per cell; 'not run' means no data)")
        lines = [head, "",
                 f"{'t':>4} " + " ".join(f"s={v:g}".rjust(8) for v in s_opts)]
        for tv in range(t_lo, t_hi + 1):
            cells = []
            for v in s_opts:
                row = sel[(sel["t"] == tv) & (sel["sigma"] == v)]
                if not len(row) or bool(row["missing"].iloc[0]):
                    cells.append(" not run".rjust(8))
                else:
                    cells.append(f"{row[which].iloc[0]:.3f}".rjust(8))
            lines.append(f"{tv:>4} " + " ".join(cells)
                         + ("  <- u_max" if tv == trials["u_max"] else ""))
        st.code("\n".join(lines), language="text")

        # retired: the diagonal-offset "folder" stack of every d on one set of
        # axes. The offsets made neighbouring cells hard to compare -- the thing
        # it was supposed to help with -- and the thin high-d layers read as
        # confidently as the 1000-trial d=1 sheet. Replaced by the d selector
        # above, which shows one d at a time and marks unmeasured cells.
        # Kept in case a layered view is wanted again.
        # # ---- the stack: one layer per multiplicity d ---------------------------
        # #
        # # The grid above is a single d. The interesting comparison is ACROSS d, and
        # # a row of separate heatmaps makes that hard -- the eye has to carry a
        # # colour from one panel to another. So the layers are drawn into one set of
        # # axes, offset diagonally like sheets in a folder: low d in front, each
        # # higher d stepped up and to the left so its cells stay visible.
        # #
        # # Coordinates are computed by hand rather than left to an ordinal scale,
        # # because an ordinal axis cannot place a cell at a fractional offset.
        # st.divider()
        # st.markdown("#### The stack — every `d` on one set of axes")
        #
        # layers = all_d_layers(_grid_mtime())
        # if layers.empty:
        #     st.info("No grids on disk yet.")
        # else:
        #     lsig = sorted(layers["sigma"].unique())
        #     keep_s = st.multiselect(
        #         "σ columns", [f"{v:g}" for v in lsig],
        #         default=[f"{v:g}" for v in lsig if v in (0.0, 1.2, 1.4)]
        #         or [f"{v:g}" for v in lsig], key="g_stack_sig")
        #     ds = sorted(layers["d"].unique())
        #     keep_d = st.multiselect("layers (d)", ds, default=ds, key="g_stack_d")
        #     tl, th = st.slider("t range", int(layers["t"].min()),
        #                        int(layers["t"].max()),
        #                        (max(10, int(layers["t"].min())),
        #                         int(layers["t"].max())), key="g_stack_t")
        #     step = st.slider("layer offset", 0.0, 1.0, 0.42, 0.02,
        #                      key="g_stack_off",
        #                      help="0 stacks the layers exactly on top of each "
        #                           "other; larger values fan the folder open.")
        #
        #     keep_sv = [float(v) for v in keep_s]
        #     L = layers[layers["sigma"].isin(keep_sv) & layers["d"].isin(keep_d)
        #                & layers["t"].between(tl, th)].copy()
        #     if L.empty:
        #         st.info("Nothing selected.")
        #     else:
        #         sidx = {v: i for i, v in enumerate(sorted(keep_sv))}
        #         didx = {v: i for i, v in enumerate(sorted(keep_d))}
        #         L["si"] = [sidx[v] for v in L["sigma"]]
        #         L["k"] = [didx[v] for v in L["d"]]
        #         # back layers step UP and LEFT, so the front layer is unobscured
        #         L["x1"] = L["si"] - L["k"] * step
        #         L["x2"] = L["x1"] + 0.92
        #         L["y1"] = L["t"] + L["k"] * step
        #         L["y2"] = L["y1"] + 0.92
        #         L["cell"] = [f"d={d}  t={t}  σ={sg:g}"
        #                      for d, t, sg in zip(L["d"], L["t"], L["sigma"])]
        #         # a cell measured once or twice is an anecdote, not a rate
        #         L["thin"] = L["n"] < 3
        #         # draw back-to-front: later rows paint over earlier ones
        #         L = L.sort_values(["k"], ascending=False)
        #
        #         base = alt.Chart(L)
        #         cells = base.mark_rect(stroke=C["surface"], strokeWidth=1.5).encode(
        #             x=alt.X("x1:Q", title="noise level  σ  (layers step left as d grows)",
        #                     # a hand-built nested ternary: the x positions are
        #                     # numeric (so layers can sit at fractional offsets) but
        #                     # the ticks must read as the sigma values they stand for
        #                     axis=alt.Axis(values=list(range(len(sidx))),
        #                                   labelExpr="".join(
        #                                       f"datum.value === {i} ? '{v:g}' : "
        #                                       for v, i in sidx.items()) + "''")),
        #             x2="x2:Q",
        #             y=alt.Y("y1:Q", title="codewords bundled,  t"),
        #             y2="y2:Q",
        #             color=alt.Color(f"{which}:Q", title=which,
        #                             scale=alt.Scale(range=C["heat"], domain=[0, 1])),
        #             opacity=alt.condition(alt.datum.thin, alt.value(0.55),
        #                                   alt.value(1.0)),
        #             tooltip=[alt.Tooltip("cell:N", title=""),
        #                      alt.Tooltip("mean recall:Q", format=".3f"),
        #                      alt.Tooltip("recovery exact:Q", format=".3f"),
        #                      alt.Tooltip("n:Q", title="decodes")])
        #         # one label per layer, riding the top-left corner of the sheet
        #         tags = pd.DataFrame([
        #             {"x": -k * step - 0.12, "y": th + k * step + 1.1, "lab": f"d={d}"}
        #             for d, k in didx.items()])
        #         labels = alt.Chart(tags).mark_text(
        #             align="right", baseline="middle", fontSize=12,
        #             fontWeight="bold", color=C.get("muted", "#888")).encode(
        #             x="x:Q", y="y:Q", text="lab:N")
        #         st.altair_chart(
        #             styled(cells + labels,
        #                    height=max(300, 46 * (th - tl + 1) + 26 * len(didx))),
        #             width="stretch", theme=None, key="g_stack")
        #         st.caption(
        #             f"{which}. Faded cells carry fewer than 3 decodes — at d≥4 a "
        #             f"single decode costs minutes, so those are single "
        #             f"observations, not rates. Hover for the count."
        #             + ("  Rows at t > u_max are blank for the layers that cannot "
        #                "represent them at all." if th > trials["u_max"] else ""))
        #
        # ---- click a cell to replay its trials one at a time --------------------
        st.markdown("---")
        picked = ((gevent.get("selection") or {}).get("cell") or []) if gevent else []
        if not picked:
            st.caption("**Click a cell** in the grid above to replay its individual "
                       "trials — what went in, what Stage A did to it, and what came "
                       "back.")
        else:
            gp, gm, gK = trials["p"], trials["m"], trials["K"]
            gd, gN = gd_pick, trials["N"]          # the d the user is looking at
            pt = int(picked[0]["t"])
            ps = float(picked[0]["sigma"])
            psi = d_sigma_index(trials, ps)
            # look the cell up in the unioned rows for this d -- it may have come
            # from cliff_probe.py rather than from the grid file `trials` points at
            _m = tdf[(tdf["t"] == pt) & (abs(tdf["sigma"] - ps) < 1e-9)]
            cell = _m.iloc[0].to_dict() if len(_m) else None

            st.markdown(f"#### d = {gd}, t = {pt}, σ = {ps:g}")
            if cell:
                st.caption(f"Across {int(cell['n']):,} decodes: "
                           f"**{int(cell['exact'])}** exact, mean recall "
                           f"**{cell['n_correct'] / (cell['n'] * pt):.3f}**, "
                           f"Stage A exact **{int(cell['stage_a_ok'])}**.")
            else:
                st.caption("**Not run yet.** Nothing has been decoded in this cell.")

            # ---- run this cell -------------------------------------------------
            # Costs span two orders of magnitude across d, so the estimate is shown
            # BEFORE the button, from this machine's own recorded timings. Work is
            # handed to cliff_probe.py in a detached process rather than run inline:
            # a single d=5 decode is ~6 minutes, which would hang the whole app.
            _cj = cliff_path(p, m, K)
            costs = decode_costs(_cj.stat().st_mtime if _cj.exists() else 0,
                                 p, m, K)
            one, how = estimate_sec(costs, gd, pt)
            have = int(cell["n"]) if cell else 0
            with st.container(border=True):
                st.markdown("**Run more decodes here**")
                rc1, rc2 = st.columns([1, 2])
                with rc1:
                    want = st.number_input("decodes wanted (total)", 1, 200,
                                           max(have + 1, 3), key="g_runs")
                todo = max(0, int(want) - have)
                par = min(todo, 4) or 1
                with rc2:
                    if ps == 0:
                        st.info("σ = 0 is deterministic — one decode is the whole "
                                "answer, so extra seeds would be identical copies.")
                    st.metric(
                        f"estimated wall time for {todo} more",
                        human_time(todo * one / par) if todo else "nothing to do",
                        help=f"{human_time(one)} per decode ({how}), "
                             f"{par} in parallel. {have} already on disk."
                             + (f"  Past u_max ({trials['u_max']}) this is a FLOOR: "
                                f"a decode that finds nothing walks the whole ell "
                                f"ladder up to ell_sat = {gd * trials['N'] // (trials['K'] - 1)} "
                                f"before giving up, which costs several times a "
                                f"success." if pt > trials["u_max"] else ""))
                run_now = st.button(f"Run {todo} decode(s) at d={gd}, t={pt}, "
                                    f"σ={ps:g}", disabled=todo == 0,
                                    key="g_runbtn", type="primary")
                if run_now:
                    env = dict(os.environ,
                               P=str(p), M=str(m), K=str(K),
                               D_LIST=str(gd), T_LIST=str(pt), SIGMAS=f"{ps:g}",
                               TRIALS=str(int(want)), WORKERS=str(par),
                               BUDGET="86400", MAX_BY_D="", TRIALS_BY_D="")
                    log = DATA_DIR / f"cell_{p}{m}{K}_d{gd}_t{pt}_s{ps:g}.log"
                    with log.open("w") as fh:
                        subprocess.Popen([sys.executable, str(CLIFF_PROBE)],
                                         stdout=fh, stderr=subprocess.STDOUT,
                                         env=env, cwd=str(REPO),
                                         start_new_session=True)
                    st.session_state.setdefault("g_jobs", []).append(
                        {"d": gd, "t": pt, "sigma": ps, "want": int(want),
                         "log": str(log)})
                    st.success(
                        f"Started. cliff_probe.py appends each decode to "
                        f"`cliff_3_3_3.jsonl` as it finishes, so results appear "
                        f"here as they land — rerun the page to pick them up. "
                        f"Log: `{log.name}`")
                for j in st.session_state.get("g_jobs", []):
                    st.caption(f"· queued d={j['d']} t={j['t']} σ={j['sigma']:g} "
                               f"→ {j['want']} decodes")

            # Replay re-decodes trials INLINE through the worker, so it is only
            # offered where decodes already exist. On an unmeasured cell it would
            # silently run the very work the panel above is there to schedule --
            # at d=5 that is six minutes of a frozen page per trial.
            if not cell:
                st.caption("Replay becomes available once this cell has decodes. "
                           "Use the panel above to run some.")
                st.stop()

            per = st.select_slider("trials per page", [1, 3, 5, 10], value=3,
                                   key="g_per")
            # how many trials this particular cell actually has, not the
            # grid-wide figure -- they differ once cliff_probe rows are in
            n_cell = int(cell["n"])
            n_pages = max(1, n_cell // per)
            page = st.number_input(f"page (of {n_pages:,})", 1, n_pages, 1, 1,
                                   key="g_page") - 1
            lo = int(page) * int(per)

            ck = (gp, gm, gK, gd, pt, ps, lo, int(per))
            cache = st.session_state.setdefault("trial_detail", {})
            if ck not in cache:
                with st.spinner(f"replaying trials {lo}–{lo + per - 1} …"):
                    try:
                        cache[ck] = run_job({"op": "trials", "p": gp, "m": gm, "K": gK,
                                             "d": gd, "t": pt, "sigma": ps,
                                             "sigma_i": psi, "lo": lo,
                                             "hi": lo + int(per)})
                    except RuntimeError as exc:
                        st.error(f"replay failed: {exc}")
                        cache[ck] = None
            detail = cache.get(ck)

            if detail:
                for row in detail["rows"]:
                    head = (f"trial {row['seed']}   "
                            f"Stage A: "
                            + ("exact" if row["stage_a_ok"]
                               else f"{row['bad_cells']} cells wrong in "
                                    f"{row['bad_cols']} of {gN} columns")
                            + f"   ->   recovered {row['n_correct']}/{pt}")
                    body = [head, "-" * max(58, len(head)),
                            f"  {'#':>3}  {'bundled message':<22} "
                            f"{'symbol lost at':<22} {'recovered':>9}"]
                    for i, w in enumerate(row["words"], 1):
                        lost = w["lost"]
                        txt = ("—" if not lost else
                               (", ".join(str(j) for j in lost[:4])
                                + (f" (+{len(lost) - 4})" if len(lost) > 4 else "")))
                        body.append(f"  {i:>3}  {str(w['u']):<22} {txt:<22} "
                                    f"{('yes' if w['recovered'] else 'NO'):>9}")
                    if row["spurious"]:
                        body += ["", f"  spurious returned: {row['spurious']}"]
                    st.code("\n".join(body), language="text")

                surv = [(len(w["lost"]) == 0, w["recovered"])
                        for r in detail["rows"] for w in r["words"]]
                agree = sum(1 for a, b in surv if a == b)
                st.caption(
                    f"**Intact support ⇔ recovered** holds for {agree}/{len(surv)} "
                    f"codewords on this page. The lemma needs agreement at *every* "
                    f"position (`d·27 > 27d − 1`), so a codeword losing even one cell "
                    f"forfeits the guarantee — which is why partial recovery exists "
                    f"at all.")



# ==============================================================================
# 6 -- In versus out
# ==============================================================================
# ================================================================================
with tabs[5]:
    st.markdown(r"""
### Spiking noise

AWGN perturbs every coordinate a little; **spiking** noise hits a few coordinates
hard. The paper already has this model, though not under that name. Proposition 2
(§V-A) is stated *per block* — the hypervector is $N$ blocks of $N$, one per codeword
position, and Stage A decodes each independently:

> for non-negative $s, r$ with $(r+1)s < N$, recovery succeeds whenever every $e_i$
> satisfies $\lVert e_i\rVert_2 < \sqrt{N}/2$ **except at most $s$** indices, in which
> $e_i \in E_r$.

So $s$ is **how many** coordinates are badly hit and $r$ is **how badly** — exactly
$\mu$ and $\nu$. Below, $\mu$ spikes of magnitude $\nu$ are added on top of a
Gaussian floor $\sigma$.
""")
    sp = spike_cells(spikes_path(p, m, K).stat().st_mtime
                     if spikes_path(p, m, K).exists() else 0, p, m, K)
    if sp.empty:
        st.warning(
            f"No spiking data for p={p}, m={m}, K={K}. Build some with\n\n"
            f"```\nP={p} M={m} K={K} T_LIST=5 MU=1,2,3,5,8 NU=2,5,10,14,20 \\\n"
            f"  uv run --group sage --group app python streamlit_app/spike_grid.py\n```")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            s_t = st.selectbox("**t** — codewords bundled",
                               sorted(sp["t"].unique()), key="sp_t")
        with c2:
            s_w = st.selectbox("**width** — entries per spike",
                               sorted(sp["width"].unique()), key="sp_w",
                               help="1 is a single coordinate. Width N spreads the "
                                    "spike evenly, which is a lattice direction and "
                                    "behaves very differently.")
        with c3:
            s_sig = st.selectbox("**σ** — Gaussian floor under the spikes",
                                 sorted(sp["sigma"].unique()), key="sp_sig")
        which_s = st.radio("show", ["mean recall", "recovery exact",
                                    "blocks past Corollary 2", "histograms damaged"],
                           horizontal=True, key="sp_which")
        st.caption({
            "mean recall": "Fraction of bundled codewords returned.",
            "recovery exact": "Fraction of trials returning the bundle exactly.",
            "blocks past Corollary 2":
                "Mean number of blocks with ||e_i|| >= sqrt(N)/2 — the paper's s. "
                "Compare against the next option: most of these still decode fine.",
            "histograms damaged":
                "Mean number of histograms that actually came out wrong. Far smaller "
                "than the count above, which is the measure of how conservative "
                "Corollary 2 is.",
        }[which_s])
        col = {"mean recall": "mean recall", "recovery exact": "recovery exact",
               "blocks past Corollary 2": "s_pred",
               "histograms damaged": "s_true"}[which_s]

        # Canonical axes: the union of every mu and nu measured for this code, so
        # the grid has the same shape whatever t / width / sigma is selected.
        # Deriving the axes from the rows that happen to exist made t=5 and t=10
        # different shapes -- the width sweep contributed extra nu values at t=5
        # only -- which silently hid the gaps and made the two incomparable.
        mu_all = sorted(sp["mu"].unique())
        nu_all = sorted(sp["nu"].unique())
        have = sp[(sp["t"] == s_t) & (sp["width"] == s_w)
                  & (sp["sigma"] == s_sig)].copy()
        full = pd.MultiIndex.from_product([mu_all, nu_all],
                                          names=["mu", "nu"]).to_frame(index=False)
        sel = full.merge(have.drop(columns=["t", "width", "sigma"]),
                         on=["mu", "nu"], how="left")
        sel["missing"] = sel["n"].isna()
        sel["n"] = sel["n"].fillna(0).astype(int)
        if sel["missing"].all():
            st.info("No cells at this combination.")
        else:
            sel["mu_lab"] = sel["mu"].astype(str)
            sel["nu_lab"] = [f"{v:g}" for v in sel["nu"]]
            rng = [0, 1] if col.startswith(("mean", "recovery")) else \
                [0, float(max(1, sel[col].max()))]
            enc_s = dict(
                x=alt.X("nu_lab:O", title="spike magnitude  ν",
                        sort=[f"{v:g}" for v in nu_all]),
                y=alt.Y("mu_lab:O", title="number of spikes  μ",
                        sort=[str(v) for v in reversed(mu_all)]))
            gone_s = sel[sel["missing"]]
            blank_s = alt.Chart(gone_s).mark_rect(
                stroke=C["surface"], strokeWidth=2,
                fill=C.get("surface2", "#2a2a2a"), fillOpacity=0.55).encode(**enc_s)
            blank_t = alt.Chart(gone_s).mark_text(
                fontSize=9, fontStyle="italic",
                color=C.get("muted", "#8a8a8a")).encode(
                text=alt.value("not run"), **enc_s)
            heat = alt.Chart(sel[~sel["missing"]]).mark_rect(
                stroke=C["surface"], strokeWidth=2).encode(
                color=alt.Color(f"{col}:Q", title=which_s,
                                scale=alt.Scale(range=C["heat"], domain=rng)),
                **enc_s,
                tooltip=[alt.Tooltip("mu:Q", title="mu"),
                         alt.Tooltip("nu:Q", title="nu"),
                         alt.Tooltip("mean recall:Q", format=".3f"),
                         alt.Tooltip("recovery exact:Q", format=".3f"),
                         alt.Tooltip("s_pred:Q", title="blocks past Cor 2",
                                     format=".1f"),
                         alt.Tooltip("s_true:Q", title="histograms wrong",
                                     format=".1f"),
                         alt.Tooltip("r:Q", title="worst hist error", format=".1f"),
                         alt.Tooltip("outside_E:Q", title="blocks outside E_r"),
                         alt.Tooltip("n:Q", title="decodes")])
            st.altair_chart(
                styled(blank_s + blank_t + heat, height=max(260, 34 * len(mu_all))),
                width="stretch", theme=None, key="sp_heat")
            n_miss = int(sel["missing"].sum())
            if n_miss:
                st.caption(f"**{n_miss} of {len(sel)} cells not run** at "
                           f"t={s_t}, width={s_w}, σ={s_sig:g}.")

            lines = [f"{which_s} — t={s_t}, width={s_w}, sigma={s_sig:g}", "",
                     f"{'mu':>4} " + " ".join(f"ν={v:g}".rjust(8) for v in nu_all)]
            for mu in mu_all:
                cells = []
                for nu in nu_all:
                    row = sel[(sel["mu"] == mu) & (sel["nu"] == nu)]
                    cells.append(" not run".rjust(8)
                                 if not len(row) or bool(row["missing"].iloc[0])
                                 else f"{row[col].iloc[0]:.2f}".rjust(8))
                lines.append(f"{mu:>4} " + " ".join(cells))
            st.code("\n".join(lines), language="text")

        # the two findings the grid is there to make visible
        tot_blocks = int(sp["s_pred"].mul(sp["n"]).sum())
        tot_out = int(sp["outside_E"].sum())
        st.divider()
        a, b = st.columns(2)
        a.metric("spiked blocks outside E_r",
                 f"{tot_out:,} / {tot_blocks:,}" if tot_blocks else "—",
                 help="Proposition 2 requires the exceptional blocks to lie in E_r. "
                      "A spike is concentrated in one coordinate; a lattice point Hv "
                      "is spread over all N because H has unit-magnitude entries. "
                      "They are essentially never close, so the guarantee does not "
                      "cover spiking noise at all — yet the decoder still works.")
        b.metric("Corollary 2 threshold vs the real onset",
                 f"{np.sqrt(N)/2:.2f}  vs  {N/2:g}",
                 help="A single spike of magnitude A moves each of that block's N "
                      "histogram entries by A/N, so rounding only flips at A > N/2. "
                      "Corollary 2 gives up at sqrt(N)/2 — conservative by sqrt(N).")
        st.caption(
            "Damage is governed by whether a histogram entry crosses **zero**, not by "
            "the block norm. `Y = (hist > 0)`, so a perturbation that *raises* an "
            "entry adds a point to `Y` and costs nothing — every codeword still "
            "agrees everywhere. Only a perturbation that drives an entry down to 0 "
            "erases support and kills the codewords that needed it. See FINDINGS.md §9.")


with tabs[6]:
    st.markdown("Run the whole of Part 3 for every $t$ and plot how many went in "
                "against how many came back.")

    # u_max scales with N, so at the larger (p, m) a full sweep is hundreds of decodes
    # and the late ones dominate -- measured, a single t costs roughly t^2. Let the
    # sweep have its own ceiling rather than inheriting T_LIMIT blindly.
    cap = int(min(T_LIMIT, 40))
    sweep_to = int_slider("**sweep t = 1 …**", 2, T_LIMIT, cap, key="sweep_to",
                          help="Each t is a full decode. Cost grows about as t^2, so "
                               "the last few dominate the total.")

    key = (p, m, K, d, ell_max, sweep_to)
    store = st.session_state.setdefault("sweeps", {})

    # calibrated off measured runs: N=81 took 2.3s at t=3 and 15.8s at t=8
    est = sum(0.25 * (tt ** 2) / 10 for tt in range(1, sweep_to + 1)) * (N / 81) ** 0.5
    st.caption(f"t = 1 … {sweep_to}"
               + (f" (u_max = {U_MAX}, so this stops short of the cliff)"
                  if sweep_to <= U_MAX else f", past u_max = {U_MAX}")
               + f". Failures are the slow ones — each tries every ell up to "
                 f"{ell_max} before giving up.")
    if est > 60:
        st.warning(f"Rough estimate: **{est / 60:.0f}–{est / 20:.0f} minutes** at these "
                   f"settings. Lower the sweep ceiling, or ell_max, to cut it down.")
    if st.button("Run the sweep", type="primary", key="sweep_btn") or key in store:
        if key not in store:
            bar = st.progress(0.0, text="bundling and decoding ...")
            t0 = time.time()
            try:
                out = run_job(
                    {"op": "sweep", "p": p, "m": m, "K": K, "t_limit": sweep_to,
                     "d": d, "ell_max": ell_max},
                    on_progress=lambda i, n: bar.progress(
                        i / n, text=f"t = {i} of {n} — {time.time() - t0:.0f}s elapsed"))
            except RuntimeError as exc:
                bar.empty()
                st.error(f"**The sweep failed.**\n\n```\n{exc}\n```")
                st.stop()
            bar.empty()
            store[key] = out["rows"]
        rows = store[key]

        sdf = pd.DataFrame(rows)
        ok = [r["t"] for r in rows if r["exact"]]
        last_ok = max(ok) if ok else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("last t fully recovered", last_ok)
        m2.metric("u_max = (N-1)/(K-1)", U_MAX)
        m3.metric("", "on the nose" if last_ok == U_MAX else f"off by {last_ok - U_MAX}",
                  help="the theory predicts the wall lands exactly at u_max")

        io_df = pd.concat([
            sdf[["t"]].assign(value=sdf["t"], series=IN_LABEL),
            sdf[["t"]].assign(value=sdf["n_rec"], series=OUT_LABEL)])
        st.altair_chart(two_series(
            io_df, "number of codewords",
            f"In versus out  (p={p}, m={m}, K={K}, N={N}, d={d})",
            [IN_LABEL, OUT_LABEL], rule_at=U_MAX, rule_label=f"u_max = {U_MAX}"),
            width="stretch", theme=None)

        st.caption(f"The two lines sit on top of each other for t = 1 … {last_ok}, then "
                   f"part company. t = {last_ok} returns all {last_ok}; "
                   f"t = {last_ok + 1} returns none.")
        st.code(report.sweep_table(ctx, rows, U_MAX), language="text")

        with st.expander("Can a bigger ell rescue it?"):
            st.markdown(f"""
The obvious thing to try past the wall is a larger $\\ell$, the $y$-degree cap, in case
the curve simply needed more room. It cannot, and the reason is easy to miss: **$\\ell$
is capped by the degree budget, not by the number you pass in.** A monomial
$x^i y^j$ is admitted when $i + j(K-1) <$ budget, so with a budget of $N = {N}$ nothing
with $j > {(N - 1) // (K - 1)}$ ever enters the list regardless of $\\ell$.

Raising the *budget* instead does give non-empty kernels — and zero useful roots.
Guruswami-Sudan's Lemma 4 needs $rt > l$ for $y - f(x)$ to be forced to divide $Q$;
once the budget grows past $N d$ that fails, so an arbitrary kernel element no longer
factors into the true codewords. **A loose budget is actively worse.**
""")
            probe_t = int_slider("t to probe", 1, sweep_to, U_MAX + 1, key="probe_t")
            if st.button("Probe", key="probe_btn"):
                cap = (N - 1) // (K - 1)
                probes = [(N, cap), (N, ell_max), (N, 200),
                          (probe_t * (K - 1), 200), (probe_t * (K - 1) + 1, 200),
                          (probe_t * (K - 1) + 3, 200)]
                try:
                    with st.spinner("interpolating at each (budget, ell) ..."):
                        out = run_job({"op": "attempt", "p": p, "m": m, "K": K, "d": d,
                                       "messages": core.message_list(ctx, probe_t),
                                       "probes": probes})
                except RuntimeError as exc:
                    st.error(f"**The probe failed.**\n\n```\n{exc}\n```")
                    st.stop()
                plines = [f"{'budget':>8} {'ell':>6} {'monomials':>11} "
                          f"{'kernel':>8} {'roots':>7} {'matched':>9}", "-" * 52]
                for r in out["rows"]:
                    plines.append(f"{r['budget']:>8} {r['ell']:>6} "
                                  f"{r['monomials']:>11} {r['kernel dim']:>8} "
                                  f"{r['roots']:>7} {r['matched the bundle']:>9}")
                st.code("\n".join(plines), language="text")
                st.caption("The monomial count does not move with ell, and no budget "
                           "recovers anything past u_max.")
    else:
        st.info("Press **Run the sweep**. Results are kept per parameter set, so "
                "flipping tabs or coming back to a combination you have already run "
                "is instant.")

# ================================================================================
# 7 -- Repeats
# ================================================================================
with tabs[7]:
    st.markdown(f"""
Everything above bundles **distinct** codewords. If the same one goes in twice,
something has to give — and the thing that gives is $Y$.

$Y$ records only *whether* a symbol appeared, not how often. So bundling $a, a, b$
produces exactly the same $Y$ as $a, b$, and one pass of the decoder can only ever
return the distinct ones. §IV-B3 recovers the multiset by iterating: decode, subtract
each recovered codeword's histogram from the residual, repeat until the residual is
zero. For $a, a, b$ that is round 1 → $\\{{a, b\\}}$, round 2 → $\\{{a\\}}$.

The loop length is bounded by the number of **distinct** codewords, not the total —
so the total $t$ is unrestricted, and only the distinct count is capped at
$u_{{max}} = {U_MAX}$.
""")
    n_distinct = int_slider("how many distinct codewords", 1, min(4, U_MAX), 2,
                            key="rep_distinct")
    msgs = core.message_list(ctx, min(max(4, n_distinct), B["size"]))
    counts = []
    rcols = st.columns(n_distinct)
    for i in range(n_distinct):
        with rcols[i]:
            counts.append(st.number_input(f"copies of {msgs[i]}", 1, 30,
                                          3 if i == 0 else 1, key=f"rep{i}"))

    multiset = [msgs[i] for i in range(n_distinct) for _ in range(counts[i])]
    st.caption(f"bundling {len(multiset)} codewords, {n_distinct} distinct")

    if st.button("Peel the multiset apart", type="primary", key="rep_btn"):
        h = core.histograms(ctx, core.bundle_of(ctx, multiset))
        Y_full = core.support_matrix(h)
        Y_set = core.support_matrix(core.histograms(
            ctx, core.bundle_of(ctx, msgs[:n_distinct])))

        q1, q2, q3 = st.columns(3)
        q1.metric("column sums", int(h.sum(axis=0)[0]),
                  help="the TOTAL with multiplicity")
        q2.metric("largest count in any cell", int(h.max()),
                  help="marks where a codeword doubled up")
        q3.metric("Y identical to the distinct-only Y",
                  "yes" if np.array_equal(Y_full, Y_set) else "no",
                  help="which is exactly why one pass cannot see multiplicities")

        try:
            with st.spinner("peeling ..."):
                out = run_job({"op": "peel", "p": p, "m": m, "K": K,
                               "multiset": multiset, "d": d, "ell_max": ell_max})
        except RuntimeError as exc:
            st.error(f"**The peel failed.**\n\n```\n{exc}\n```")
            st.stop()
        recovered, rounds = out["recovered"], out["rounds"]

        plines = [f"bundling the multiset {multiset}", "",
                  f"  column sums: all {int(h.sum(axis=0)[0])}  -> t = "
                  f"{len(multiset)}, the TOTAL with multiplicity",
                  f"  largest count in any cell: {int(h.max())}", ""]
        for i, r in enumerate(rounds, 1):
            plines.append(f"  round {i}: residual holds {r['residual']} codeword(s); "
                          f"decoder returns {len(r['found'])} distinct {r['found']}")
        plines += ["", f"  residual now all zero -> stop after {len(rounds)} rounds",
                   f"  recovered multiset: {sorted(recovered)}"]
        st.code("\n".join(plines), language="text")

        if sorted(recovered) == sorted(multiset):
            st.success(f"Recovered the multiset exactly in {len(rounds)} round(s): "
                       f"{len(recovered)} codewords, {n_distinct} distinct.")
        else:
            st.error(f"Recovered {len(recovered)} of {len(multiset)}. With "
                     f"{n_distinct} distinct codewords this still works; past "
                     f"u_max = {U_MAX} distinct it does not.")

# ================================================================================
# Bounds
# ================================================================================
with tabs[8]:
    st.markdown("Several different numbers get called *capacity*. They are not "
                "interchangeable.")
    st.dataframe(pd.DataFrame([
        {"bound": "d_min", "formula": "N - K + 1", "value": f"{B['d_min']}",
         "question it answers": "how far apart are two codewords"},
        {"bound": "unique radius", "formula": "(d_min - 1) / 2",
         "value": f"{B['unique_radius']}",
         "question it answers": "how many symbol errors classical decoding fixes"},
        {"bound": "Johnson", "formula": "N - sqrt(N(K-1))", "value": f"{B['johnson']:.2f}",
         "question it answers": "how far list decoding can ever reach"},
        {"bound": "mu", "formula": "(K-1) / N", "value": f"{B['mu']:.4f}",
         "question it answers": "how much any two codewords overlap (Prop. 1)"},
        {"bound": "l_max", "formula": "(2N - d_min) / (2(N - d_min))",
         "value": f"{B['l_max']:.2f}",
         "question it answers": "how many bundled items an inner product can detect"},
        {"bound": "u_max", "formula": "floor((N-1) / (K-1))", "value": f"{B['u_max']}",
         "question it answers": "how many distinct bundled items IV-B can reconstruct"},
    ]), hide_index=True, width="stretch")

    st.markdown(f"""
`l_max` and `u_max` are the two that get confused. Detecting whether $X$ is in a bundle
is much easier than listing the bundle's contents, so `u_max > l_max` always — here
{B['l_max']:.2f} against {U_MAX}. **`l_max` is loose; `u_max` is a cliff.**

### Why it falls off exactly at u_max

Two arguments, from two different papers, that turn out to be the same inequality.
All source PDFs are in the repo (`ls *.pdf` — they are gitignored).

**The score argument** — `histo.pdf`, Lemma 2. A spurious codeword can collect at most
$t(K-1)$ of its $N$ symbols from the bundle, because it agrees with each of the $t$ real
ones in at most $K-1$ places (two distinct degree-$<K$ polynomials meet at most $K-1$
times). So as long as $t(K-1) < N$, nothing but the real codewords can score the full
$N$ — and that inequality is exactly $t \\le u_{{max}} = \\lfloor (N-1)/(K-1) \\rfloor
= {U_MAX}$.

**The degree argument** — Sidorenko-Fischer, Theorem 1. The interpolation looks for
$Q$ of weighted degree below the budget $Nd$. The one curve that is guaranteed to
work is $\\prod_{{i=1}}^{{t}} (y - f_i(x))$, of weighted $(1, K-1)$-degree $t(K-1)$.
It fits while $t(K-1) < N$ and does not after. Same inequality.

Both arguments are **sufficient** conditions: each says *if* $t \\le u_{{max}}$ *then*
decoding works. Neither proves the converse, and measurement below shows the converse
is false — at $d \\ge 4$, $t = u_{{max}} + 1$ decodes.
""")

    # ------------------------------------------------------------------------
    # the interpolation system as a function of d
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### The interpolation system, as a function of `d`")
    st.markdown(
        "Whether a curve exists at all is a counting question, and `d` does **not** "
        "cancel out of it. Both sides grow like $d^2$, so which one wins comes down "
        "to constants — and to rank, since dependent rows can open a kernel the count "
        "says is impossible.")

    dsel = st.radio("multiplicity **d** — *goes past the sidebar\'s cap of 4*",
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    index=min(max(d, 1), 8) - 1, horizontal=True, key="b_dsel",
                    help="GS's multiplicity s. Nothing here runs a decode — the "
                         "counting is arithmetic and the kernel column is read from "
                         "measurements already on disk.")

    W = K - 1
    ell_sat = dsel * N // W if W else 0
    unknowns = monomial_count(dsel * N, ell_sat, W) if W else 0
    per_point = dsel * (dsel + 1) // 2
    t_top = min(U_MAX + 3, 40)
    n_ps = observed_points(p, m, K, t_top)
    f_meas = DATA_DIR / f"gs_bound_{p}_{m}_{K}.jsonl"
    meas = measured_kernels(p, m, K, f_meas.stat().st_mtime if f_meas.exists() else 0)

    if meas:
        have_d = sorted({int(k.split(",")[0]) for k in meas})
        have_t = sorted({int(k.split(",")[1]) for k in meas})
        past = sorted({int(k.split(",")[0]) for k, v in meas.items()
                       if v["exact"] and int(k.split(",")[1]) > U_MAX})
        st.caption(
            f"Measured so far: **d = {', '.join(map(str, have_d))}** at "
            f"**t = {have_t[0]}–{have_t[-1]}** (from `gs_bound.ipynb`). "
            + (f"Past u_max = {U_MAX}, exact recovery at **d = "
               f"{', '.join(map(str, past))}**." if past else
               f"Nothing past u_max = {U_MAX} yet."))

    c1, c2, c3 = st.columns(3)
    c1.metric("ell_sat = floor(dN/(K-1))", ell_sat,
              help="Beyond this the monomial count stops growing, so no larger ell "
                   "can help. At d=1 it caps y-degree — and so the number of "
                   "codewords Q can represent — at exactly this.")
    c2.metric("unknowns (monomials)", f"{unknowns:,}")
    c3.metric("equations per point", per_point,
              help="d(d+1)/2 Hasse-derivative conditions per observed point.")

    if ell_sat < U_MAX + 1:
        st.info(f"At d = {dsel}, `deg_y Q <= {ell_sat}`, so **Q cannot represent more "
                f"than {ell_sat} codewords** however the counting goes. That ceiling "
                f"alone stops recovery at t = {ell_sat}.")

    rows = [f"{'t':>4} {'n_P':>6} {'tN':>6} {'equations':>10} {'unknowns':>9} "
            f"{'slack':>7} {'kernel':>7} {'found':>6}", "-" * 68]
    for t in range(1, t_top + 1):
        eq = n_ps[t - 1] * per_point
        sl = unknowns - eq
        mk = meas.get(f"{dsel},{t}")
        rows.append(
            f"{t:>4} {n_ps[t - 1]:>6} {t * N:>6} {eq:>10} {unknowns:>9} {sl:>+7} "
            + (f"{mk['kernel']:>7} {mk['found']:>6}" if mk else f"{'.':>7} {'.':>6}")
            + ("   <- u_max" if t == U_MAX else ""))
    st.code("\n".join(rows), language="text")
    st.caption(
        "`slack = unknowns - equations`. **Positive slack guarantees a curve; negative "
        "slack does not forbid one**, because the equations need not be independent. "
        "Dots are combinations gs_bound.ipynb has not measured yet. "
        + (f"At d = {dsel} the measured columns come from "
           f"`data/gs_bound_{p}_{m}_{K}.jsonl`." if meas else
           "Run `gs_bound.ipynb` to fill the measured columns."))
