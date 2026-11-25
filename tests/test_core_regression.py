"""Known decodes through streamlit_app/core.py, pinned exactly.

Every other test file in this directory execs a *notebook* -- that is the project's
convention, because the notebooks are the source of truth for the mathematics. But
core.py is not in any notebook, and it is what the Streamlit app, sweep.py,
noise_trials.py and cliff_probe.py all actually run. Until this file it had no
coverage at all.

The reason it needs some is specific. core.unbundle builds an interpolation matrix
over GF(27) and takes its right kernel, then roots ONE element of that kernel --
`ker.basis()[0]`. Which vector that is depends on the matrix backend Sage happens to
have installed: GF(27) falls back to Matrix_generic_dense today, and a different
backend can return a different basis of the same kernel, hence a different Q, hence a
different root set. The kernel is 27-dimensional at d=4 and 156-dimensional at d=5,
so "the first basis vector" is a very arbitrary thing to depend on.

That is not hypothetical -- adding passagemath-linbox changes which classes Sage
picks. These cases exist so that such a change produces a diff that can be read,
rather than a silent shift in results that only shows up as a strange number in a
grid three days later.

`ell` is asserted along with the recovered count on purpose: it is the field most
sensitive to a change of basis, and a decode that starts succeeding at a different
rung is worth knowing about even when the answer is unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "streamlit_app"))

P, M, K = 3, 3, 3


@pytest.fixture(scope="module")
def ctx():
    core = pytest.importorskip("core", reason="needs the sage group")
    return core, core.build_context(P, M, K)


def _run(core, c, d, t, sigma, seed):
    """One decode, exactly as noise_trials.py and cliff_probe.py issue it."""
    msgs = core.message_list(c, t)
    clean = core.bundle_of(c, msgs)
    h0 = core.histograms(c, clean)
    want = sorted(msgs)
    e = core.gaussian_noise(c, sigma, seed=seed) if sigma else 0
    res = core.decode(c, clean + e, d=d, ell_min=2, ell_max=40)
    got = res["messages"]
    return {"n_correct": sum(1 for u in got if u in want),
            "n_returned": len(got),
            "exact": sorted(got) == want,
            "ell": res.get("ell"),
            "stage_a_ok": bool(np.array_equal(res["hist"], h0))}


# (d, t, sigma, seed, n_correct, n_returned, exact, ell, stage_a_ok)
# Recorded 2026-11-25 against passagemath-{modules,categories,pari,symbolics,ntl},
# i.e. with every matrix falling back to Matrix_generic_dense.
FAST = [
    (1, 1, 0.0, 0, 1, 1, True, 2, True),
    (1, 5, 0.0, 0, 5, 5, True, 5, True),
    (1, 13, 0.0, 0, 13, 13, True, 13, True),
    # past u_max at d=1: no interpolant at any rung, ell = -1. Not an error --
    # the bundling cliff, and it holds at sigma = 0 too.
    (1, 14, 0.0, 0, 0, 0, False, -1, True),
    (1, 5, 1.2, 7, 5, 5, True, 5, False),
    # Stage A histogram is NOT exact here, yet recovery still is: the support
    # survived even though the counts did not. Worth pinning both flags.
    (1, 13, 1.2, 7, 13, 13, True, 13, False),
    (1, 13, 1.4, 3, 0, 0, False, -1, False),
    (1, 9, 1.4, 11, 9, 9, True, 9, False),
]

SLOW = [
    (2, 6, 0.0, 0, 6, 6, True, 6, True),
    (2, 12, 0.0, 0, 12, 12, True, 12, True),
    (2, 13, 1.2, 7, 13, 13, True, 13, False),
    (3, 6, 0.0, 0, 6, 6, True, 6, True),
]


def _check(core, c, case):
    d, t, sigma, seed, n_corr, n_ret, exact, ell, sa = case
    got = _run(core, c, d, t, sigma, seed)
    assert got == {"n_correct": n_corr, "n_returned": n_ret, "exact": exact,
                   "ell": ell, "stage_a_ok": sa}, \
        f"core.decode(d={d}, t={t}, sigma={sigma}, seed={seed}) changed"


@pytest.mark.parametrize("case", FAST, ids=lambda c: f"d{c[0]}_t{c[1]}_s{c[2]}")
def test_known_decode_is_unchanged(ctx, case):
    _check(*ctx, case)


@pytest.mark.slow
@pytest.mark.parametrize("case", SLOW, ids=lambda c: f"d{c[0]}_t{c[1]}_s{c[2]}")
def test_known_decode_is_unchanged_higher_d(ctx, case):
    """d >= 2, where a kernel of dimension > 1 makes the basis choice real.
    ~25s each, so these are opt-in: pytest -m slow"""
    _check(*ctx, case)


def test_the_score_filter_certifies_every_survivor(ctx):
    """Whatever the backend returns, a kept candidate must agree with the
    histogram support at all N positions -- score >= d*N is the whole filter.
    This is the property that stops a changed kernel basis from turning into a
    wrong answer; it can cost recall, but it must never admit a stray codeword.
    """
    core, c = ctx
    N = c["N"]
    for d, t in ((1, 9), (1, 13)):
        msgs = core.message_list(c, t)
        Y = core.support_matrix(core.histograms(c, core.bundle_of(c, msgs)))
        found, _, _ = core.unbundle(c, Y, d=d, ell_min=2, ell_max=40)
        for f in found:
            syms = [c["from_sage"][f(x)] for x in c["points"]]
            assert core.score(c, syms, d * Y) >= d * N
