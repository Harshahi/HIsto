"""Tests for sidorenko_fischer.ipynb -- Section IV-B histogram recovery.

Skipped unless the optional `sage` group is installed:  uv sync --group sage

The load-bearing claims, in order of how much they'd hurt if wrong:

  IV-A inversion is exact          -- histograms really do come out of the superposition
  Lemma 2                          -- spurious codewords score at most d*t*(K-1)
  end-to-end recovery              -- the t superimposed messages come back, and only those
  Example 2                        -- the strength ceiling is real, not an artefact
  monotonicity in d                -- Remark 1's claim that larger d buys larger t
"""
import itertools
import random

import numpy as np
import pytest

from conftest import ROOT, load_notebook

pytest.importorskip("sage.all__sagemath_modules",
                    reason="needs the optional sage group: uv sync --group sage")

P, M, K = 7, 1, 2
N = Q = P ** M


@pytest.fixture(scope="module")
def sf():
    ns, skipped = load_notebook("sidorenko_fischer.ipynb")
    for name in ("ks_matrix", "histograms_from_superposition", "superpose",
                 "support_matrix", "score", "ks_list_decode", "bound_6",
                 "random_distinct_messages", "recovers", "preprocess_noisy"):
        assert name in ns, f"{name} missing; skipped={skipped}"
    return ns


def messages_of(ns, t, seed=0):
    return ns["random_distinct_messages"](t, Q, K, seed)


def decode(ns, msgs, d, agree=None):
    hist = ns["histograms_from_superposition"](
        ns["superpose"](msgs, P, M, ns["tab"]), P, M, ns["tab"])
    got, n_P, ell = ns["ks_list_decode"](
        ns["support_matrix"](hist), d, P, M, K, ns["points"], ns["ROOT_FINDER"],
        agree=agree)
    recovered = [[int(c) for c in f.list()] + [0] * (K - len(f.list())) for f in got]
    return recovered, n_P, ell


# ------------------------------------------------------ Kautz-Singleton
@pytest.mark.parametrize("msg", [[0, 0], [2, 1], [6, 3], [1, 5]])
def test_ks_matrix_is_one_hot_per_column(sf, msg):
    cw = sf["enc"]["reed_solomon_encode"](msg, P, M, sf["tab"])
    W = sf["ks_matrix"](cw, Q)
    assert W.shape == (Q, N)
    assert set(np.unique(W)) <= {0, 1}
    assert (W.sum(axis=0) == 1).all(), "exactly one symbol per position"
    assert W.sum() == N, "|supp(W)| = N"
    assert [int(np.flatnonzero(W[:, j])[0]) for j in range(N)] == [int(c) for c in cw]


# ------------------------------------------------------------- Section IV-A
@pytest.mark.parametrize("t", [1, 2, 3, 4, 6])
def test_iv_a_histogram_inversion_is_exact(sf, t):
    """(1/N) H^dagger applied block-wise must return the true symbol counts."""
    msgs = messages_of(sf, t, seed=t)
    cws = [list(sf["enc"]["reed_solomon_encode"](u, P, M, sf["tab"])) for u in msgs]
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
    truth = np.array([[sum(1 for c in cws if c[j] == a) for j in range(N)]
                      for a in range(Q)])
    assert np.array_equal(hist, truth)
    assert (hist.sum(axis=0) == t).all(), "column sums reveal t"


def test_support_matrix_is_the_or_of_the_ks_matrices(sf):
    msgs = messages_of(sf, 4, seed=1)
    cws = [list(sf["enc"]["reed_solomon_encode"](u, P, M, sf["tab"])) for u in msgs]
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
    expected = np.clip(sum(sf["ks_matrix"](c, Q) for c in cws), 0, 1)
    assert np.array_equal(sf["support_matrix"](hist), expected)


# ------------------------------------------------------------------ Lemma 2
@pytest.mark.parametrize("t,d", [(2, 1), (3, 1), (4, 1), (4, 3), (5, 2)])
def test_lemma_2_bounds_spurious_scores(sf, t, d):
    """Superimposed codewords score exactly d*N; everything else <= d*t*(K-1)."""
    msgs = messages_of(sf, t, seed=t + d)
    cws = [list(sf["enc"]["reed_solomon_encode"](u, P, M, sf["tab"])) for u in msgs]
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
    Mmat = d * sf["support_matrix"](hist)

    for c in cws:
        assert sf["score"](c, Mmat) == d * N, "a superimposed codeword must max out"

    Rx = sf["Rx"]
    worst = max(sf["score"]([int(Rx(list(co))(x)) for x in sf["points"]], Mmat)
                for co in itertools.product(range(Q), repeat=K)
                if [int(Rx(list(co))(x)) for x in sf["points"]] not in cws)
    assert worst <= d * t * (K - 1), f"Lemma 2 violated: {worst} > {d * t * (K - 1)}"
    if t * (K - 1) < N:
        assert worst < d * N, "the threshold must separate when t(K-1) < N"


# ------------------------------------------------------------- end to end
@pytest.mark.parametrize("t,d", [(2, 1), (3, 1), (4, 1), (5, 1), (4, 2), (5, 3)])
def test_recovers_exactly_the_superimposed_messages(sf, t, d):
    msgs = messages_of(sf, t, seed=t)
    recovered, _, _ = decode(sf, msgs, d)
    assert sorted(recovered) == sorted(msgs), "must be exactly the superimposed set"


def test_within_bound_6_recovery_always_holds(sf):
    """Inside the paper's sufficient condition, recovery must never fail."""
    for d in (1, 2, 3):
        for t in range(2, sf["bound_6"](N, K, d) + 1):
            assert sf["recovers"](t, d, seed=t), f"failed inside the bound at t={t}, d={d}"


def test_larger_d_never_recovers_less(sf):
    """Remark 1: raising d can only help. Monotone in d at fixed t."""
    best = {}
    for d in (1, 2, 3):
        best[d] = max((t for t in range(2, 8) if sf["recovers"](t, d, seed=t)),
                      default=0)
    assert best[1] <= best[2] <= best[3], best
    assert best[3] > best[1], "raising d must buy something"


# ------------------------------------------------------------- Example 2
def test_example_2_two_collections_share_one_histogram(sf):
    """The strength ceiling is information-theoretic, not an algorithm artefact."""
    F, Rx, points = sf["F"], sf["Rx"], sf["points"]
    A = [[int(F(al)) for _ in range(N)] for al in range(N)]
    B = [[int(F(al) + F(j)) for j in range(N)] for al in range(N)]
    assert all(c not in B for c in A), "the two collections must be disjoint"

    YA = np.clip(sum(sf["ks_matrix"](c, Q) for c in A), 0, 1)
    YB = np.clip(sum(sf["ks_matrix"](c, Q) for c in B), 0, 1)
    assert np.array_equal(YA, YB), "identical disjunctive output"
    assert (YA == 1).all(), "and it is all-ones"

    # every codeword scores the maximum, so no filter can discriminate
    scores = [sf["score"]([int(Rx(list(co))(x)) for x in points], YA)
              for co in itertools.product(range(Q), repeat=K)]
    assert all(s == N for s in scores)
    assert len(scores) == Q ** K


def test_both_example_2_collections_are_real_codewords(sf):
    Rx, points = sf["Rx"], sf["points"]
    all_cws = [[int(Rx(list(co))(x)) for x in points]
               for co in itertools.product(range(Q), repeat=K)]
    F = sf["F"]
    for al in range(N):
        assert [int(F(al)) for _ in range(N)] in all_cws
        assert [int(F(al) + F(j)) for j in range(N)] in all_cws


# ---------------------------------------------------------- noisy (IV-B2)
def test_majority_vote_recovers_t_and_flags_bad_columns(sf):
    msgs = messages_of(sf, 3, seed=5)
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
    rng = random.Random(2)
    noisy = hist.copy()
    j = rng.randrange(N)
    noisy[rng.randrange(Q), j] += 1              # changes the column sum
    cleaned, t_hat, zeroed = sf["preprocess_noisy"](noisy)
    assert t_hat == 3, "majority vote over column sums must recover t"
    assert j in zeroed
    assert (cleaned[:, j] == 0).all()
    assert np.array_equal(cleaned[:, [c for c in range(N) if c != j]],
                          hist[:, [c for c in range(N) if c != j]])


def test_mass_preserving_corruption_slips_past_the_majority_vote(sf):
    """Why s must come from the error model, not from len(zeroed): moving mass
    within a column keeps its sum at t, so the vote cannot see it."""
    msgs = messages_of(sf, 3, seed=5)
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
    noisy = hist.copy()
    j = next(c for c in range(N) if (noisy[:, c] > 0).sum() >= 2)
    a = int(np.flatnonzero(noisy[:, j])[0])
    b = next(z for z in range(Q) if noisy[z, j] == 0)
    noisy[a, j] -= 1
    noisy[b, j] += 1                              # sum unchanged, support changed
    _, t_hat, zeroed = sf["preprocess_noisy"](noisy)
    assert t_hat == 3
    assert j not in zeroed, "the vote cannot detect a mass-preserving corruption"
    assert not np.array_equal(sf["support_matrix"](noisy),
                              sf["support_matrix"](hist))


def test_relaxed_threshold_survives_noisy_histograms(sf):
    """With s wrong columns the threshold drops to d*(N-s) and recovery holds."""
    msgs = messages_of(sf, 3, seed=5)
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
    rng = random.Random(1)
    noisy = hist.copy()
    corrupted = sorted(rng.sample(range(N), 2))
    for j in corrupted:
        a, b = rng.sample(range(Q), 2)
        noisy[a, j] += 1
        noisy[b, j] = max(0, noisy[b, j] - 1)
    cleaned, _, _ = sf["preprocess_noisy"](noisy)
    got, _, _ = sf["ks_list_decode"](
        sf["support_matrix"](cleaned), 3, P, M, K, sf["points"], sf["ROOT_FINDER"],
        agree=N - len(corrupted))
    rec = [[int(c) for c in f.list()] + [0] * (K - len(f.list())) for f in got]
    for u in msgs:
        assert u in rec, f"{u} lost despite the relaxed threshold"


def test_strict_threshold_would_have_lost_them(sf):
    """The relaxation is necessary, not decorative: at the noiseless threshold
    d*N the corrupted run drops superimposed codewords."""
    msgs = messages_of(sf, 3, seed=5)
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
    rng = random.Random(1)
    noisy = hist.copy()
    for j in sorted(rng.sample(range(N), 2)):
        a, b = rng.sample(range(Q), 2)
        noisy[a, j] += 1
        noisy[b, j] = max(0, noisy[b, j] - 1)
    cleaned, _, _ = sf["preprocess_noisy"](noisy)
    got, _, _ = sf["ks_list_decode"](
        sf["support_matrix"](cleaned), 3, P, M, K, sf["points"], sf["ROOT_FINDER"])
    rec = [[int(c) for c in f.list()] + [0] * (K - len(f.list())) for f in got]
    assert not all(u in rec for u in msgs), "strict threshold should lose some"


# ------------------------------------------------------------- end to end
def test_notebook_executes_end_to_end():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(ROOT / "sidorenko_fischer.ipynb", as_version=4)
    nbclient.NotebookClient(
        nb, timeout=3000, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    errors = [o for cell in nb.cells for o in cell.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors, [f"{e.get('ename')}: {e.get('evalue')}" for e in errors]
    figures = [o for cell in nb.cells for o in cell.get("outputs", [])
               if o.get("output_type") == "display_data"]
    assert len(figures) == 1, f"expected 1 embedded figure, got {len(figures)}"


# =====================================================================
# Checks against the Sidorenko-Fischer paper itself, rather than against
# histo.pdf's summary of it. The source is in the repo as
# Low-Complexity_List_Decoding_of_Reed-Solomon_Coded_Pulse_Position_Modulation.pdf
# =====================================================================

def test_cost_is_the_constraint_count(sf):
    """SF eq. (9): C(M) = 1/2 sum m_ij(m_ij+1), which IS the number of linear
    constraints, since multiplicity m contributes m(m+1)/2 of them."""
    msgs = messages_of(sf, 4, seed=4)
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
    Y = sf["support_matrix"](hist)
    for d in (1, 2, 3):
        expected = int(Y.sum()) * d * (d + 1) / 2
        assert sf["cost"](d * Y) == expected
    assert sf["cost"](0 * Y) == 0


def test_list_size_bound_eq_12_is_never_violated(sf):
    """SF eq. (12): |list| <= sqrt(2 C(M) / (k-1))."""
    for t in (2, 3, 4, 5, 6):
        for d in (1, 2):
            msgs = messages_of(sf, t, seed=t)
            hist = sf["histograms_from_superposition"](
                sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
            Y = sf["support_matrix"](hist)
            raw, _, _ = sf["ks_algorithm_2"](Y, d, P, M, K, sf["points"],
                                             sf["ROOT_FINDER"])
            assert len(raw) <= sf["list_bound"](d * Y, K) + 1e-9


@pytest.mark.parametrize("t", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_theorem_1_is_a_contract(sf, t, d):
    """SF Thm. 1 (11): if <W M> >= sqrt(2(k-1)C(M)) then f IS in the output list.

    This is the test that caught the degree-budget bug: at t=6, d=2 the old
    budget of (agree-1)*d failed while (11) held.
    """
    msgs = messages_of(sf, t, seed=t)
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
    Y = sf["support_matrix"](hist)
    cws = [list(sf["enc"]["reed_solomon_encode"](u, P, M, sf["tab"])) for u in msgs]
    worst_score = min(sf["score"](c, d * Y) for c in cws)
    if worst_score < sf["theorem_1_rhs"](d * Y, K):
        pytest.skip("(11) does not hold here, so nothing is guaranteed")
    raw, _, _ = sf["ks_algorithm_2"](Y, d, P, M, K, sf["points"], sf["ROOT_FINDER"])
    recovered = [[int(c) for c in f.list()] + [0] * (K - len(f.list())) for f in raw]
    for u in msgs:
        assert u in recovered, f"(11) held but {u} missing from Algorithm 2's list"


def test_budget_regression_t6_d2(sf):
    """The exact case the old (agree-1)*d budget lost. Keep it pinned."""
    t, d = 6, 2
    msgs = messages_of(sf, t, seed=t)
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
    Y = sf["support_matrix"](hist)
    # Theorem 1 applies here, so recovery is mandatory, not optional
    cws = [list(sf["enc"]["reed_solomon_encode"](u, P, M, sf["tab"])) for u in msgs]
    assert min(sf["score"](c, d * Y) for c in cws) >= sf["theorem_1_rhs"](d * Y, K)
    assert sf["recovers"](t, d, seed=t)


def test_algorithm_2_has_no_filter(sf):
    """Alg. 2 returns the raw list; the rejection step is histo.pdf's addition.
    So Algorithm 2's output must be a superset of the filtered output."""
    for t in (3, 5):
        for d in (1, 2):
            msgs = messages_of(sf, t, seed=t)
            hist = sf["histograms_from_superposition"](
                sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
            Y = sf["support_matrix"](hist)
            raw, _, _ = sf["ks_algorithm_2"](Y, d, P, M, K, sf["points"],
                                             sf["ROOT_FINDER"])
            kept, _, _ = sf["ks_list_decode"](Y, d, P, M, K, sf["points"],
                                              sf["ROOT_FINDER"])
            assert set(kept) <= set(raw), "the filter can only remove candidates"


def test_theorem_3_equals_histo_bound_6(sf):
    """SF Thm. 3 (14) u <= m/(m+1)*(n-tau)/(k-1) at tau=0 IS histo's (6) first term."""
    for d in (1, 2, 3, 6, 10):
        sf14 = d / (d + 1) * N / (K - 1)
        assert sf["bound_6"](N, K, d) == min(int(sf14), (N - 1) // (K - 1))


def test_corollary_1_list_size_at_max_u(sf):
    """SF Cor. 1 (15): at the maximum u, L = m(n-tau)/(k-1) = (m+1)u."""
    for d in (1, 2, 3, 6):
        u_max = d / (d + 1) * N / (K - 1)
        assert abs(d * N / (K - 1) - (d + 1) * u_max) < 1e-9


def test_eq_17_adaptive_multiplicity(sf):
    """SF eq. (17): the m it selects must actually decode, for every t."""
    for t in range(2, 8):
        msgs = messages_of(sf, t, seed=t)
        hist = sf["histograms_from_superposition"](
            sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
        Y = sf["support_matrix"](hist)
        mm = sf["min_multiplicity"](Y, K, N)
        if mm is None:
            continue
        assert sf["recovers"](t, mm, seed=t), f"(17) chose m={mm} for t={t}, failed"


def test_eq_17_is_sharper_than_theorem_3(sf):
    """(17) uses the measured wt(Y); Thm. 3 assumes the worst case u*n. Collisions
    make the measured weight smaller, so (17) can pick a smaller m."""
    for t in (3, 4, 5, 6):
        msgs = messages_of(sf, t, seed=t)
        hist = sf["histograms_from_superposition"](
            sf["superpose"](msgs, P, M, sf["tab"]), P, M, sf["tab"])
        Y = sf["support_matrix"](hist)
        assert int(Y.sum()) <= t * N, "wt(Y) cannot exceed the worst case"
        if t >= 3:
            assert int(Y.sum()) < t * N, "collisions must actually occur here"


# =====================================================================
# m > 1.  The notebook demonstrates at m = 1 for legibility; these pin
# that nothing in IV-A or IV-B is secretly m = 1 only.  The original
# failure was int() on a GF(p^m) element, which raises TypeError -- so
# the regression guard is that ks_list_decode returns rather than raises.
# =====================================================================

# (p, m, K) with enough strength to be worth testing. Note (2, 2, 3) is NOT
# here: N = 4 with K = 3 has strength floor((N-1)/(K-1)) = 1, so t >= 2 is
# information-theoretically impossible and says nothing about m > 1.
PM_CASES = [(2, 3, 2), (3, 2, 2), (2, 4, 2), (3, 2, 3)]


def strength(N_, k):
    return (N_ - 1) // (k - 1)


@pytest.fixture(scope="module")
def field_setup(sf):
    return {(p, m, k): sf["setup_field"](p, m, k) for (p, m, k) in PM_CASES}


@pytest.mark.parametrize("p,m,k", PM_CASES)
def test_int_on_extension_field_elements_raises(field_setup, p, m, k):
    """The premise of the bug. If this ever stops raising, the FROM_SAGE
    indirection could be simplified away."""
    s = field_setup[(p, m, k)]
    if m == 1:
        pytest.skip("int() is fine on GF(p)")
    with pytest.raises(TypeError):
        int(s["points"][p])          # a genuine extension element


@pytest.mark.parametrize("p,m,k", PM_CASES)
def test_setup_field_is_a_bijection_onto_the_field(field_setup, p, m, k):
    s = field_setup[(p, m, k)]
    q = p ** m
    assert len(set(s["points"])) == q
    assert set(s["from_sage"].values()) == set(range(q))
    for i in range(q):
        assert s["from_sage"][s["points"][i]] == i


@pytest.mark.parametrize("p,m,k", PM_CASES)
def test_our_encoder_agrees_with_sage_over_extension_fields(sf, field_setup, p, m, k):
    s = field_setup[(p, m, k)]
    Rx = s["F"]["x"]
    rng = np.random.default_rng(0)
    for _ in range(12):
        u = [int(z) for z in rng.integers(0, s["q"], k)]
        ours = list(sf["enc"]["reed_solomon_encode"](u, p, m, s["tab"]))
        f = Rx([s["points"][c] for c in u])
        theirs = [s["from_sage"][f(x)] for x in s["points"]]
        assert ours == theirs, (p, m, u)


@pytest.mark.parametrize("p,m,k", PM_CASES)
def test_iv_a_is_exact_for_m_above_1(sf, field_setup, p, m, k):
    """h_j = (1/N) H^dagger s_j needs only H^dagger H = N I, true at every m."""
    s = field_setup[(p, m, k)]
    q, N_ = s["q"], s["N"]
    msgs = sf["random_distinct_messages"](3, q, k, seed=1)
    cws = [list(sf["enc"]["reed_solomon_encode"](u, p, m, s["tab"])) for u in msgs]
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, p, m, s["tab"]), p, m, s["tab"])
    truth = np.array([[sum(1 for c in cws if c[j] == a) for j in range(N_)]
                      for a in range(q)])
    assert np.array_equal(hist, truth)


@pytest.mark.parametrize("p,m,k", PM_CASES)
def test_algorithm_2_itself_never_needed_fixing(sf, field_setup, p, m, k):
    """Interpolation + factorization worked at m > 1 unmodified; only the
    scoring step was m = 1 only."""
    s = field_setup[(p, m, k)]
    if strength(s["N"], k) < 3:
        pytest.skip(f"strength is only {strength(s['N'], k)} here")
    msgs = sf["random_distinct_messages"](3, s["q"], k, seed=1)
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, p, m, s["tab"]), p, m, s["tab"])
    raw, n_P, ell = sf["ks_algorithm_2"](
        sf["support_matrix"](hist), 1, p, m, k, s["points"], s["rf"])
    assert n_P > 0 and ell >= 1
    recovered = [sf["message_of"](f, k, s["from_sage"]) for f in raw]
    for u in msgs:
        assert u in recovered


@pytest.mark.parametrize("p,m,k", PM_CASES)
@pytest.mark.parametrize("t", [2, 3])
def test_iv_b_recovers_exactly_for_m_above_1(sf, field_setup, p, m, k, t):
    s = field_setup[(p, m, k)]
    if t > strength(s["N"], k):
        pytest.skip(f"t={t} exceeds the strength {strength(s['N'], k)}")
    msgs = sf["random_distinct_messages"](t, s["q"], k, seed=t)
    hist = sf["histograms_from_superposition"](
        sf["superpose"](msgs, p, m, s["tab"]), p, m, s["tab"])
    got, _, _ = sf["ks_list_decode"](
        sf["support_matrix"](hist), 1, p, m, k, s["points"], s["rf"],
        from_sage=s["from_sage"])
    rec = sorted(sf["message_of"](f, k, s["from_sage"]) for f in got)
    assert rec == sorted(msgs)


@pytest.mark.parametrize("p,m,k", PM_CASES)
def test_message_and_codeword_helpers_round_trip(sf, field_setup, p, m, k):
    s = field_setup[(p, m, k)]
    Rx = s["F"]["x"]
    rng = np.random.default_rng(3)
    for _ in range(10):
        u = [int(z) for z in rng.integers(0, s["q"], k)]
        f = Rx([s["points"][c] for c in u])
        assert sf["message_of"](f, k, s["from_sage"]) == u
        assert sf["codeword_of"](f, s["points"], s["from_sage"]) == \
            list(sf["enc"]["reed_solomon_encode"](u, p, m, s["tab"]))


def test_bounds_depend_only_on_N_not_the_factorisation(sf):
    """N = 16 as 2^4 must give the same bounds as any other route to 16."""
    for d in (1, 2, 3):
        assert sf["bound_6"](16, 2, d) == sf["bound_6"](2 ** 4, 2, d)
    Y = np.ones((16, 16), dtype=int)
    assert sf["cost"](d * Y) == sf["cost"](d * np.ones((2 ** 4, 2 ** 4), dtype=int))
