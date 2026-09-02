"""Tests for encode_m.ipynb -- the general-m encoder over F_{p^m}.

Plain functions, same shape as encode.ipynb: everything takes (p, m) and
returns numpy arrays. Two tests here earn their keep beyond re-checking the
m = 1 properties at m > 1:

  test_hadamard_uses_the_dot_product_not_MUL
      pins the one substitution that looks right and is wrong.
  test_m_equals_1_reproduces_encode_notebook_exactly
      ties the general implementation to the simple one already trusted.
"""
from itertools import product

import numpy as np
import pytest

from conftest import ROOT, load_notebook

PM = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (5, 2), (5, 1), (7, 1)]
PMK = [(2, 3, 2), (2, 4, 2), (3, 2, 2), (2, 2, 3), (5, 1, 2), (7, 1, 3)]


# ------------------------------------------------------------- notebook health
def test_notebook_loads_cleanly():
    ns, skipped = load_notebook("encode_m.ipynb")
    assert not skipped, f"cells failed to load: {skipped}"
    for name in ("phi", "gf_tables", "reed_solomon_encode", "hadamard_exponents",
                 "hadamard_matrix", "histo_encode", "d_min", "l_max", "u_max",
                 "min_N", "prime_powers", "next_prime_power", "dim_for",
                 "choose_params", "find_irreducible", "is_irreducible"):
        assert name in ns, f"{name} missing"


def test_no_classes_are_defined(encode_m_ns):
    """This notebook is deliberately plain functions -- keep it that way.

    Only classes *defined here* count; imports like itertools.product are fine.
    """
    here = encode_m_ns["__name__"]
    classes = [k for k, v in encode_m_ns.items()
               if isinstance(v, type) and getattr(v, "__module__", None) == here]
    assert not classes, f"unexpected classes defined in the notebook: {classes}"


# ----------------------------------------------------------------- the field
@pytest.mark.parametrize("p,m", PM)
def test_gf_tables_is_a_field(encode_m_ns, p, m):
    vec, ADD, MUL = encode_m_ns["gf_tables"](p, m)
    q = p ** m
    assert ADD.shape == MUL.shape == (q, q)
    assert (ADD[0] == np.arange(q)).all(), "0 is not the additive identity"
    assert (MUL[1] == np.arange(q)).all(), "1 is not the multiplicative identity"
    assert (MUL[0] == 0).all(), "0 must annihilate"
    assert (ADD == ADD.T).all() and (MUL == MUL.T).all(), "not commutative"
    for a in range(1, q):
        assert sorted(MUL[a]) == list(range(q)), f"{a} is a zero divisor"
        assert 0 in ADD[a], f"{a} has no additive inverse"


@pytest.mark.parametrize("p,m", PM)
def test_gf_tables_is_associative_and_distributive(encode_m_ns, p, m):
    vec, ADD, MUL = encode_m_ns["gf_tables"](p, m)
    q = p ** m
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b, c = (int(x) for x in rng.integers(0, q, 3))
        assert MUL[a, ADD[b, c]] == ADD[MUL[a, b], MUL[a, c]]
        assert MUL[MUL[a, b], c] == MUL[a, MUL[b, c]]
        assert ADD[ADD[a, b], c] == ADD[a, ADD[b, c]]


@pytest.mark.parametrize("p,m", PM)
def test_vec_is_an_additive_isomorphism_onto_f_p_to_the_m(encode_m_ns, p, m):
    """The paper's 'represented as a vector in F_p^m': addition commutes with vec."""
    vec, ADD, MUL = encode_m_ns["gf_tables"](p, m)
    q = p ** m
    assert vec.shape == (q, m)
    pw = p ** np.arange(m)
    for a in range(q):
        assert vec[a] @ pw == a, "digits must reconstruct the integer"
        for b in range(q):
            assert (vec[ADD[a, b]] == (vec[a] + vec[b]) % p).all()


@pytest.mark.parametrize("p", [2, 3, 5, 7, 11])
def test_m_equals_1_collapses_to_plain_mod_p_arithmetic(encode_m_ns, p):
    """This is why encode.ipynb never needed tables."""
    vec, ADD, MUL = encode_m_ns["gf_tables"](p, 1)
    idx = np.arange(p)
    assert (vec.ravel() == idx).all()
    assert (ADD == (idx[:, None] + idx) % p).all()
    assert (MUL == (idx[:, None] * idx) % p).all()


@pytest.mark.parametrize("p,m", PM)
def test_modulus_is_monic_irreducible_of_degree_m(encode_m_ns, p, m):
    ns = encode_m_ns
    f = ns["find_irreducible"](p, m)
    assert len(f) - 1 == m
    assert f[-1] == 1, "must be monic"
    assert ns["is_irreducible"](f, p)


def test_reducible_polynomials_are_rejected(encode_m_ns):
    ns = encode_m_ns
    # x^2 + 1 factors over F_2 as (x+1)^2; x^2 is obviously reducible
    assert not ns["is_irreducible"]([1, 0, 1], 2)
    assert not ns["is_irreducible"]([0, 0, 1], 2)
    assert ns["is_irreducible"]([1, 1, 1], 2), "x^2 + x + 1 is irreducible over F_2"


# --------------------------------------------------------------- the one trap
@pytest.mark.parametrize("p,m", PM)
def test_hadamard_uses_the_dot_product_not_MUL(encode_m_ns, p, m):
    """f_Had(a) = (a b^T) is a dot product of digit vectors mod p, NOT MUL.

    The two coincide at m = 1, which is why substituting MUL looks reasonable
    and stays invisible until m > 1.
    """
    ns = encode_m_ns
    vec, ADD, MUL = ns["gf_tables"](p, m)
    L = ns["hadamard_exponents"](p, m)
    assert (L == (vec @ vec.T) % p).all()
    assert (L == L.T).all(), "the form is symmetric"
    if m == 1:
        assert (L == MUL).all(), "at m = 1 the two must agree"
    else:
        assert not (L == MUL).all(), "at m > 1 they must differ"


@pytest.mark.parametrize("p", [2, 3, 5, 7, 11])
def test_at_m_equals_1_exponents_are_encode_notebooks_outer_product(encode_m_ns, p):
    idx = np.arange(p)
    assert (encode_m_ns["hadamard_exponents"](p, 1) == np.outer(idx, idx) % p).all()


@pytest.mark.parametrize("p,m", PM)
def test_hadamard_matrix_is_orthogonal(encode_m_ns, p, m):
    H = encode_m_ns["hadamard_matrix"](p, m)
    N = p ** m
    assert H.shape == (N, N)
    assert np.allclose(H.conj().T @ H, N * np.eye(N))
    assert np.allclose(np.abs(H), 1.0)


def test_phi_is_real_valued_at_p_equals_2(encode_m_ns):
    """p = 2 puts phi's range at {+1, -1}, the conventional VSA setting."""
    vals = encode_m_ns["phi"](np.arange(2), 2)
    assert np.allclose(vals.imag, 0.0)
    assert np.allclose(sorted(vals.real), [-1.0, 1.0])


# ------------------------------------------------------------- the encoder
@pytest.mark.parametrize("p,m,K", PMK)
def test_hypervector_shape_and_modulus(encode_m_ns, p, m, K):
    ns = encode_m_ns
    tab = ns["gf_tables"](p, m)
    q, dim = p ** m, p ** (2 * m)
    for i in (0, 1, q ** K // 3, q ** K - 1):
        u = [(i // q ** k) % q for k in range(K)]
        hv = ns["histo_encode"](u, p, m, tab)
        assert hv.shape == (dim,)
        assert np.allclose(np.abs(hv), 1.0)
        assert abs(np.linalg.norm(hv) - np.sqrt(dim)) < 1e-9


@pytest.mark.parametrize("p,m,K", PMK)
def test_rs_is_linear_over_the_extension_field(encode_m_ns, p, m, K):
    ns = encode_m_ns
    vec, ADD, MUL = tab = ns["gf_tables"](p, m)
    rs, q = ns["reed_solomon_encode"], p ** m
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = [int(x) for x in rng.integers(0, q, K)]
        v = [int(x) for x in rng.integers(0, q, K)]
        w = [int(ADD[a, b]) for a, b in zip(u, v)]
        assert (rs(w, p, m, tab) == ADD[rs(u, p, m, tab), rs(v, p, m, tab)]).all()


@pytest.mark.parametrize("p,m,K", PMK)
def test_rs_meets_the_singleton_bound(encode_m_ns, p, m, K):
    ns = encode_m_ns
    tab = ns["gf_tables"](p, m)
    q = p ** m
    if q ** K > 4096:
        pytest.skip(f"{q ** K} codewords is too many to enumerate")
    words = [np.asarray(ns["reed_solomon_encode"](list(u), p, m, tab))
             for u in product(range(q), repeat=K)]
    assert len({tuple(w) for w in words}) == q ** K
    worst = max(int((a == b).sum())
                for i, a in enumerate(words) for b in words[i + 1:])
    assert worst == K - 1
    assert ns["d_min"](q, K) == q - worst


@pytest.mark.parametrize("p,m,K", PMK)
def test_binding_is_pointwise_product_under_field_addition(encode_m_ns, p, m, K):
    ns = encode_m_ns
    vec, ADD, MUL = tab = ns["gf_tables"](p, m)
    enc, q = ns["histo_encode"], p ** m
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = [int(x) for x in rng.integers(0, q, K)]
        v = [int(x) for x in rng.integers(0, q, K)]
        w = [int(ADD[a, b]) for a, b in zip(u, v)]
        assert np.allclose(enc(u, p, m, tab) * enc(v, p, m, tab), enc(w, p, m, tab))


@pytest.mark.parametrize("p,m,K", [t for t in PMK if t[1] > 1])
def test_naive_mod_q_addition_breaks_binding_for_m_above_1(encode_m_ns, p, m, K):
    """(a + b) % p**m is not F_{p^m} addition, and is wrong silently."""
    ns = encode_m_ns
    tab = ns["gf_tables"](p, m)
    enc, q = ns["histo_encode"], p ** m
    rng = np.random.default_rng(2)
    failures = 0
    for _ in range(30):
        u = [int(x) for x in rng.integers(0, q, K)]
        v = [int(x) for x in rng.integers(0, q, K)]
        naive = [(a + b) % q for a, b in zip(u, v)]
        if not np.allclose(enc(u, p, m, tab) * enc(v, p, m, tab), enc(naive, p, m, tab)):
            failures += 1
    assert failures > 0, "expected mod-q addition to be wrong at least sometimes"


@pytest.mark.parametrize("p,m,K", PMK)
def test_proposition_1_incoherence_bound_is_met_exactly(encode_m_ns, p, m, K):
    ns = encode_m_ns
    tab = ns["gf_tables"](p, m)
    q, dim = p ** m, p ** (2 * m)
    if q ** K > 512:
        pytest.skip(f"{q ** K} codewords is too many to enumerate")
    V = np.array([ns["histo_encode"](list(u), p, m, tab)
                  for u in product(range(q), repeat=K)])
    S = np.abs(V.conj() @ V.T) / dim
    np.fill_diagonal(S, 0.0)
    mu = (K - 1) / q
    assert S.max() <= mu + 1e-9
    assert abs(S.max() - mu) < 1e-6, "Prop 1 should be tight"


def test_out_of_range_symbols_fail_loudly(encode_m_ns):
    """Like encode.ipynb, these functions skip validation to stay readable.

    A symbol outside [0, p^m) therefore fails on the table lookup rather than
    with a tidy message -- but it does fail, rather than returning nonsense.
    """
    ns = encode_m_ns
    tab = ns["gf_tables"](2, 3)
    assert ns["reed_solomon_encode"]([7, 1], 2, 3, tab).shape == (8,)   # 7 is valid
    with pytest.raises(IndexError):
        ns["reed_solomon_encode"]([8, 1], 2, 3, tab)                    # 8 == p^m is not


# ------------------------------------------------- figures of merit from N
@pytest.mark.parametrize("N,K", [(16, 2), (16, 3), (9, 2), (27, 2), (17, 2)])
def test_figures_of_merit_depend_only_on_N(encode_m_ns, N, K):
    ns = encode_m_ns
    assert ns["d_min"](N, K) == N - K + 1
    D = ns["d_min"](N, K)
    mu = (K - 1) / N
    assert abs(ns["l_max"](N, K) - (2 * N - D) / (2 * N - 2 * D)) < 1e-12
    assert abs(ns["l_max"](N, K) - (1 + mu) / (2 * mu)) < 1e-12
    assert ns["u_max"](N, K) == (N - 1) // (K - 1)


def test_l_max_is_infinite_at_K_equals_1(encode_m_ns):
    assert encode_m_ns["l_max"](16, 1) == float("inf")
    assert encode_m_ns["u_max"](16, 1) == float("inf")


@pytest.mark.parametrize("K", range(1, 9))
@pytest.mark.parametrize("l", [1, 2, 3, 5, 8, 13, 20])
def test_min_N_is_exactly_the_rearrangement_of_l_max(encode_m_ns, K, l):
    ns = encode_m_ns
    N = ns["min_N"](K, l)
    assert ns["l_max"](N, K) >= l - 1e-9
    assert N >= (K - 1) * (2 * l - 1)
    assert N >= K


# ---------------------------------------------------- prime-power selection
def test_prime_powers_are_exactly_the_prime_powers(encode_m_ns):
    ns = encode_m_ns
    got = {N for N, _, _ in ns["prime_powers"](40)}
    assert got == {2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37}
    for N, p, m in ns["prime_powers"](40):
        assert p ** m == N


def test_next_prime_power_is_inclusive_and_beats_the_next_prime(encode_m_ns):
    npp = encode_m_ns["next_prime_power"]
    assert npp(16) == (16, 2, 4), "16 is itself a prime power"
    assert npp(15) == (16, 2, 4), "16 = 2^4 beats the next prime, 17"
    assert npp(10) == (11, 11, 1)
    assert npp(2) == (2, 2, 1)
    assert npp(0) == (2, 2, 1)


@pytest.mark.parametrize("cap", [2, 3, 5])
def test_max_p_caps_the_alphabet(encode_m_ns, cap):
    npp = encode_m_ns["next_prime_power"]
    for n in (5, 12, 30, 100):
        N, p, m = npp(n, max_p=cap)
        assert p <= cap and p ** m == N >= n


def test_max_p_2_always_rounds_up_to_a_power_of_two(encode_m_ns):
    npp = encode_m_ns["next_prime_power"]
    for n in range(2, 70):
        N, p, m = npp(n, max_p=2)
        assert p == 2 and 2 ** m == N and N >= n
        assert N // 2 < n, f"{N} is not the smallest power of two >= {n}"


@pytest.mark.parametrize("l,n_items,cap", [(2, 10, None), (4, 100, None), (4, 100, 2),
                                           (8, 100, None), (20, 1000, None),
                                           (20, 1000, 2), (4, 100000, 2)])
def test_choose_params_satisfies_all_three_constraints(encode_m_ns, l, n_items, cap):
    ns = encode_m_ns
    r = ns["choose_params"](l, n_items, max_p=cap)
    assert r["N"] == r["p"] ** r["m"]
    assert r["N"] >= r["K"], "Reed-Solomon needs N >= K"
    assert r["l_max"] >= l - 1e-9
    assert r["N"] >= (r["K"] - 1) * (2 * l - 1)
    assert r["size"] >= n_items
    assert r["dim"] == r["N"] ** 2
    if cap is not None:
        assert r["p"] <= cap
    if r["N"] <= 27:                      # actually build it when that is cheap
        tab = ns["gf_tables"](r["p"], r["m"])
        hv = ns["histo_encode"]([0] * r["K"], r["p"], r["m"], tab)
        assert hv.shape == (r["dim"],)


def test_allowing_m_above_1_never_costs_dimension(encode_ns, encode_m_ns):
    """Prime powers are a superset of primes, so the m > 1 search can only win.

    Compared against encode.ipynb's primes-only min_p, the m = 1 answer to the
    same question. It must be strictly better somewhere, else generalising
    bought nothing.
    """
    min_p = encode_ns["min_p"]                   # m = 1: N must be prime
    dim_for = encode_m_ns["dim_for"]              # any prime power
    strictly_better = 0
    for K in (2, 3, 4):
        for l in range(1, 21):
            general, primes_only = dim_for(K, l), min_p(K, l) ** 2
            assert general <= primes_only, f"K={K}, l={l}: {general} > {primes_only}"
            strictly_better += general < primes_only
    assert strictly_better > 0, "prime powers never helped, which cannot be right"


def test_capping_the_alphabet_never_helps(encode_m_ns):
    dim_for = encode_m_ns["dim_for"]
    for K in (2, 3):
        for l in range(1, 21):
            assert dim_for(K, l) <= dim_for(K, l, max_p=3) <= dim_for(K, l, max_p=2)


# --------------------------------------------- cross-notebook agreement
@pytest.mark.parametrize("p,K", [(5, 2), (7, 2), (11, 3), (13, 2), (7, 3)])
def test_m_equals_1_reproduces_encode_notebook_exactly(encode_ns, encode_m_ns, p, K):
    """The general implementation must collapse onto the trusted m = 1 one."""
    ns = encode_m_ns
    tab = ns["gf_tables"](p, 1)
    assert np.allclose(ns["hadamard_matrix"](p, 1), encode_ns["hadamard_matrix"](p))
    rng = np.random.default_rng(3)
    for _ in range(15):
        u = [int(x) for x in rng.integers(0, p, K)]
        assert (ns["reed_solomon_encode"](u, p, 1, tab) ==
                encode_ns["reed_solomon_encode"](u, p)).all()
        assert np.allclose(ns["histo_encode"](u, p, 1, tab),
                           encode_ns["histo_encode"](u, p))


def test_l_max_matches_between_the_two_notebooks(encode_ns, encode_m_ns):
    for N in range(2, 40):
        for K in range(2, min(N, 8) + 1):
            assert abs(encode_ns["l_max"](N, K) - encode_m_ns["l_max"](N, K)) < 1e-12


def test_phi_is_identical_between_the_two_notebooks(encode_ns, encode_m_ns):
    for p in (2, 3, 5, 7, 11):
        assert np.allclose(encode_ns["phi"](np.arange(p), p),
                           encode_m_ns["phi"](np.arange(p), p))


# ------------------------------------------------------------- end to end
def test_notebook_executes_end_to_end():
    """Run every cell, including the figures, in a real kernel."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(ROOT / "encode_m.ipynb", as_version=4)
    nbclient.NotebookClient(
        nb, timeout=1200, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    errors = [o for cell in nb.cells for o in cell.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors, [f"{e.get('ename')}: {e.get('evalue')}" for e in errors]
    figures = [o for cell in nb.cells for o in cell.get("outputs", [])
               if o.get("output_type") == "display_data"]
    assert len(figures) == 3, f"expected 3 embedded figures, got {len(figures)}"
