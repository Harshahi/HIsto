"""Tests for encode_m.ipynb -- the general-m encoder over F_{p^m}.

Beyond re-checking the m = 1 properties at m > 1, two tests here earn their
keep. `test_hadamard_uses_the_bilinear_form_not_field_multiplication` pins the
one substitution that looks right and is wrong, and
`test_m_equals_1_reproduces_encode_notebook_exactly` ties the general
implementation to the simple one that is already trusted.
"""
import numpy as np
import pytest

from conftest import ROOT, load_notebook

PM = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (5, 2), (5, 1), (7, 1)]
PMK = [(2, 3, 2), (2, 4, 2), (3, 2, 2), (2, 4, 3), (3, 2, 3), (5, 1, 2)]


# ------------------------------------------------------------- notebook health
def test_notebook_loads_cleanly():
    ns, skipped = load_notebook("encode_m.ipynb")
    assert not skipped, f"cells failed to load: {skipped}"
    for name in ("GF", "phi", "Code", "hadamard_exponents", "hadamard_matrix",
                 "d_min", "mu", "l_max", "u_max", "min_N", "prime_powers",
                 "next_prime_power", "dim_for", "choose_params"):
        assert name in ns, f"{name} missing"


# ----------------------------------------------------------------- the field
@pytest.mark.parametrize("p,m", PM)
def test_gf_is_a_field(encode_m_ns, p, m):
    F = encode_m_ns["GF"](p, m)
    q = p ** m
    assert F.q == q
    assert (F.ADD[0] == np.arange(q)).all(), "0 is not the additive identity"
    assert (F.MUL[F.one] == np.arange(q)).all(), "1 is not the multiplicative identity"
    assert (F.ADD == F.ADD.T).all() and (F.MUL == F.MUL.T).all(), "not commutative"
    assert (F.MUL[0] == 0).all(), "0 must annihilate"
    for a in range(1, q):
        assert sorted(F.MUL[a].tolist()) == list(range(q)), f"{a} is a zero divisor"
        assert F.MUL[a, F.INV[a]] == F.one
        assert F.ADD[a, F.NEG[a]] == 0


@pytest.mark.parametrize("p,m", PM)
def test_gf_is_associative_and_distributive(encode_m_ns, p, m):
    F = encode_m_ns["GF"](p, m)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b, c = (int(x) for x in rng.integers(0, F.q, 3))
        assert F.MUL[a, F.ADD[b, c]] == F.ADD[F.MUL[a, b], F.MUL[a, c]]
        assert F.MUL[F.MUL[a, b], c] == F.MUL[a, F.MUL[b, c]]
        assert F.ADD[F.ADD[a, b], c] == F.ADD[a, F.ADD[b, c]]


@pytest.mark.parametrize("p,m", PM)
def test_vec_is_an_additive_isomorphism_onto_f_p_to_the_m(encode_m_ns, p, m):
    """The paper's 'represented as a vector in F_p^m' -- addition must commute with vec."""
    F = encode_m_ns["GF"](p, m)
    assert F.vec.shape == (p ** m, m)
    for a in range(F.q):
        assert F.from_vec(F.vec[a]) == a, "vec and from_vec must be inverse"
        for b in range(F.q):
            assert (F.vec[F.ADD[a, b]] == (F.vec[a] + F.vec[b]) % p).all()


@pytest.mark.parametrize("p,m", PM)
def test_modulus_is_monic_irreducible_of_degree_m(encode_m_ns, p, m):
    ns = encode_m_ns
    F = ns["GF"](p, m)
    assert len(F.modulus) - 1 == m
    assert F.modulus[-1] == 1, "must be monic"
    assert ns["is_irreducible"](F.modulus, p)


# --------------------------------------------------------------- the one trap
@pytest.mark.parametrize("p,m", PM)
def test_hadamard_uses_the_bilinear_form_not_field_multiplication(encode_m_ns, p, m):
    """f_Had(a) = (a b^T) is a dot product of coefficient vectors mod p, NOT F.MUL.

    The two coincide at m = 1, which is why substituting F.MUL looks reasonable
    and stays invisible until m > 1.
    """
    ns = encode_m_ns
    F = ns["GF"](p, m)
    L = ns["hadamard_exponents"](F)
    assert L.shape == (F.q, F.q)
    assert (L == (F.vec @ F.vec.T) % p).all()
    assert (L == L.T).all(), "the form is symmetric"
    if m == 1:
        assert (L == F.MUL).all(), "at m = 1 the two must agree"
    else:
        assert not (L == F.MUL).all(), "at m > 1 they must differ"


@pytest.mark.parametrize("p,m", PM)
def test_hadamard_matrix_is_orthogonal(encode_m_ns, p, m):
    ns = encode_m_ns
    F = ns["GF"](p, m)
    H = ns["hadamard_matrix"](F)
    assert np.allclose(H.conj().T @ H, F.q * np.eye(F.q))
    assert np.allclose(np.abs(H), 1.0)


def test_phi_is_real_valued_at_p_equals_2(encode_m_ns):
    """p = 2 puts phi's range at {+1, -1}, the conventional VSA setting."""
    vals = encode_m_ns["phi"](np.arange(2), 2)
    assert np.allclose(vals.imag, 0.0)
    assert np.allclose(sorted(vals.real), [-1.0, 1.0])


# ------------------------------------------------------------------- the code
@pytest.mark.parametrize("p,m,K", PMK)
def test_encode_shape_and_modulus(encode_m_ns, p, m, K):
    c = encode_m_ns["Code"](p, m, K)
    assert c.N == p ** m
    assert c.dim == p ** (2 * m) == c.N ** 2
    assert c.size == p ** (m * K) == c.N ** K
    assert c.d_min == c.N - K + 1
    for i in (0, 1, c.size // 3, c.size - 1):
        hv = c.encode(c.index_to_message(i))
        assert hv.shape == (c.dim,)
        assert np.allclose(np.abs(hv), 1.0)


@pytest.mark.parametrize("p,m,K", PMK)
def test_rs_is_linear_over_the_extension_field(encode_m_ns, p, m, K):
    c = encode_m_ns["Code"](p, m, K)
    rng = np.random.default_rng(0)
    for _ in range(20):
        u, v = c.random_messages(2, rng)
        lhs = c.rs_encode(c.message_add(u, v))
        rhs = c.F.ADD[c.rs_encode(u), c.rs_encode(v)]
        assert np.array_equal(lhs, rhs)


@pytest.mark.parametrize("p,m,K", PMK)
def test_index_to_message_is_a_bijection(encode_m_ns, p, m, K):
    c = encode_m_ns["Code"](p, m, K)
    seen = {tuple(c.index_to_message(i)) for i in range(c.size)}
    assert len(seen) == c.size


@pytest.mark.parametrize("p,m,K", PMK)
def test_binding_is_pointwise_product_under_field_addition(encode_m_ns, p, m, K):
    c = encode_m_ns["Code"](p, m, K)
    rng = np.random.default_rng(1)
    for _ in range(20):
        u, v = c.random_messages(2, rng)
        assert np.allclose(c.encode(u) * c.encode(v), c.encode(c.message_add(u, v)))


@pytest.mark.parametrize("p,m,K", [t for t in PMK if t[1] > 1])
def test_naive_mod_q_addition_breaks_binding_for_m_above_1(encode_m_ns, p, m, K):
    """(a + b) % p**m is not F_{p^m} addition, and gives a silently wrong answer."""
    c = encode_m_ns["Code"](p, m, K)
    rng = np.random.default_rng(2)
    failures = 0
    for _ in range(30):
        u, v = c.random_messages(2, rng)
        naive = [(a + b) % c.q for a, b in zip(u, v)]
        if not np.allclose(c.encode(u) * c.encode(v), c.encode(naive)):
            failures += 1
    assert failures > 0, "expected mod-q addition to be wrong at least sometimes"


@pytest.mark.parametrize("p,m,K", PMK)
def test_proposition_1_incoherence_bound_is_met_exactly(encode_m_ns, p, m, K):
    c = encode_m_ns["Code"](p, m, K)
    if c.size > 512:
        pytest.skip(f"{c.size} codewords is too many to enumerate")
    V = np.array([c.encode(c.index_to_message(i)) for i in range(c.size)])
    S = np.abs(V.conj() @ V.T) / c.dim
    np.fill_diagonal(S, 0.0)
    assert S.max() <= c.mu + 1e-9
    assert abs(S.max() - c.mu) < 1e-6, "Prop 1 should be tight"


def test_code_rejects_invalid_parameters_and_messages(encode_m_ns):
    Code = encode_m_ns["Code"]
    with pytest.raises(ValueError):
        Code(2, 1, 5)                              # N = 2 < K = 5
    c = Code(3, 2, 2)
    with pytest.raises(ValueError):
        c.rs_encode([1])                           # wrong length
    with pytest.raises(ValueError):
        c.rs_encode([0, c.q])                      # symbol out of range
    with pytest.raises(ValueError):
        c.rs_encode([0, -1])


# --------------------------------------------------- N depends only on p^m
@pytest.mark.parametrize("N,K", [(16, 2), (16, 3), (9, 2), (27, 2)])
def test_figures_of_merit_depend_only_on_N(encode_m_ns, N, K):
    ns = encode_m_ns
    assert ns["d_min"](N, K) == N - K + 1
    assert ns["mu"](N, K) == (K - 1) / N
    D = ns["d_min"](N, K)
    assert abs(ns["l_max"](N, K) - (2 * N - D) / (2 * N - 2 * D)) < 1e-12
    assert ns["u_max"](N, K) == (N - 1) // (K - 1)


@pytest.mark.parametrize("p,m,K", PMK)
def test_code_bounds_are_computed_from_N_alone(encode_m_ns, p, m, K):
    """A code's figures of merit must not depend on how N factors into p^m."""
    ns = encode_m_ns
    c = ns["Code"](p, m, K)
    assert c.mu == ns["mu"](c.N, K)
    assert c.d_min == ns["d_min"](c.N, K)
    assert c.l_max == ns["l_max"](c.N, K)
    assert c.u_max == ns["u_max"](c.N, K)


# ---------------------------------------------------- prime-power selection
def test_prime_powers_are_exactly_the_prime_powers(encode_m_ns):
    ns = encode_m_ns
    got = {N for N, _, _ in ns["prime_powers"](40)}
    expected = {2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37}
    assert got == expected
    for N, p, m in ns["prime_powers"](40):
        assert p ** m == N


def test_next_prime_power_is_inclusive_and_beats_the_next_prime(encode_m_ns):
    npp = encode_m_ns["next_prime_power"]
    assert npp(16) == (16, 2, 4), "16 is itself a prime power"
    assert npp(15) == (16, 2, 4), "16 = 2^4 beats the next prime, 17"
    assert npp(10) == (11, 11, 1)
    assert npp(2) == (2, 2, 1)
    assert npp(0) == (2, 2, 1)


@pytest.mark.parametrize("cap,expected_p", [(2, 2), (3, 3), (5, 5)])
def test_max_p_caps_the_alphabet(encode_m_ns, cap, expected_p):
    npp = encode_m_ns["next_prime_power"]
    for n in (5, 12, 30, 100):
        N, p, m = npp(n, max_p=cap)
        assert p <= cap
        assert p ** m == N >= n


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
    # the encoder accepts what was chosen, when it is small enough to build
    if r["N"] <= 32:
        c = ns["Code"](r["p"], r["m"], r["K"])
        assert c.encode(c.index_to_message(0)).shape == (r["dim"],)


def test_allowing_m_above_1_never_costs_dimension(encode_ns, encode_m_ns):
    """Prime powers are a superset of primes, so the m > 1 search can only win.

    Compared against encode.ipynb's primes-only `min_p`, which is the m = 1
    answer to the same question. It must be strictly better somewhere, else
    generalising bought nothing.
    """
    min_p = encode_ns["min_p"]                  # m = 1: N must be prime
    dim_for = encode_m_ns["dim_for"]             # any prime power
    strictly_better = 0
    for K in (2, 3, 4):
        for l in range(1, 21):
            general = dim_for(K, l)
            primes_only = min_p(K, l) ** 2
            assert general <= primes_only, f"K={K}, l={l}: {general} > {primes_only}"
            strictly_better += general < primes_only
    assert strictly_better > 0, "prime powers never helped, which cannot be right"


def test_capping_the_alphabet_never_helps(encode_m_ns):
    """max_p shrinks the menu, so the dimension can only rise or stay put."""
    dim_for = encode_m_ns["dim_for"]
    for K in (2, 3):
        for l in range(1, 21):
            assert dim_for(K, l) <= dim_for(K, l, max_p=3) <= dim_for(K, l, max_p=2)


# --------------------------------------------- cross-notebook agreement
@pytest.mark.parametrize("p,K", [(5, 2), (7, 2), (11, 3), (13, 2), (7, 3)])
def test_m_equals_1_reproduces_encode_notebook_exactly(encode_ns, encode_m_ns, p, K):
    """The general implementation must collapse onto the trusted m = 1 one."""
    Code = encode_m_ns["Code"]
    c = Code(p, 1, K)
    assert np.allclose(encode_m_ns["hadamard_matrix"](c.F),
                       encode_ns["hadamard_matrix"](p))
    rng = np.random.default_rng(3)
    for u in c.random_messages(min(15, c.size), rng):
        assert np.array_equal(c.rs_encode(u), encode_ns["reed_solomon_encode"](u, p))
        assert np.allclose(c.encode(u), encode_ns["histo_encode"](u, p))


def test_l_max_matches_between_the_two_notebooks(encode_ns, encode_m_ns):
    for N in range(2, 40):
        for K in range(2, min(N, 8) + 1):
            assert abs(encode_ns["l_max"](N, K) - encode_m_ns["l_max"](N, K)) < 1e-12


# ------------------------------------------------------------- end to end
def test_notebook_executes_end_to_end():
    """Run every cell, including the figures, in a real kernel."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(ROOT / "encode_m.ipynb", as_version=4)
    nbclient.NotebookClient(
        nb, timeout=900, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    errors = [o for cell in nb.cells for o in cell.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors, [f"{e.get('ename')}: {e.get('evalue')}" for e in errors]
    figures = [o for cell in nb.cells for o in cell.get("outputs", [])
               if o.get("output_type") == "display_data"]
    assert len(figures) == 3, f"expected 3 embedded figures, got {len(figures)}"
