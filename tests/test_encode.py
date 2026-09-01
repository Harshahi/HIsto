"""Tests for encode.ipynb -- the m = 1 encoder over the prime field F_p.

Every claim the notebook makes in prose is checked here against the paper:
phi is an isomorphism, Reed-Solomon is linear and meets the Singleton bound,
the Hadamard matrix is orthogonal, binding is point-wise product, and the code
meets Proposition 1's incoherence bound.
"""
from itertools import product

import numpy as np
import pytest

from conftest import code_cells, load_notebook

PK = [(5, 2), (7, 2), (11, 2), (7, 3), (13, 2)]


# --------------------------------------------------------------- notebook health
def test_notebook_loads_without_skipping_needed_cells():
    ns, skipped = load_notebook("encode.ipynb")
    for name in ("phi", "reed_solomon_encode", "hadamard_matrix", "histo_encode",
                 "next_prime", "l_max", "min_p", "choose_params"):
        assert name in ns, f"{name} missing; skipped cells: {skipped}"


def test_every_code_cell_compiles():
    bad = []
    for cid, src in code_cells("encode.ipynb"):
        try:
            compile(src, f"encode.ipynb[{cid}]", "exec")
        except SyntaxError as e:
            bad.append((cid, e.msg))
    assert not bad, f"cells that do not compile: {bad}"


# --------------------------------------------------------------------------- phi
@pytest.mark.parametrize("p", [2, 3, 5, 7, 11])
def test_phi_is_additive_to_multiplicative_isomorphism(encode_ns, p):
    phi = encode_ns["phi"]
    for a, b in product(range(p), repeat=2):
        assert np.allclose(phi(a, p) * phi(b, p), phi((a + b) % p, p))


@pytest.mark.parametrize("p", [2, 3, 5, 7])
def test_phi_lands_on_the_unit_circle_and_is_injective(encode_ns, p):
    phi = encode_ns["phi"]
    vals = phi(np.arange(p), p)
    assert np.allclose(np.abs(vals), 1.0)
    assert len(np.unique(np.round(vals, 9))) == p


# ---------------------------------------------------------------- Reed-Solomon
@pytest.mark.parametrize("p,K", PK)
def test_rs_is_linear_over_f_p(encode_ns, p, K):
    rs = encode_ns["reed_solomon_encode"]
    rng = np.random.default_rng(0)
    for _ in range(25):
        u = rng.integers(0, p, K)
        v = rng.integers(0, p, K)
        assert np.array_equal(rs((u + v) % p, p), (rs(u, p) + rs(v, p)) % p)


@pytest.mark.parametrize("p,K", PK)
def test_rs_codeword_count_is_p_to_the_K(encode_ns, p, K):
    rs = encode_ns["reed_solomon_encode"]
    words = {tuple(int(x) for x in rs(u, p)) for u in product(range(p), repeat=K)}
    assert len(words) == p ** K


@pytest.mark.parametrize("p,K", PK)
def test_rs_meets_the_singleton_bound(encode_ns, p, K):
    """Distinct codewords agree in at most K-1 places, i.e. d_min = N - K + 1."""
    rs = encode_ns["reed_solomon_encode"]
    words = [np.asarray(rs(u, p)) for u in product(range(p), repeat=K)]
    worst = max(int((a == b).sum())
                for i, a in enumerate(words) for b in words[i + 1:])
    assert worst == K - 1
    assert p - worst == p - K + 1                      # d_min


# -------------------------------------------------------------------- Hadamard
@pytest.mark.parametrize("p", [2, 3, 5, 7, 11])
def test_hadamard_is_orthogonal(encode_ns, p):
    H = encode_ns["hadamard_matrix"](p)
    assert H.shape == (p, p)
    assert np.allclose(H.conj().T @ H, p * np.eye(p))
    assert np.allclose(np.abs(H), 1.0)


# --------------------------------------------------------------------- encoder
@pytest.mark.parametrize("p,K", PK)
def test_hypervectors_are_roots_of_unity_of_the_right_dimension(encode_ns, p, K):
    enc = encode_ns["histo_encode"]
    rng = np.random.default_rng(1)
    for _ in range(10):
        hv = enc([int(x) for x in rng.integers(0, p, K)], p)
        assert hv.shape == (p ** 2,)
        assert np.allclose(np.abs(hv), 1.0)
        assert abs(np.linalg.norm(hv) - p) < 1e-9      # sqrt(p^2)


@pytest.mark.parametrize("p,K", PK)
def test_binding_is_pointwise_product(encode_ns, p, K):
    """encode(u) * encode(v) == encode(u + v mod p) -- the F_p-linearity of C."""
    enc = encode_ns["histo_encode"]
    rng = np.random.default_rng(2)
    for _ in range(20):
        u = [int(x) for x in rng.integers(0, p, K)]
        v = [int(x) for x in rng.integers(0, p, K)]
        w = [(a + b) % p for a, b in zip(u, v)]
        assert np.allclose(enc(u, p) * enc(v, p), enc(w, p))


@pytest.mark.parametrize("p,K", PK)
def test_proposition_1_incoherence_bound_is_met_exactly(encode_ns, p, K):
    """Prop 1: C is (1 - (N-K+1)/N)-incoherent, i.e. mu = (K-1)/p, and it is tight."""
    enc = encode_ns["histo_encode"]
    V = np.array([enc(list(u), p) for u in product(range(p), repeat=K)])
    S = np.abs(V.conj() @ V.T) / (p ** 2)
    np.fill_diagonal(S, 0.0)
    mu = (K - 1) / p
    assert S.max() <= mu + 1e-9
    assert abs(S.max() - mu) < 1e-6, "the bound should be attained, not merely respected"


# ------------------------------------------------------- parameter selection
def test_next_prime_is_inclusive_and_returns_plain_int(encode_ns):
    """sympy.nextprime is strictly greater and returns Integer; both are corrected."""
    np_ = encode_ns["next_prime"]
    assert np_(11) == 11
    assert np_(12) == 13
    assert np_(1) == 2 and np_(0) == 2 and np_(-5) == 2
    assert isinstance(np_(12), int), "a sympy Integer would break ':.4f' formatting"
    assert f"{1 / np_(12):.4f}" == "0.0769"


@pytest.mark.parametrize("N", range(2, 60))
def test_l_max_agrees_with_both_algebraic_forms(encode_ns, N):
    """(2N-D)/(2N-2D) == (1+mu)/(2mu), the two ways the paper's bound is written."""
    l_max = encode_ns["l_max"]
    for K in range(2, min(N, 9) + 1):
        mu = (K - 1) / N
        d_min = N - K + 1
        assert abs(l_max(N, K) - (1 + mu) / (2 * mu)) < 1e-12
        assert abs(l_max(N, K) - (2 * N - d_min) / (2 * N - 2 * d_min)) < 1e-12


def test_l_max_is_infinite_at_K_equals_1(encode_ns):
    assert encode_ns["l_max"](7, 1) == float("inf")


@pytest.mark.parametrize("K", range(1, 9))
@pytest.mark.parametrize("l", [1, 2, 3, 5, 8, 13, 20])
def test_min_p_is_exactly_the_rearrangement_of_l_max(encode_ns, K, l):
    min_p, l_max = encode_ns["min_p"], encode_ns["l_max"]
    p = min_p(K, l)
    assert l_max(p, K) >= l - 1e-9, "min_p must clear l_max"
    assert p >= (K - 1) * (2 * l - 1), "the p >= (K-1)(2l-1) form"
    assert p >= K, "Reed-Solomon needs N >= K"


@pytest.mark.parametrize("l,n_items", [(1, 10), (2, 10), (2, 1000), (4, 100),
                                       (4, 100000), (8, 100), (16, 1000)])
def test_choose_params_satisfies_all_three_constraints(encode_ns, l, n_items):
    r = encode_ns["choose_params"](l, n_items)
    assert r["p"] >= r["K"]
    assert r["l_max"] >= l - 1e-9
    assert r["p"] >= (r["K"] - 1) * (2 * l - 1)
    assert r["n_codewords"] >= n_items
    assert r["dim"] == r["p"] ** 2
    assert r["d_min"] == r["p"] - r["K"] + 1
    # and the encoder actually accepts the chosen parameters
    hv = encode_ns["histo_encode"]([0] * r["K"], r["p"])
    assert hv.shape == (r["dim"],)


def test_choose_params_is_minimal_in_dimension(encode_ns):
    """No smaller *prime* satisfies the constraints.

    Only primes are candidates here: m = 1 forces N = p, so a prime power like
    N = 4 is not available in this notebook. encode_m.ipynb is where that
    changes, and where 4 does in fact beat 5.
    """
    choose, l_max, isprime = (encode_ns["choose_params"], encode_ns["l_max"],
                              encode_ns["isprime"])
    for l, n in [(2, 10), (4, 100), (8, 100), (16, 1000)]:
        r = choose(l, n)
        for p in (x for x in range(2, r["p"]) if isprime(x)):
            for K in range(1, 9):
                feasible = (p >= K and l_max(p, K) >= l - 1e-9 and p ** K >= n)
                assert not feasible, f"({p},{K}) beats the chosen ({r['p']},{r['K']})"
