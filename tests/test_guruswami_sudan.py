"""Tests for guruswami_sudan.ipynb -- Sage's Guruswami-Sudan list decoder.

Skipped entirely unless the optional `sage` dependency group is installed:

    uv sync --group sage

The decoder itself is off-the-shelf, so these tests check the two things that are
ours: that Sage's Reed-Solomon code is built to be *identical* to the one
encode_m.ipynb encodes, and that the decoding radii behave exactly as the theory
says (unique up to (d-1)/2, list up to tau, nothing past the Johnson radius).
"""
import math
import random

import numpy as np
import pytest

from conftest import ROOT, load_notebook

sage_modules = pytest.importorskip(
    "sage.all__sagemath_modules",
    reason="needs the optional sage group: uv sync --group sage")

from sage.coding.decoder import DecodingError                      # noqa: E402
from sage.coding.grs_code import (GeneralizedReedSolomonCode,      # noqa: E402
                                  GRSBerlekampWelchDecoder)
from sage.coding.guruswami_sudan.gs_decoder import (                # noqa: E402
    GRSGuruswamiSudanDecoder)
from sage.rings.finite_rings.finite_field_constructor import GF     # noqa: E402

P, M, K = 2, 4, 2
N = P ** M


@pytest.fixture(scope="module")
def setup():
    """The same code encode_m.ipynb encodes, plus the integer <-> field mapping."""
    enc, _ = load_notebook("encode_m.ipynb")
    vec, ADD, MUL = enc["gf_tables"](P, M)
    modulus = enc["find_irreducible"](P, M)
    F = GF(N, "a", modulus=modulus)
    gen = F.gen()

    def to_sage(i):
        return sum(int(c) * gen ** k for k, c in enumerate(vec[i]))

    points = [to_sage(j) for j in range(N)]
    code = GeneralizedReedSolomonCode(points, K)
    return dict(enc=enc, vec=vec, ADD=ADD, MUL=MUL, modulus=modulus, F=F,
                to_sage=to_sage, from_sage={to_sage(i): i for i in range(N)},
                points=points, code=code,
                nonzero=[x for x in F if x != F.zero()])


# ------------------------------------------------- the code matches encode_m
def test_sage_uses_our_modulus(setup):
    assert setup["F"].modulus().list() == setup["modulus"]


def test_int_to_field_map_is_a_bijection(setup):
    assert len(setup["from_sage"]) == N
    for i in range(N):
        assert setup["from_sage"][setup["to_sage"](i)] == i


def test_code_has_the_expected_parameters(setup):
    code = setup["code"]
    assert (code.length(), code.dimension()) == (N, K)
    assert code.minimum_distance() == N - K + 1, "full-length RS is MDS"


def test_our_encoder_agrees_with_sage(setup):
    """The decisive check: this is encode_m.ipynb's code, not a lookalike."""
    code, enc = setup["code"], setup["enc"]
    Rx = code.base_field()["x"]
    rng = np.random.default_rng(0)
    for _ in range(30):
        u = [int(x) for x in rng.integers(0, N, K)]
        ours = enc["reed_solomon_encode"](u, P, M, (setup["vec"], setup["ADD"],
                                                    setup["MUL"]))
        f = Rx([setup["to_sage"](c) for c in u])
        theirs = [setup["from_sage"][f(x)] for x in setup["points"]]
        assert theirs == list(ours), (u, list(ours), theirs)


# --------------------------------------------------------------- the radii
def test_radii_match_the_formulas(setup):
    code = setup["code"]
    n, d = code.length(), code.minimum_distance()
    assert (d - 1) // 2 == 7
    assert math.isclose(n - math.sqrt(n * (n - d)), 12.0, abs_tol=1e-9)


def test_tau_must_stay_below_the_johnson_radius(setup):
    code = setup["code"]
    n, d = code.length(), code.minimum_distance()
    johnson = n - math.sqrt(n * (n - d))
    GRSGuruswamiSudanDecoder(code, tau=int(johnson) - 1,
                             interpolation_alg="LinearAlgebra")
    with pytest.raises(ValueError):
        GRSGuruswamiSudanDecoder(code, tau=int(johnson),
                                 interpolation_alg="LinearAlgebra")


@pytest.mark.parametrize("tau", [8, 9, 10, 11])
def test_gs_beats_unique_decoding_at_every_reachable_tau(setup, tau):
    code = setup["code"]
    assert tau > (code.minimum_distance() - 1) // 2
    D = GRSGuruswamiSudanDecoder(code, tau=tau, interpolation_alg="LinearAlgebra")
    s, ell = D.parameters()
    assert s >= 1 and ell >= s, f"degenerate parameters {(s, ell)}"


# ------------------------------------------------------------- decoding
def corrupt(code, codeword, n_errors, nonzero, rng):
    r = list(codeword)
    for i in rng.sample(range(len(r)), n_errors):
        r[i] += rng.choice(nonzero)
    return code.ambient_space()(r)


@pytest.mark.parametrize("n_errors", [0, 1, 4, 7, 8, 10, 11])
def test_gs_always_returns_the_true_codeword_within_tau(setup, n_errors):
    code, nonzero = setup["code"], setup["nonzero"]
    D = GRSGuruswamiSudanDecoder(code, tau=11, interpolation_alg="LinearAlgebra")
    rng = random.Random(n_errors)
    for _ in range(8):
        sent = code.random_element()
        received = corrupt(code, sent, n_errors, nonzero, rng)
        assert any(c == sent for c in D.decode_to_code(received))


@pytest.mark.parametrize("n_errors", [8, 10, 11])
def test_unique_decoding_cannot_do_what_gs_does(setup, n_errors):
    """Past its radius Berlekamp-Welch either fails or is wrong -- never reliable."""
    code, nonzero = setup["code"], setup["nonzero"]
    BW = GRSBerlekampWelchDecoder(code)
    rng = random.Random(100 + n_errors)
    correct = 0
    for _ in range(8):
        sent = code.random_element()
        received = corrupt(code, sent, n_errors, nonzero, rng)
        try:
            correct += BW.decode_to_code(received) == sent
        except (DecodingError, ZeroDivisionError):
            pass
    assert correct == 0, f"unique decoding got {correct}/8 right at {n_errors} errors"


@pytest.mark.parametrize("n_errors", [0, 4, 7, 11])
def test_list_stays_within_its_bound(setup, n_errors):
    code, nonzero = setup["code"], setup["nonzero"]
    D = GRSGuruswamiSudanDecoder(code, tau=11, interpolation_alg="LinearAlgebra")
    ell = D.parameters()[1]
    rng = random.Random(7)
    for _ in range(8):
        sent = code.random_element()
        received = corrupt(code, sent, n_errors, nonzero, rng)
        assert len(D.decode_to_code(received)) <= ell


def test_lee_osullivan_interpolation_is_the_one_that_segfaults(setup):
    """Documents why interpolation_alg is pinned. If this starts passing, the
    upstream NTL bug is fixed and the pin can be dropped."""
    code, nonzero = setup["code"], setup["nonzero"]
    rng = random.Random(0)
    sent = code.random_element()
    received = corrupt(code, sent, 11, nonzero, rng)
    D = GRSGuruswamiSudanDecoder(code, tau=11, interpolation_alg="LeeOSullivan")
    try:
        D.decode_to_code(received)
    except BaseException as e:                              # noqa: BLE001
        pytest.skip(f"LeeOSullivan still broken upstream: {type(e).__name__}")
    pytest.fail("LeeOSullivan now works -- drop the interpolation_alg pin")


# ------------------------------------------------------------- end to end
def test_notebook_executes_end_to_end():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(ROOT / "guruswami_sudan.ipynb", as_version=4)
    nbclient.NotebookClient(
        nb, timeout=2400, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    errors = [o for cell in nb.cells for o in cell.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors, [f"{e.get('ename')}: {e.get('evalue')}" for e in errors]
    figures = [o for cell in nb.cells for o in cell.get("outputs", [])
               if o.get("output_type") == "display_data"]
    assert len(figures) == 2, f"expected 2 embedded figures, got {len(figures)}"


@pytest.mark.parametrize("tau,seed,n_roots,n_kept", [(8, 0, 1, 1), (10, 8, 2, 1),
                                                     (10, 36, 2, 2)])
def test_walkthrough_cases_are_reproducible(setup, tau, seed, n_roots, n_kept):
    """The notebook's three worked cases depend on exact seeds and on the
    evaluation-point ORDER (ours, not Sage's list(F)). Pin them."""
    from sage.coding.guruswami_sudan.interpolation import gs_interpolation_linalg
    code, F, points = setup["code"], setup["F"], setup["points"]
    Rx = F["x"]
    elements, nonzero = list(F), setup["nonzero"]

    D = GRSGuruswamiSudanDecoder(code, tau=tau, interpolation_alg="LinearAlgebra")
    s, ell = D.parameters()
    rng = random.Random(seed)
    f_true = Rx([rng.choice(elements) for _ in range(K)])
    sent = code.ambient_space()([f_true(x) for x in points])
    received = list(sent)
    for i in rng.sample(range(N), tau):
        received[i] += rng.choice(nonzero)
    received = code.ambient_space()(received)

    Q = gs_interpolation_linalg(list(zip(points, list(received))), tau, (s, ell), K - 1)
    assert all(Q(x, r) == 0 for x, r in zip(points, list(received))), "Q must interpolate"

    roots = D.rootfinding_algorithm()(Q, maxd=K - 1)
    kept = [f for f in roots
            if sum(1 for u, v in zip([f(x) for x in points], received) if u != v) <= tau]
    assert (len(roots), len(kept)) == (n_roots, n_kept)
    assert f_true in kept, "the transmitted polynomial must always survive"
    assert len(D.decode_to_code(received)) == n_kept, "hand pipeline must match library"
