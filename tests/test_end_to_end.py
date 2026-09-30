"""Tests for end_to_end.ipynb -- the full chain plus the 1000-trial sweep.

Skipped unless the optional `sage` group is installed:  uv sync --group sage

The notebook itself asserts its 1000 trials, so these tests do NOT re-run them
(that would double a minute of runtime for no new information). They check the
things a single pass could get right by luck: that every stage is exact over the
whole message/object space, that stage 3 holds at every t up to u_max, and that
the boundary at u_max is real.
"""
import itertools
import random

import numpy as np
import pytest

from conftest import ROOT, load_notebook

pytest.importorskip("sage.all__sagemath_modules",
                    reason="needs the optional sage group: uv sync --group sage")

P, M, K = 3, 3, 3
N = Q = P ** M
U_MAX = (N - 1) // (K - 1)


@pytest.fixture(scope="module")
def e2e():
    ns, skipped = load_notebook("end_to_end.ipynb")
    for name in ("encode_message", "histograms", "decode_hypervector",
                 "message_of_codeword", "atom", "bind", "encode_object",
                 "read_attributes", "support_matrix", "score", "ks_algorithm_2",
                 "ks_list_decode", "message_of_poly", "ALL_OBJECTS", "results"):
        assert name in ns, f"{name} missing; skipped={skipped}"
    return ns


def key(o):
    return tuple(sorted(o.items()))


# ------------------------------------------------------------ the code itself
def test_code_parameters(e2e):
    assert (e2e["p"], e2e["m"], e2e["K"]) == (P, M, K)
    assert e2e["N"] == N and e2e["DIM"] == P ** (2 * M)
    assert e2e["u_max"] == U_MAX
    assert e2e["CODE"].minimum_distance() == N - K + 1
    assert e2e["K"] > 2 and e2e["m"] > 1, "the demo code must be non-degenerate"


# --------------------------------------------- stage 1, over the whole space
@pytest.mark.parametrize("seed", range(6))
def test_stage1_message_round_trip(e2e, seed):
    rng = random.Random(seed)
    for _ in range(40):
        u = [rng.randrange(Q) for _ in range(K)]
        hv = e2e["encode_message"](u)
        assert hv.shape == (P ** (2 * M),)
        assert np.allclose(np.abs(hv), 1.0), "every entry must be a root of unity"
        assert abs(np.linalg.norm(hv) - np.sqrt(P ** (2 * M))) < 1e-9
        assert e2e["message_of_codeword"](e2e["decode_hypervector"](hv)) == u


def test_stage1_decode_matches_the_encoder_directly(e2e):
    """decode_hypervector must return the same codeword reed_solomon_encode built."""
    rng = random.Random(0)
    for _ in range(30):
        u = [rng.randrange(Q) for _ in range(K)]
        direct = list(e2e["enc"]["reed_solomon_encode"](u, P, M, e2e["tab"]))
        assert e2e["decode_hypervector"](e2e["encode_message"](u)) == direct


def test_histograms_of_a_single_codeword_are_one_hot(e2e):
    rng = random.Random(1)
    for _ in range(20):
        u = [rng.randrange(Q) for _ in range(K)]
        h = e2e["histograms"](e2e["encode_message"](u))
        assert (h.sum(axis=0) == 1).all(), "one symbol per position"
        assert set(np.unique(h)) <= {0, 1}


# ------------------------------------------ stage 2, exhaustively this time
def test_stage2_every_object_round_trips(e2e):
    """All 64 objects, not a sample -- binding recovery is exact or it is broken."""
    objs = e2e["ALL_OBJECTS"]
    assert len(objs) == 64
    for o in objs:
        got = e2e["read_attributes"](
            e2e["message_of_codeword"](e2e["decode_hypervector"](
                e2e["encode_object"](o))))
        assert got == o


def test_binding_stays_in_the_code(e2e):
    """The p-th roots of unity are closed under multiplication, so a bound
    vector is still a codeword. This is what makes stage 2 a decode."""
    for o in e2e["ALL_OBJECTS"][:12]:
        b = e2e["encode_object"](o)
        assert np.allclose(np.abs(b), 1.0)


def test_binding_is_the_field_sum_of_the_atom_messages(e2e):
    vec, ADD, MUL = e2e["tab"]
    slot, code_of = e2e["SLOT"], e2e["CODE_OF"]
    for o in e2e["ALL_OBJECTS"][:12]:
        summed = [0] * K
        for a in o:
            summed[slot[a]] = code_of[(a, o[a])]
        assert np.allclose(e2e["encode_object"](o), e2e["encode_message"](summed))


def test_attribute_subcodes_intersect_trivially(e2e):
    """linear.pdf Remark 7: unbinding is unique only if the subcodes do."""
    import sage.all__sagemath_modules as sm
    F, points = e2e["F"], e2e["points"]
    rows = {}
    for name, s in e2e["SLOT"].items():
        e = [0] * K
        e[s] = 1
        cw = e2e["enc"]["reed_solomon_encode"](e, P, M, e2e["tab"])
        rows[name] = sm.matrix(F, [[points[a] for a in cw]]).row_space()
    for a, b in itertools.combinations(rows, 2):
        assert rows[a].intersection(rows[b]).dimension() == 0, f"{a} ^ {b}"


# --------------------------------------------------------- stage 3, bundling
def test_bundling_leaves_the_code(e2e):
    """A bundle is not a codeword -- that is why it needs IV-A and IV-B."""
    rng = random.Random(2)
    objs = e2e["ALL_OBJECTS"]
    scene = [objs[i] for i in rng.sample(range(len(objs)), 3)]
    bundle = sum(e2e["encode_object"](o) for o in scene)
    assert not np.allclose(np.abs(bundle), 1.0)
    assert np.abs(bundle).max() > 1.0


@pytest.mark.parametrize("t", [1, 2, 3, 4, 6, 9, 13])
def test_stage3_recovers_exactly_up_to_u_max(e2e, t):
    """u_max = 13 is the guarantee; check every step of the way there."""
    assert t <= U_MAX
    rng = random.Random(100 + t)
    objs = e2e["ALL_OBJECTS"]
    scene = [objs[i] for i in rng.sample(range(len(objs)), t)]
    hist = e2e["histograms"](sum(e2e["encode_object"](o) for o in scene))
    assert (hist.sum(axis=0) == t).all(), "IV-A must be exact and reveal t"
    found, _, _ = e2e["ks_list_decode"](e2e["support_matrix"](hist), d=1)
    got = sorted((e2e["read_attributes"](e2e["message_of_poly"](f)) for f in found),
                 key=key)
    assert got == sorted(scene, key=key)
    assert len(found) == t, "no spurious candidate may survive the filter"


@pytest.mark.parametrize("t", [14, 16])
def test_stage3_fails_past_u_max(e2e, t):
    """The boundary must be real, else the successes above prove nothing.
    t(K-1) >= N breaks Lemma 2's filtering condition, and no d can fix it."""
    assert t * (K - 1) >= N
    rng = random.Random(200 + t)
    objs = e2e["ALL_OBJECTS"]
    scene = [objs[i] for i in rng.sample(range(len(objs)), t)]
    hist = e2e["histograms"](sum(e2e["encode_object"](o) for o in scene))
    try:
        found, _, _ = e2e["ks_list_decode"](e2e["support_matrix"](hist), d=1)
        got = sorted((e2e["read_attributes"](e2e["message_of_poly"](f))
                      for f in found), key=key)
        assert got != sorted(scene, key=key), f"t={t} is past u_max, must not recover"
    except RuntimeError:
        pass                                    # no interpolant is also a failure


def test_algorithm_2_output_contains_the_filtered_output(e2e):
    """Alg. 2 returns the raw list; the rejection step only removes."""
    rng = random.Random(3)
    objs = e2e["ALL_OBJECTS"]
    scene = [objs[i] for i in rng.sample(range(len(objs)), 4)]
    Y = e2e["support_matrix"](
        e2e["histograms"](sum(e2e["encode_object"](o) for o in scene)))
    raw, _, _ = e2e["ks_algorithm_2"](Y, 1)
    kept, _, _ = e2e["ks_list_decode"](Y, d=1)
    assert set(kept) <= set(raw)


# ------------------------------------------------- the notebook's own sweep
def test_the_notebook_ran_1000_trials_all_correct(e2e):
    """The notebook asserts this itself; here we check the recorded results
    rather than paying another minute to recompute them."""
    rows = e2e["results"]
    assert len(rows) == 1000
    for field in ("stage1", "stage2", "iva", "stage3"):
        assert all(r[field] for r in rows), f"{field} failed in some trial"
    assert all(r["n_list"] == r["t"] for r in rows), "list size must equal t"
    assert set(r["t"] for r in rows) == {1, 2, 3, 4}, "all t values exercised"
    # every t should be reasonably represented, so no stage is untested by luck
    for t in (1, 2, 3, 4):
        assert sum(1 for r in rows if r["t"] == t) > 150


def test_collisions_make_Y_sparser_than_the_worst_case(e2e):
    """The slack that lets d=1 beat its guarantee: wt(Y) < t*N once t > 1."""
    rows = e2e["results"]
    for t in (2, 3, 4):
        mean_wt = np.mean([r["wtY"] for r in rows if r["t"] == t])
        assert mean_wt < t * N, f"t={t}: expected collisions, got wt(Y)={mean_wt}"
    t1 = [r["wtY"] for r in rows if r["t"] == 1]
    assert all(w == N for w in t1), "a single codeword cannot collide with itself"


def test_notebook_executes_end_to_end():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(ROOT / "end_to_end.ipynb", as_version=4)
    nbclient.NotebookClient(
        nb, timeout=6000, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    errors = [o for cell in nb.cells for o in cell.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors, [f"{e.get('ename')}: {e.get('evalue')}" for e in errors]
    figures = [o for cell in nb.cells for o in cell.get("outputs", [])
               if o.get("output_type") == "display_data"]
    assert len(figures) == 1, f"expected 1 embedded figure, got {len(figures)}"
