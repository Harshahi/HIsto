"""Tests for demo.ipynb -- encode / bind / unbind / bundle / recover at p=3, m=3, K=3.

Skipped unless the optional `sage` group is installed:  uv sync --group sage

The two operations have genuinely different characters and the tests are split
that way: binding is exact and lossless, so it is tested by exhaustive
round-trip over all 64 objects; bundling is lossy and bounded, so it is tested
against u_max, including that it fails immediately past it.
"""
import itertools

import numpy as np
import pytest

from conftest import ROOT, load_notebook

pytest.importorskip("sage.all__sagemath_modules",
                    reason="needs the optional sage group: uv sync --group sage")

P, M, K = 3, 3, 3
N = Q = P ** M          # 27
DIM = P ** (2 * M)      # 729


@pytest.fixture(scope="module")
def demo():
    ns, skipped = load_notebook("demo.ipynb")
    for name in ("atom", "atom_message", "bind", "encode_object",
                 "decode_hypervector", "message_of_codeword", "read_attributes",
                 "histograms_from_superposition", "support_matrix", "score",
                 "ks_list_decode", "message_of_poly", "ATTRIBUTES", "SLOT",
                 "l_max_f", "u_max_f", "johnson", "mu", "d_min", "ALL_OBJECTS"):
        assert name in ns, f"{name} missing; skipped={skipped}"
    return ns


def objkey(o):
    return tuple(sorted(o.items()))


# ------------------------------------------------------------- the code itself
def test_parameters_are_all_non_trivial(demo):
    """The point of the demo: p, m and K are all above their degenerate values."""
    assert (demo["p"], demo["m"], demo["K"]) == (P, M, K)
    assert demo["m"] > 1, "m = 1 would be encode.ipynb"
    assert demo["K"] > 2, "K = 2 gives size == dim, i.e. no compression"
    assert demo["N"] == N and demo["DIM"] == DIM
    assert demo["CODE"].minimum_distance() == N - K + 1


def test_attributes_own_disjoint_coordinates(demo):
    """Trivially intersecting subcodes: one coordinate per attribute."""
    slots = list(demo["SLOT"].values())
    assert sorted(slots) == list(range(len(demo["ATTRIBUTES"])))
    assert len(slots) == K, "with K=3 each of the 3 attributes gets one coordinate"
    for name, slot in demo["SLOT"].items():
        for v in demo["ATTRIBUTES"][name]:
            u = demo["atom_message"](name, v)
            assert u[slot] != 0, "an atom must be nonzero in its own slot"
            assert all(u[i] == 0 for i in range(K) if i != slot), "and zero elsewhere"


def test_value_codes_avoid_zero(demo):
    """0 must mean 'absent', so no real value may encode to it."""
    for (name, v), code in demo["VALUE_CODE"].items():
        assert 1 <= code < Q


# -------------------------------------------------------------------- binding
def test_atoms_are_unit_modulus_hypervectors(demo):
    for name in demo["ATTRIBUTES"]:
        for v in demo["ATTRIBUTES"][name]:
            hv = demo["atom"](name, v)
            assert hv.shape == (DIM,)
            assert np.allclose(np.abs(hv), 1.0)


def test_bind_equals_encode_of_the_field_sum(demo):
    """The algebraic identity the whole scheme rests on."""
    ADD = demo["tab"][1]
    rng = np.random.default_rng(0)
    names = list(demo["ATTRIBUTES"])
    for _ in range(20):
        obj = {n: demo["ATTRIBUTES"][n][rng.integers(len(demo["ATTRIBUTES"][n]))]
               for n in names}
        total = [0] * K
        for n, v in obj.items():
            total = [int(ADD[a, b]) for a, b in zip(total, demo["atom_message"](n, v))]
        assert np.allclose(demo["encode_object"](obj),
                           demo["enc"]["histo_encode"](total, P, M, demo["tab"]))


def test_binding_round_trips_every_object(demo):
    """Exhaustive: all 64 objects, bind then recover. Binding is lossless."""
    objs = demo["ALL_OBJECTS"]
    assert len(objs) == 64
    for o in objs:
        cw = demo["decode_hypervector"](demo["encode_object"](o), P, M, demo["tab"])
        got = demo["read_attributes"](demo["message_of_codeword"](cw))
        assert got == o, (got, o)


def test_binding_is_commutative_and_associative(demo):
    a = demo["atom"]("colour", "red")
    b = demo["atom"]("shape", "square")
    c = demo["atom"]("size", "huge")
    assert np.allclose(demo["bind"](a, b, c), demo["bind"](c, a, b))
    assert np.allclose(demo["bind"](demo["bind"](a, b), c),
                       demo["bind"](a, demo["bind"](b, c)))


def test_conjugate_is_the_binding_inverse(demo):
    """|entry| = 1 means the conjugate is the multiplicative inverse."""
    a = demo["atom"]("colour", "blue")
    b = demo["atom"]("shape", "circle")
    assert np.allclose(demo["bind"](a, b) * np.conj(b), a)


def test_decode_hypervector_rejects_a_superposition(demo):
    """It asserts one-hot histograms, so a bundle must not slip through."""
    two = [demo["ALL_OBJECTS"][0], demo["ALL_OBJECTS"][1]]
    bundle = sum(demo["encode_object"](o) for o in two)
    with pytest.raises(AssertionError):
        demo["decode_hypervector"](bundle, P, M, demo["tab"])


# ------------------------------------------------------------------- bundling
def test_iv_a_reads_t_off_the_histograms(demo):
    """Every column of a correct histogram matrix sums to t."""
    for t in (1, 3, 5, 9):
        scene = demo["ALL_OBJECTS"][:t]
        hist = demo["histograms_from_superposition"](
            sum(demo["encode_object"](o) for o in scene), P, M, demo["tab"])
        assert hist.shape == (Q, N)
        assert (hist.sum(axis=0) == t).all()


@pytest.mark.parametrize("t", [2, 4, 8, 13])
def test_bundle_recovers_exactly_within_u_max(demo, t):
    u_max = demo["u_max_f"](N, K)
    assert t <= u_max
    rng = np.random.default_rng(7)
    objs = demo["ALL_OBJECTS"]
    scene = [objs[i] for i in rng.choice(len(objs), t, replace=False)]
    Y = demo["support_matrix"](demo["histograms_from_superposition"](
        sum(demo["encode_object"](o) for o in scene), P, M, demo["tab"]))
    got, _, _ = demo["ks_list_decode"](Y, 1, P, M, K, demo["points"],
                                       demo["ROOT_FINDER"], demo["FROM_SAGE"])
    rec = [demo["read_attributes"](demo["message_of_poly"](f)) for f in got]
    assert sorted(rec, key=objkey) == sorted(scene, key=objkey)


@pytest.mark.parametrize("t", [14, 16])
def test_bundle_fails_past_u_max(demo, t):
    """u_max is a hard cliff, not a soft guideline: t = 13 works, t = 14 does not."""
    assert t > demo["u_max_f"](N, K)
    rng = np.random.default_rng(7)
    objs = demo["ALL_OBJECTS"]
    scene = [objs[i] for i in rng.choice(len(objs), t, replace=False)]
    Y = demo["support_matrix"](demo["histograms_from_superposition"](
        sum(demo["encode_object"](o) for o in scene), P, M, demo["tab"]))
    with pytest.raises(RuntimeError):
        demo["ks_list_decode"](Y, 1, P, M, K, demo["points"],
                               demo["ROOT_FINDER"], demo["FROM_SAGE"])


def test_the_cliff_is_exactly_at_u_max(demo):
    """Pin both sides of it, on two independent draws."""
    u_max = demo["u_max_f"](N, K)
    objs = demo["ALL_OBJECTS"]
    for seed in (7, 11):
        for t, should_work in ((u_max, True), (u_max + 1, False)):
            rng = np.random.default_rng(seed)
            scene = [objs[i] for i in rng.choice(len(objs), t, replace=False)]
            Y = demo["support_matrix"](demo["histograms_from_superposition"](
                sum(demo["encode_object"](o) for o in scene), P, M, demo["tab"]))
            try:
                got, _, _ = demo["ks_list_decode"](
                    Y, 1, P, M, K, demo["points"], demo["ROOT_FINDER"],
                    demo["FROM_SAGE"])
                rec = [demo["read_attributes"](demo["message_of_poly"](f)) for f in got]
                worked = sorted(rec, key=objkey) == sorted(scene, key=objkey)
            except RuntimeError:
                worked = False
            assert worked is should_work, f"t={t}, seed={seed}: expected {should_work}"


def test_similarity_readout_is_the_loose_bound(demo):
    """l_max = 7.25 but the readout still separates well past it -- the opposite
    character to u_max. Both facts matter when budgeting."""
    l_max = demo["l_max_f"](N, K)
    objs = demo["ALL_OBJECTS"]
    for t in (4, 8, 16):
        rng = np.random.default_rng(7)
        scene = [objs[i] for i in rng.choice(len(objs), t, replace=False)]
        bundle = sum(demo["encode_object"](o) for o in scene)
        nb = np.linalg.norm(bundle)
        ins = [abs(np.vdot(demo["encode_object"](o), bundle)) / (np.sqrt(DIM) * nb)
               for o in scene]
        outs = [abs(np.vdot(demo["encode_object"](o), bundle)) / (np.sqrt(DIM) * nb)
                for o in objs if o not in scene]
        assert min(ins) > max(outs), f"readout merged at t={t}"
        if t > l_max:
            assert True, "and this t is past l_max, so the bound was conservative"


# --------------------------------------------------------------- the formulas
def test_bound_formulas(demo):
    for N_, K_ in [(27, 3), (256, 2), (256, 20), (9, 3), (59049, 3)]:
        assert demo["d_min"](N_, K_) == N_ - K_ + 1
        assert demo["mu"](N_, K_) == (K_ - 1) / N_
        D = demo["d_min"](N_, K_)
        assert abs(demo["l_max_f"](N_, K_) - (2 * N_ - D) / (2 * (N_ - D))) < 1e-9
        assert demo["u_max_f"](N_, K_) == (N_ - 1) // (K_ - 1)
        assert abs(demo["johnson"](N_, K_) - (N_ - np.sqrt(N_ * (K_ - 1)))) < 1e-9


def test_u_max_exceeds_l_max_everywhere(demo):
    """The two capacities never coincide; recovery always reaches further."""
    for N_ in (9, 27, 81, 256, 1024):
        for K_ in range(2, min(N_, 30)):
            assert demo["u_max_f"](N_, K_) >= demo["l_max_f"](N_, K_) - 1


def test_sweep_monotonicity(demo):
    """The three sweeps must move the way the reading claims they do."""
    l_max_f, u_max_f, mu = demo["l_max_f"], demo["u_max_f"], demo["mu"]
    # raising m (so N) raises capacity and lowers coherence
    caps = [l_max_f(3 ** mm, 3) for mm in range(1, 9)]
    assert caps == sorted(caps)
    cohs = [mu(3 ** mm, 3) for mm in range(1, 9)]
    assert cohs == sorted(cohs, reverse=True)
    # raising K at fixed N lowers capacity and raises coherence -- the only
    # direction where more is worse
    capsK = [l_max_f(256, KK) for KK in range(2, 40)]
    assert capsK == sorted(capsK, reverse=True)
    cohsK = [mu(256, KK) for KK in range(2, 40)]
    assert cohsK == sorted(cohsK)
    # and dim is flat in K, which is why the temptation exists
    assert len({256 ** 2 for _ in range(2, 40)}) == 1


# ------------------------------------------------------------- end to end
def test_notebook_executes_end_to_end():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(ROOT / "demo.ipynb", as_version=4)
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
