"""histogram_recovery.ipynb, as plain functions the app can call with any (p, m, K).

The encoder is not re-implemented here. It is read straight out of encode_m.ipynb at
import time, the same trick tests/conftest.py uses, so the app cannot drift from the
notebook it mirrors. Everything downstream of that -- bundling, the two decode stages,
the sweep, the multiplicity peeling -- is a direct transcription of the notebook's cells
with the hard-coded p, m, K lifted into arguments.
"""

import ast
import contextlib
import os
import io
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

PRIMES = [2, 3, 5, 7, 11, 13]


def load_notebook_defs(path):
    """exec another notebook's code cells, skipping figures and magics."""

    def draws(src):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return False
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ("show", "savefig") for n in ast.walk(tree))

    nb = json.loads(Path(path).read_text())
    ns = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if draws(src):
            continue
        src = "\n".join(("# " + ln) if ln.lstrip()[:1] in ("%", "!") else ln
                        for ln in src.split("\n"))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, str(path), "exec"), ns)
        except Exception:
            pass
    return ns


def encoder():
    """The encode_m.ipynb namespace. Call once and hold on to it -- about a second."""
    return load_notebook_defs(REPO / "encode_m.ipynb")


# --------------------------------------------------------------------------------
# the code itself
# --------------------------------------------------------------------------------

def numpy_context(p, m, K, enc=None):
    """The Sage-free half: everything encoding, bundling and Stage A need.

    Kept separate because Sage cannot be touched from Streamlit's worker thread (see
    worker.py), while all of this is plain numpy and runs happily in-process.
    """
    enc = enc or encoder()
    N = p ** m
    tab = enc["gf_tables"](p, m)
    return {"enc": enc, "p": p, "m": m, "K": K, "q": N, "N": N, "dim": p ** (2 * m),
            "tab": tab, "H": enc["hadamard_matrix"](p, m, tab),
            "L": enc["hadamard_exponents"](p, m, tab),
            "u_max": (N - 1) // (K - 1)}


def build_context(p, m, K, enc=None):
    """Everything that depends only on (p, m, K): the field, the evaluation points,
    the Hadamard matrix, and Sage's root finder. Only ever called inside worker.py."""
    import sage.all__sagemath_modules                    # noqa: F401 -- before .coding
    from sage.coding.grs_code import GeneralizedReedSolomonCode
    from sage.coding.guruswami_sudan.gs_decoder import GRSGuruswamiSudanDecoder
    from sage.rings.finite_rings.finite_field_constructor import GF

    enc = enc or encoder()
    q = N = p ** m
    tab = enc["gf_tables"](p, m)
    vec = tab[0]

    F = GF(q, "a", modulus=enc["find_irreducible"](p, m))
    gen = F.gen()
    points = [sum(int(c) * gen ** k for k, c in enumerate(vec[i])) for i in range(N)]
    code = GeneralizedReedSolomonCode(points, K)

    return {
        "enc": enc, "p": p, "m": m, "K": K, "q": q, "N": N, "dim": p ** (2 * m),
        "tab": tab, "F": F, "points": points,
        "from_sage": {points[i]: i for i in range(N)},
        "code": code,
        "root_finder": GRSGuruswamiSudanDecoder(
            code, tau=1, interpolation_alg="LinearAlgebra").rootfinding_algorithm(),
        "H": enc["hadamard_matrix"](p, m, tab),
        "L": enc["hadamard_exponents"](p, m, tab),
        "u_max": (N - 1) // (K - 1),
    }


def bounds(ctx):
    """The several numbers that all get called 'capacity', which are not
    interchangeable -- plus the vocabulary size, which is not a capacity at all."""
    N, K = ctx["N"], ctx["K"]
    d = N - K + 1
    return {
        "d_min": d,
        "unique_radius": (d - 1) // 2,
        "johnson": N - np.sqrt(N * (K - 1)),
        "mu": (K - 1) / N,
        "l_max": (2 * N - d) / (2 * (N - d)) if N != d else float("inf"),
        "u_max": (N - 1) // (K - 1),
        "size": N ** K,
    }


# --------------------------------------------------------------------------------
# encode and bundle
# --------------------------------------------------------------------------------

def nth_message(i, q, K):
    """The i-th message in a fixed order. Every slot varies, so these are generic
    codewords rather than (say) a family of constants. No randomness anywhere.

    Note this family has period q, so it yields at most q distinct messages however
    large K is -- message_list tops up from the full space when it runs dry.
    """
    return [((k + 1) * i + k // 2 + 1) % q for k in range(K)]


def message_list(ctx, count):
    """The first `count` distinct messages, in a fixed order.

    The affine family above is only q long, so asking for more than q messages used
    to spin forever: `while len(out) < count` with a generator that had stopped
    producing anything new. Both loops here are bounded by the size of the message
    space, and running out raises rather than hangs.
    """
    q, K = ctx["q"], ctx["K"]
    space = q ** K
    if count > space:
        raise ValueError(f"asked for {count} distinct messages but only {space} = "
                         f"{q}^{K} exist for p^m = {q}, K = {K}")

    seen, out = set(), []

    def take(u):
        if tuple(u) not in seen:
            seen.add(tuple(u))
            out.append(u)
        return len(out) == count

    for i in range(space):                  # the generic family, while it lasts
        if take(nth_message(i, q, K)):
            return out
    for i in range(space):                  # then the rest of the space, base q
        if take([(i // q ** k) % q for k in range(K)]):
            return out
    raise AssertionError("unreachable: the space is exactly `space` messages")


def rs_encode(ctx, u):
    """The Reed-Solomon codeword, as plain ints -- numpy scalars print as
    `np.int64(1)` and the notebook's output shows `1`."""
    return [int(c) for c in
            ctx["enc"]["reed_solomon_encode"](u, ctx["p"], ctx["m"], ctx["tab"])]


def encode(ctx, u):
    return ctx["enc"]["histo_encode"](u, ctx["p"], ctx["m"], ctx["tab"])


def ks_matrix(ctx, codeword):
    """Kautz-Singleton: one-hot each RS symbol into a column. q x N, binary.

    The same codeword the Hadamard layer turns into phases, written the way
    Sidorenko-Fischer's Algorithm 2 wants it. This is the representation where every
    nonzero entry is just 1: the Hadamard grid stores an exponent in 0..p-1, while W
    records only *that* a symbol occurred at a position. Y is the entrywise OR of the
    t matrices W(c_1) .. W(c_t), and M = dY is what the decoder consumes.
    """
    W = np.zeros((ctx["q"], len(codeword)), dtype=int)
    for j, a in enumerate(codeword):
        W[int(a), j] = 1
    return W


def bundle(hvs):
    """Bundling is addition, entry by entry. The result is no longer a codeword.

    Pass a generator, not a list. At the larger (p, m) the app now reaches, holding
    every hypervector at once is gigabytes -- p = 5, m = 4 gives dim = 390,625, so
    300 of them is 2 GB. Summing a generator stays at O(dim).
    """
    return sum(hvs)


def bundle_of(ctx, messages):
    """The bundle of these messages, encoding them one at a time."""
    return bundle(encode(ctx, u) for u in messages)


# --------------------------------------------------------------------------------
# decode -- stage A, then stage B
# --------------------------------------------------------------------------------

def noise_for(ctx, alpha, kind="isotropic", seed=0):
    """Noise e with ||e_j||_2 = alpha * sqrt(N)/2 in every block j of N entries.

    Corollary 2 guarantees round((1/N)Re(H' s_j)) = h_j whenever ||e_j||_2 < sqrt(N)/2,
    so alpha is the fraction of the permitted level: alpha < 1 is inside the paper's
    guarantee, alpha >= 1 is outside it. How much slack that leaves depends entirely on
    the direction of the noise, which is what the three kinds are for.

    real       e_j = H a' with a real and concentrated in a single signed coordinate.
               Lemma 1 needs a in (-0.5, 0.5)^N, and here ||a||_inf = ||a||_2 =
               alpha/2, so the histogram fails the moment alpha reaches 1. The bound
               is tight in this direction.
    isotropic  generic complex Gaussian. Half the energy lands in the imaginary part
               and the real half spreads over N coordinates, so ||a||_inf stays far
               below ||a||_2 and the histogram survives well past alpha = 1.
    imaginary  e_j = i H b' with b real. Lemma 1's Voronoi cell is H h' +
               H(-0.5,0.5)^N + i H R^N -- unbounded in the imaginary direction -- so
               this never breaks, at any alpha. It is the paper's remark that decoding
               "succeeds even for some unbounded error vectors".
    """
    N, H = ctx["N"], ctx["H"]
    rng = np.random.default_rng(seed)
    target = alpha * np.sqrt(N) / 2
    blocks = np.empty((ctx["N"], N), dtype=complex)

    for j in range(N):
        if kind == "real":
            a = np.zeros(N)
            # a signed bump in one coordinate: ||a||_inf = ||a||_2 = alpha/2, so this
            # is exactly the direction Lemma 1 cares about. The sign matters -- a
            # positive bump only invents counts, which the score filter shrugs off,
            # while a negative one can delete a symbol that was really there.
            a[rng.integers(N)] = (alpha / 2) * rng.choice([-1.0, 1.0])
            e = H @ a
        elif kind == "imaginary":
            b = rng.standard_normal(N)
            e = 1j * (H @ (b * (alpha / 2) / np.linalg.norm(b)))
        else:
            z = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            e = z * (target / np.linalg.norm(z))
        blocks[j] = e

    return blocks.reshape(-1)                       # block j is entries jN .. jN+N-1


def gaussian_noise(ctx, sigma, seed=0):
    """AWGN: every entry of the hypervector gets an independent CN(0, sigma^2).

    This is the physically meaningful model. A system that recovers a bundled
    hypervector off some channel perturbs each entry independently and roughly
    normally; it does not hand you a perturbation of some chosen norm.

    The difference from noise_for's alpha matters: alpha pins ||e_j||_2 to exactly
    the same value in every block, which makes it a clean probe of Corollary 2 but
    not a channel. Here the block norms are random, so some blocks sit above the
    Corollary 2 threshold and some below, and the sharp cliff smears into a curve.

    sigma is in units of the signal: a codeword entry has |entry| = 1, so sigma = 1
    is noise as large as the thing it is corrupting.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(ctx["dim"]) + 1j * rng.standard_normal(ctx["dim"])
    return z * (sigma / np.sqrt(2))


def spiking_noise(ctx, sigma=0.0, mu=0, nu=0.0, width=1, seed=0):
    """Gaussian background plus a few very large coordinates.

    The paper's guarantee (Proposition 2) is not stated in terms of a noise
    DISTRIBUTION at all -- it is stated per block. Write e = (e_1, ..., e_N) with
    e_i in C^N, one block per coordinate of the Reed-Solomon codeword. Then
    recovery is guaranteed when

        ||e_i||_2 < sqrt(N)/2   for all but at most s blocks,
        and those s blocks have e_i in E_r,          with (r+1)s < N.

    That is exactly a spiking model: s is how MANY coordinates are badly hit, r is
    how BADLY, and the product is what has to stay under N. So the natural
    parameters here are

        sigma  background CN(0, sigma^2), the ordinary noise floor
        mu     number of spiked entries              (drives s)
        nu     spike magnitude, in units of a codeword entry (drives r)
        width  how many adjacent entries one spike touches -- 1 is a single
               coordinate, larger smears it over neighbours within the block

    `width` exists because a spike confined to one entry and a spike spread over
    several are very different for Corollary 2: the block norm adds in quadrature,
    so one entry of size nu costs the same as `width` entries of size nu/sqrt(width).

    Returns the noise vector, same shape and units as gaussian_noise.
    """
    N, dim = ctx["N"], ctx["dim"]
    rng = np.random.default_rng(seed)
    e = np.zeros(dim, dtype=complex)
    if sigma:
        e += (rng.standard_normal(dim) + 1j * rng.standard_normal(dim)) \
            * (sigma / np.sqrt(2))
    if mu and nu:
        # spike positions are drawn without replacement so mu really is the number
        # of distinct hit entries, not a binomial draw that sometimes collides
        where = rng.choice(dim, size=min(int(mu), dim), replace=False)
        for pos in where:
            blk, off = divmod(int(pos), N)
            phase = np.exp(2j * np.pi * rng.random())
            # spread within the block; wrapping keeps the energy inside one block,
            # which is what Proposition 2 counts
            for k in range(int(width)):
                e[blk * N + (off + k) % N] += nu / np.sqrt(width) * phase
    return e


def noise_report(ctx, e, clean=None):
    """Proposition 2's bookkeeping, measured rather than assumed.

    The subtlety worth being careful about: the paper's r is NOT "how far the
    histogram moved". It is defined through

        E_r = union over v in Z^N with ||v||_1 <= r  of  B_2(Hv, sqrt(N)/2),

    an open BALL around each lattice point. Lemma 1 says the Voronoi cell of Hv is
    Hv + H(-1/2,1/2)^N + i*H*R^N, which strictly CONTAINS that ball. So a block can
    decode to the offset histogram v -- it is in the Voronoi cell -- while sitting
    outside B(Hv, sqrt(N)/2) and therefore outside E_r for every r. Such a block is
    simply not covered by Proposition 2, however small its histogram error looks.

    Reported per block:
        norm_i      ||e_i||_2
        v_i         the lattice offset Stage A will actually round to
        r_i         ||v_i||_1, the histogram's ell_1 error
        in_ball_i   whether ||e_i - H v_i||_2 < sqrt(N)/2, i.e. e_i really is in E_r

    and in aggregate s (exceptional blocks), r (worst r_i among them), whether
    Proposition 2's hypotheses actually hold, and (r+1)s < N.
    """
    N, H = ctx["N"], ctx["H"]
    blocks = np.asarray(e).reshape(N, N)
    norms = np.linalg.norm(blocks, axis=1)
    thresh = np.sqrt(N) / 2
    # the offset Stage A rounds to, block by block -- same formula as histograms()
    V = np.rint((H.conj().T @ blocks.T).real / N).astype(int).T      # (N blocks, N)
    resid = np.linalg.norm(blocks - (H @ V.T).T, axis=1)
    r_i = np.abs(V).sum(axis=1)
    in_ball = resid < thresh

    exc = norms >= thresh                      # blocks Corollary 2 does not cover
    s = int(exc.sum())
    r = int(r_i[exc].max()) if s else 0
    out = {"block_norms": norms, "threshold": float(thresh),
           "v": V, "r_i": r_i, "in_ball": in_ball, "resid": resid,
           "s": s, "r": r,
           # Proposition 2 needs every exceptional block to lie in E_r, not merely
           # to decode to a nearby histogram
           "prop2_applies": bool(s == 0 or (in_ball[exc].all() and (r + 1) * s < N)),
           "rs_ok": (r + 1) * s < N,
           "outside_E": int((exc & ~in_ball).sum())}
    if clean is not None:
        clean = np.asarray(clean)
        h0 = histograms(ctx, clean)
        h1 = histograms(ctx, clean + np.asarray(e))
        dev = np.abs(h1 - h0).sum(axis=0)
        out["r_true"] = int(dev.max()) if len(dev) else 0
        out["s_true"] = int((dev > 0).sum())
        out["deviations"] = dev
    return out


def theorem_1_bound(ctx, d, s, r, e_g=0):
    """histo.pdf Theorem 1 / eq. (11): the largest u the guarantee covers.

        u <= min(  d/(d+1) * (N-s)^2 / ((N-e_g)(K-1)),
                   (N - s - r(s - e_g) - 1) / (K-1)  )

    s is the number of corrupted histograms, r the worst ell_1 deviation among
    them, e_g how many of those were erased in pre-processing. Both terms floor.

    Note the second term falls LINEARLY in r*s -- so a single badly-damaged
    coordinate costs as much capacity as many lightly-damaged ones. That is the
    quantitative form of "a few coordinates become really large".
    """
    N, K = ctx["N"], ctx["K"]
    first = d / (d + 1) * (N - s) ** 2 / ((N - e_g) * (K - 1))
    second = (N - s - r * (s - e_g) - 1) / (K - 1)
    return int(min(np.floor(first), np.floor(second)))


def stage_a_probability(N, sigma):
    """Closed form for P(Stage A returns every histogram exactly) under AWGN.

    H has orthogonal columns of squared norm N, so (H' e)_i is CN(0, N sigma^2)
    and a = Re(H' e)/N has i.i.d. N(0, sigma^2 / 2N) coordinates. Lemma 1 needs
    every coordinate of a inside (-0.5, 0.5), and there are N blocks x N
    coordinates = dim of them, so

        P = erf( sqrt(N) / (2 sigma) ) ^ (N^2)

    Verified against Monte Carlo in gaussian_noise.ipynb.
    """
    from math import erf
    return erf(np.sqrt(N) / (2.0 * sigma)) ** (N * N)


def sigma_half(N, lo=1e-3, hi=1e3, iters=200):
    """The sigma at which Stage A succeeds half the time -- the usable threshold.

    Grows like sqrt(N / log N): bigger fields tolerate MORE noise, because the
    Hadamard transform averages each histogram coordinate over N entries and that
    processing gain outruns the extra coordinates that have to stay in range.
    """
    for _ in range(iters):
        mid = (lo + hi) / 2
        if stage_a_probability(N, mid) > 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def noisy_bundle(ctx, messages, alpha=0.0, kind="isotropic", seed=0):
    """The bundle of these messages plus noise at alpha times the permitted level."""
    s = bundle_of(ctx, messages)
    if alpha:
        s = s + noise_for(ctx, alpha, kind=kind, seed=seed)
    return s


def block_norms(ctx, e):
    """||e_j||_2 for each block -- the quantity Corollary 2 bounds."""
    return np.linalg.norm(np.asarray(e).reshape(ctx["N"], ctx["N"]), axis=1)


def histograms(ctx, vec_in):
    """Stage A (SS IV-A). h_j = round((1/N) Re(H^dagger s_j)), block by block -- the
    paper's Corollary 1. Entry (a, j) = how many bundled codewords carry symbol a at
    position j."""
    N, H = ctx["N"], ctx["H"]
    blocks = np.asarray(vec_in).reshape(N, N).T
    return np.rint((H.conj().T @ blocks).real / N).astype(int)


def support_matrix(hist):
    """Y: which symbols appeared, counts discarded."""
    return (hist > 0).astype(int)


def score(ctx, codeword_syms, M):
    """<W(c), M>: sum M over the one-hot cells of c."""
    return sum(int(M[int(a), j]) for j, a in enumerate(codeword_syms))


# --------------------------------------------------------- prime-field kernel
#
# The interpolation kernel is ~76% of a decode, and Sage computes it over GF(27)
# with Matrix_generic_dense: every entry a boxed Sage object, every arithmetic
# operation dispatched through Python. There is no fast class for GF(27) -- not in
# passagemath, not in upstream SageMath -- because odd-characteristic extension
# fields have none. GF(3), however, gets Matrix_modn_dense_float (LinBox/FFLAS,
# BLAS-backed) as soon as passagemath-linbox is installed.
#
# So we go through the prime field. F_27 is a 3-dimensional F_3-vector space, so an
# m x n matrix A over F_27 becomes a 3m x 3n matrix B over F_3 by replacing each
# entry with its 3x3 multiplication matrix. Because A is F_27-linear its kernel is
# an F_27-subspace, and therefore ker_F3(B) is exactly ker(A) viewed as an F_3-space
# -- same set of vectors, dimension 3x. That is 27x the arithmetic at a tiny
# fraction of the per-operation cost.
#
# Measured on a 1200x900 kernel: 20.8s over GF(27), 0.32s over GF(3).

_PF_CACHE = {}


def _prime_field_tables(ctx):
    """(TABLE, to_idx, basis) for the GF(27) -> GF(3) expansion, built once.

    TABLE[k] is the 3x3 matrix over F_3 of "multiply by the k-th field element",
    in the basis (1, a, a^2); column j is the coordinate vector of e * a^j.
    """
    key = id(ctx["F"])
    if key in _PF_CACHE:
        return _PF_CACHE[key]
    F = ctx["F"]
    deg = F.degree()
    basis = [F.gen() ** i for i in range(deg)]
    elts = ctx["points"]                       # the whole field, in a fixed order
    table = np.array([[[int(c) for c in list(e * b)] for b in basis] for e in elts],
                     dtype=np.int8).transpose(0, 2, 1)
    to_idx = ctx["from_sage"]
    _PF_CACHE[key] = (table, to_idx, basis)
    return _PF_CACHE[key]


def kernel_vector_via_prime_field(ctx, M):
    """One nonzero vector of M's right kernel, computed over F_3 instead of F_27.

    Returns a list of field elements (the same thing Sage's Ker.basis()[0] gives,
    though not necessarily the same vector -- the F_3 kernel has its own basis, and
    any nonzero kernel element is an equally valid interpolation polynomial).
    Returns None when the kernel is trivial.
    """
    from sage.matrix.constructor import matrix
    from sage.rings.finite_rings.finite_field_constructor import GF

    table, to_idx, basis = _prime_field_tables(ctx)
    deg = len(basis)
    m, n = M.nrows(), M.ncols()

    # M.list() is one C-level call; indexing entry by entry in Python is far slower
    codes = np.fromiter((to_idx[e] for e in M.list()),
                        dtype=np.int32, count=m * n).reshape(m, n)
    blocks = table[codes]                                   # (m, n, deg, deg)
    flat = blocks.transpose(0, 2, 1, 3).reshape(deg * m, deg * n)

    # .tolist() on the flattened array would materialise one Python int per
    # entry -- 735M of them, about 20 GB, at d=7. Handing Sage the 2-D numpy
    # array instead keeps the data in the int8 buffer and converts row by row.
    B = matrix(GF(ctx["p"]), flat)
    ker = B.right_kernel()
    if ker.dimension() == 0:
        return None
    w = ker.basis()[0]
    return [sum((int(w[deg * j + i]) * basis[i] for i in range(deg)),
                ctx["F"].zero()) for j in range(n)]


# Below this many entries in the GF(27) matrix, going through the prime field
# costs more than it saves: the expansion is 9x the entries and Sage converts them
# one at a time, which is O(n^2), while the GF(27) kernel it replaces is O(n^3).
# Measured end to end at t=12, sigma=0, both backends on identical input:
#
#     d=1 t=5    56k entries   0.6x   (gf3 slower)
#     d=2 t=12  444k entries   1.5x
#     d=3 t=12  1.5M entries   1.5x
#     d=4 t=12  4.8M entries   2.0x
#
# so the crossover sits a little under 444k. Both backends returned identical
# codewords and identical ell on every case.
PRIME_FIELD_MIN_ENTRIES = 200_000


def interpolate_prime_field(ctx, points, tau, parameters, wy, min_entries=0):
    """gs_interpolation_linalg, with the kernel taken over the prime field.

    Same matrix, same monomials, same contract: a bivariate Q vanishing to order s
    at every point, or None if no such Q exists at these parameters.

    The matrix is built once and then handed to whichever kernel routine suits its
    size, so "auto" costs nothing extra over either backend -- the expensive part
    (building M) is shared.
    """
    from sage.coding.guruswami_sudan.interpolation import _interpolation_matrix_problem

    M, monomials = _interpolation_matrix_problem(points, tau, parameters, wy)
    if M.nrows() * M.ncols() >= min_entries:
        sol = kernel_vector_via_prime_field(ctx, M)
    else:
        ker = M.right_kernel()                 # Sage's own path, on small inputs
        sol = ker.basis()[0] if ker.dimension() else None
    if sol is None:
        return None
    x, y = M.base_ring()["x", "y"].gens()
    return sum(x ** a * y ** b * sol[i] for i, (a, b) in enumerate(monomials))


# Which kernel backend unbundle uses. "gf27" is Sage's own gs_interpolation_linalg
# -- the original path, kept as the reference implementation. "gf3" routes the
# kernel through the prime field (see interpolate_prime_field above). Set
# HISTO_KERNEL=gf3 to switch, so the two can be A/B'd on identical inputs without
# editing code.
# "auto" picks per matrix by size (see PRIME_FIELD_MIN_ENTRIES); "gf3" forces the
# prime field; "gf27" is Sage's own gs_interpolation_linalg, kept as the reference.
KERNEL_BACKEND = os.environ.get("HISTO_KERNEL", "auto")


def corrupted_columns(ctx, hist):
    """How many histogram columns look wrong -- the paper's s, estimated.

    Section IV-B2 pre-processes by a majority vote over the column sums of H':
    every clean column sums to t, so a column whose sum differs has been corrupted.
    This needs no knowledge of t, which is the point of doing it this way.

    Returns (s, t_est, which): the count, the inferred t, and the column indices.
    """
    sums = np.asarray(hist).sum(axis=0)
    vals, counts = np.unique(sums, return_counts=True)
    t_est = int(vals[int(np.argmax(counts))])
    bad = np.flatnonzero(sums != t_est)
    return len(bad), t_est, bad


def _scored_roots(ctx, Q, d, Y):
    """Every root of Q paired with its score against d*Y.

    Split out because the interpolation is the whole cost of a decode and the
    threshold is free: scoring once lets a caller compare the strict rule (d*N)
    against the paper's noisy rule (d*(N-s)) without decoding twice.
    """
    K, points = ctx["K"], ctx["points"]
    out = []
    for f in ctx["root_finder"](Q, maxd=K - 1):
        syms = [ctx["from_sage"][f(x)] for x in points]
        out.append((f, score(ctx, syms, d * Y)))
    return out


def unbundle_candidates(ctx, Y, d=1, ell_max=40, ell_min=2, backend=None):
    """[(message, score)] for every root, before any threshold is applied.

    Same interpolation as unbundle, same ladder; it just stops short of filtering.
    A caller can then apply as many thresholds as it likes to one decode.
    """
    from sage.coding.guruswami_sudan.interpolation import gs_interpolation_linalg

    N, K, points = ctx["N"], ctx["K"], ctx["points"]
    P = [(points[j], points[a]) for j in range(N) for a in range(N) if Y[a, j]]
    tau = len(P) - N
    ell_sat = d * N // (K - 1)
    ladder = list(range(max(2, ell_min), ell_max))
    if ell_sat >= max(2, ell_min) and ell_sat not in ladder:
        ladder.append(ell_sat)
    backend = backend or KERNEL_BACKEND
    for ell in ladder:
        try:
            if backend in ("gf3", "auto"):
                Q = interpolate_prime_field(
                    ctx, P, tau, (d, ell), K - 1,
                    min_entries=0 if backend == "gf3" else PRIME_FIELD_MIN_ENTRIES)
                if Q is None:
                    continue
            else:
                Q = gs_interpolation_linalg(P, tau, (d, ell), K - 1)
        except (IndexError, ValueError):
            continue
        return ([(message_of_poly(ctx, f), sc)
                 for f, sc in _scored_roots(ctx, Q, d, Y)], len(P), ell)
    raise RuntimeError(f"no interpolation polynomial up to ell = {max(ladder)}")


def unbundle(ctx, Y, d=1, ell_max=40, ell_min=2, backend=None, agree=None):
    """Stage B (SS IV-B). Interpolate through every observed (position, symbol) pair
    with multiplicity d, factor out y - f(x), keep those scoring the maximum d*N.

    Returns the DISTINCT codewords only, never their multiplicities -- Y records
    *whether* a symbol appeared, not how often, so a bundle of a,a,b gives exactly the
    same Y as a,b. Use unbundle_with_multiplicities to get the multiset back.
    """
    from sage.coding.guruswami_sudan.interpolation import gs_interpolation_linalg

    N, K, points = ctx["N"], ctx["K"], ctx["points"]
    P = [(points[j], points[a]) for j in range(N) for a in range(N) if Y[a, j]]
    tau = len(P) - N                      # Sage's bound is exclusive -> deg < N*d
    # The ladder walks ell upward and returns on the first ell that admits a curve,
    # which below u_max is ell = t -- measured at d = 1 and d = 2, where the first
    # admitting ell and the kernel dimension are identical. Starting at 2 therefore
    # throws away ~t failed interpolations per decode, which dominates a large trial
    # sweep; callers that know t can pass a bracket instead.
    #
    # Past u_max that stops holding. At d = 4, t = 14 the first admitting ell is 54,
    # far outside any bracket a caller would guess and past the old ell_max of 40, so
    # the decoder could not reach its own solutions. The monomial count rises with ell
    # and then SATURATES at ell_sat = d*N/(K-1) -- beyond it the budget caps j anyway
    # -- so if any ell admits a curve, ell_sat does. Append it as a last resort: one
    # extra interpolation, and only ever on the path that would otherwise give up.
    ell_sat = d * N // (K - 1)
    ladder = list(range(max(2, ell_min), ell_max))
    if ell_sat >= max(2, ell_min) and ell_sat not in ladder:
        ladder.append(ell_sat)
    backend = backend or KERNEL_BACKEND
    for ell in ladder:
        try:
            if backend in ("gf3", "auto"):
                Q = interpolate_prime_field(
                    ctx, P, tau, (d, ell), K - 1,
                    min_entries=0 if backend == "gf3" else PRIME_FIELD_MIN_ENTRIES)
                if Q is None:                 # empty kernel at this ell
                    continue
            else:
                # the reference path: Sage takes the kernel over GF(27) itself
                Q = gs_interpolation_linalg(P, tau, (d, ell), K - 1)
        except (IndexError, ValueError):
            continue
        # `agree` is the score threshold in positions. Default N is the noiseless
        # rule: a genuine codeword puts its symbol in every one of the N
        # histograms, so it scores exactly d*N. Under noise that is harsh -- one
        # symbol lost in Stage A costs d, the candidate misses the bar by one, and
        # a CORRECT codeword is discarded. That is the partial recovery the
        # Gaussian grid shows. histo.pdf IV-B2 rejects only below d*(N - s), so
        # pass agree = N - s for the paper's noisy rule.
        need = d * (N if agree is None else int(agree))
        kept = [f for f, sc in _scored_roots(ctx, Q, d, Y) if sc >= need]
        return kept, len(P), ell
    raise RuntimeError(f"no interpolation polynomial up to ell = {max(ladder)}")


def message_of_poly(ctx, f):
    zero = f.base_ring().zero()
    K = ctx["K"]
    coeffs = list(f.list()) + [zero] * (K - len(f.list()))
    return [ctx["from_sage"][c] for c in coeffs]


def decode(ctx, vec_in, d=1, ell_max=40, ell_min=2, backend=None, agree=None):
    """The whole of Part 3 on one bundle. Plain data out, so it is safe to cache.

    Also returns the polynomials as strings, because the notebook prints `f(x) = ...`
    beside each recovered message and the app shows the same lines.
    """
    hist = histograms(ctx, vec_in)
    Y = support_matrix(hist)
    t0 = time.time()
    try:
        found, n_P, ell = unbundle(ctx, Y, d=d, ell_max=ell_max, ell_min=ell_min,
                                   backend=backend, agree=agree)
        pairs = sorted((message_of_poly(ctx, f), str(f)) for f in found)
        msgs = [u for u, _ in pairs]
        polys = [s for _, s in pairs]
        err = None
    except RuntimeError as e:
        msgs, polys, n_P, ell, err = [], [], int(Y.sum()), -1, str(e)
    return {"hist": hist, "Y": Y, "messages": msgs, "polys": polys, "n_P": n_P,
            "ell": ell, "ms": (time.time() - t0) * 1000, "error": err}


def unbundle_with_multiplicities(ctx, hist, d=1, ell_max=40):
    """SS IV-B3: recover the multiset, not just the set.

    Each round decodes the distinct codewords still present, subtracts their
    histograms, and repeats on what is left. For a, a, b: round 1 returns {a, b},
    subtracting both leaves one copy of a's histogram, and round 2 returns {a}.
    """
    residual = hist.copy()
    multiset, rounds = [], []
    while residual.any():
        found, _, _ = unbundle(ctx, support_matrix(residual), d=d, ell_max=ell_max)
        msgs = [message_of_poly(ctx, f) for f in found]
        if not msgs:
            rounds.append((int(residual.sum(axis=0)[0]), []))
            break                          # stuck: past the bound, nothing decodes
        rounds.append((int(residual.sum(axis=0)[0]), sorted(msgs)))
        for u in msgs:
            for j, sym in enumerate(rs_encode(ctx, u)):
                residual[int(sym), j] -= 1     # subtract this codeword's histogram
            multiset.append(u)
        if (residual < 0).any():
            break                          # a count went negative: not a clean peel
    return multiset, rounds


# --------------------------------------------------------------------------------
# the sweep behind Part 4
# --------------------------------------------------------------------------------

def sweep(ctx, t_limit, d=1, ell_max=40, progress=None):
    """Bundle t = 1 .. t_limit, each containing the one before, and decode each.
    Returns plain dicts so the caller can cache the whole list."""
    msgs = message_list(ctx, t_limit)
    acc = np.zeros(ctx["dim"], dtype=complex)
    rows = []
    for t in range(1, t_limit + 1):
        acc = acc + encode(ctx, msgs[t - 1])        # one at a time; see bundle()
        r = decode(ctx, acc, d=d, ell_max=ell_max)
        rows.append({
            "t": t,
            "n_P": r["n_P"],
            "tN": t * ctx["N"],
            "ell": r["ell"],
            "n_rec": len(r["messages"]),
            "exact": r["messages"] == sorted(msgs[:t]),
            "ms": r["ms"],
            "norm": float(np.linalg.norm(acc)),
            "abs_min": float(np.abs(acc).min()),
            "abs_max": float(np.abs(acc).max()),
        })
        if progress:
            progress(t, t_limit)
    return rows, msgs


def noise_sweep(ctx, messages, alphas, kind="isotropic", seed=0, d=1, ell_max=40,
                progress=None):
    """Decode the same bundle at a ladder of noise levels, and record where it breaks.

    alpha is the multiple of the permitted level sqrt(N)/2 from Corollary 2, so the
    interesting column is whether `exact` survives past alpha = 1 -- and by how much,
    which depends entirely on the direction of the noise.
    """
    N = ctx["N"]
    clean_s = bundle_of(ctx, messages)
    clean = histograms(ctx, clean_s)
    want = sorted(messages)
    rows = []
    for i, a in enumerate(alphas):
        e = noise_for(ctx, a, kind=kind, seed=seed) if a else np.zeros(ctx["dim"])
        res = decode(ctx, clean_s + e, d=d, ell_max=ell_max)
        bad_cols = sum(1 for j in range(N)
                       if not np.array_equal(res["hist"][:, j], clean[:, j]))
        rows.append({
            "alpha": float(a),
            "block_norm": float(block_norms(ctx, e).max()),
            "stage_a_ok": bool(np.array_equal(res["hist"], clean)),
            "bad_cols": int(bad_cols),
            "bad_cells": int((res["hist"] != clean).sum()),
            "n_rec": len(res["messages"]),
            "exact": res["messages"] == want,
            "ms": res["ms"],
        })
        if progress:
            progress(i + 1, len(alphas))
    return rows


def noise_grid(ctx, t_values, alphas, kinds, d=1, ell_max=40, progress=None):
    """Decode across both failure axes at once: how many codewords were bundled, and
    how much noise was added.

    The two limits are independent and come from different halves of the paper --
    u_max = (N-1)/(K-1) is SS IV-B's bundling cliff, Corollary 2's alpha = 1 is
    SS IV-A's noise cliff -- so the grid shows them as a horizontal and a vertical
    boundary on the same picture.
    """
    rows = []
    total = len(kinds) * len(t_values) * len(alphas)
    done = 0
    for kind in kinds:
        for t in t_values:
            msgs = message_list(ctx, t)
            clean_s = bundle_of(ctx, msgs)
            clean = histograms(ctx, clean_s)
            want = sorted(msgs)
            for a in alphas:
                e = noise_for(ctx, a, kind=kind, seed=0) if a else 0
                res = decode(ctx, clean_s + e, d=d, ell_max=ell_max)
                # past the bound the decoder can return spurious codewords, so
                # "how many came back" and "how many of the right ones came back"
                # are not the same number. Shade by the second.
                hit = sum(1 for u in res["messages"] if u in want)
                rows.append({
                    "kind": kind, "t": int(t), "alpha": float(a),
                    "n_rec": len(res["messages"]),
                    "n_correct": hit,
                    "frac": hit / t,
                    "exact": res["messages"] == want,
                    "stage_a_ok": bool(np.array_equal(res["hist"], clean)),
                })
                done += 1
                if progress:
                    progress(done, total)
    return rows


def attempt(ctx, Y, budget, ell, d=1):
    """Interpolate at an arbitrary (budget, ell) rather than the one unbundle picks,
    to show that a bigger ell cannot rescue a bundle past the bound."""
    from sage.coding.guruswami_sudan.interpolation import (
        _monomial_list, _interpolation_matrix_given_monomials)

    N, K, points = ctx["N"], ctx["K"], ctx["points"]
    P = [(points[j], points[a]) for j in range(N) for a in range(N) if Y[a, j]]
    monomials = list(_monomial_list(budget, ell, K - 1))
    M = _interpolation_matrix_given_monomials(P, d, monomials)
    ker = M.right_kernel()
    dim = int(ker.dimension())                   # a Sage Integer; JSON wants a plain int
    if dim == 0:
        return {"monomials": len(monomials), "kernel": 0, "roots": 0, "messages": []}

    sol = ker.basis()[0]
    xv, yv = M.base_ring()["x", "y"].gens()
    Q = sum(xv ** a * yv ** b * sol[i] for i, (a, b) in enumerate(monomials))
    roots = ctx["root_finder"](Q, maxd=K - 1)
    kept = [f for f in roots
            if score(ctx, [ctx["from_sage"][f(xx)] for xx in points], d * Y) >= d * N]
    return {"monomials": len(monomials), "kernel": dim, "roots": len(roots),
            "messages": sorted(message_of_poly(ctx, f) for f in kept)}
