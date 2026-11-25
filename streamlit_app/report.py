"""The notebook's printed output, as functions that return the text.

histogram_recovery.ipynb says everything with aligned monospace prints -- the
histogram tables, the step-by-step walkthroughs, the sweep table. The app shows the
same strings rather than re-rendering the same data as widgets, so the two read
identically side by side.
"""

import numpy as np


MAX_COLS = 36          # N reaches 625; past this the rows stop fitting on a screen


def fmt_list(xs, limit=MAX_COLS):
    """A list, truncated. A length-625 codeword is a 3,000-character line."""
    xs = list(xs)
    if len(xs) <= limit:
        return str(xs)
    return f"{str(xs[:limit])[:-1]}, ... {len(xs) - limit} more]"


def show_histogram(ctx, hist, title, max_rows=None, against=None, max_cols=MAX_COLS):
    """The notebook's show_histogram, plus truncation and an optional diff row.

    `against` is a clean histogram to compare with: columns that came back wrong get
    an arrow under them, which is the whole point of the noise section.
    """
    N = ctx["N"]
    cols = list(range(min(N, max_cols or N)))
    nz = [a for a in range(N) if hist[a].any()]
    rows = nz if max_rows is None else nz[:max_rows]

    out = [f"{title}   ({len(nz)} of {N} symbol-rows are non-empty)",
           "        position:  " + "".join(f"{j:>3}" for j in cols)]
    for a in rows:
        out.append(f"   symbol {a:>2}:        "
                   + "".join(f"{hist[a][j]:>3}" for j in cols))
    if len(rows) < len(nz):
        out.append(f"   ... {len(nz) - len(rows)} further non-empty rows not shown")
    sums = hist.sum(axis=0)
    out.append("   column sums:        " + "".join(f"{sums[j]:>3}" for j in cols))
    if against is not None:
        bad = [j for j in range(N) if not np.array_equal(hist[:, j], against[:, j])]
        out.append("   wrong column:       "
                   + "".join(f"{'^^' if j in bad else '':>3}" for j in cols))
        out.append(f"   {len(bad)} of {N} columns differ from the noiseless histogram")
    if len(cols) < N:
        out.append(f"   ... showing positions 0..{cols[-1]} of {N}")
    return "\n".join(out)


def encode_steps(ctx, message, codeword, hv):
    """Part 1 of the notebook: eq. (1) one layer at a time."""
    p, m, K, N, DIM = ctx["p"], ctx["m"], ctx["K"], ctx["N"], ctx["dim"]
    enc, vec = ctx["enc"], ctx["tab"][0]
    out = [f"STEP 0 -- the message: K = {K} symbols from F_({p}^{m})",
           f"   u = {message}",
           f"   each symbol is really {m} digits base {p}:"]
    for i, s in enumerate(message):
        out.append(f"      u[{i}] = {s:>2}  ->  digits {list(map(int, vec[s]))}")

    terms = " + ".join(f"{c}" if k == 0 else f"{c}*x^{k}"
                       for k, c in enumerate(message) if c) or "0"
    out += ["",
            "STEP 1 -- Reed-Solomon: read the message as polynomial coefficients",
            f"   f(x) = {terms}",
            f"   evaluate f at all {N} field elements:",
            f"   c = {fmt_list(codeword)}",
            f"   {K} symbols in, {N} out -- that redundancy is what the decoder spends.",
            "",
            "STEP 2 -- Hadamard: replace each symbol c_j by column c_j of the table",
            f"   the exponent table L is {N}x{N};  "
            f"L[b,a] = <digits(a), digits(b)> mod {p}",
            f"   picking out the columns named by the codeword gives a {N}x{N} grid:"]
    show = min(N, 24)
    grid = ctx["L"][:show, codeword[:show]]
    for row in grid:
        out.append("      " + "".join(f"{v:>3}" for v in row))
    if show < N:
        out.append(f"      ... top-left {show}x{show} corner of the {N}x{N} grid")
    out += ["",
            f"STEP 3 -- phi: send exponent k to the point at angle 2*pi*k/{p}",
            f"   p = {p}, so there are only {p} possible entries:"]
    for k in range(p):
        z = enc["phi"](k, p)
        out.append(f"      exponent {k} -> {z:+.3f}   (angle {360 * k // p:>3} degrees)")
    out += ["",
            "   applying phi to the whole grid and reading it down each column, left",
            f"   to right, gives the hypervector. First 18 of {DIM} entries:",
            f"   {np.array2string(np.round(hv[:18], 3), max_line_width=200)}",
            "",
            f"   dim = {hv.shape[0]} = p^2m",
            f"   |entry| = 1 everywhere: {np.allclose(np.abs(hv), 1.0)}   "
            f"(all on the unit circle)",
            f"   norm = {np.linalg.norm(hv):.1f} = sqrt(dim) -- true of EVERY codeword"]
    return "\n".join(out)


def show_matrix(title, M, max_rows=8, max_cols=8, note=None):
    """A corner of a matrix, printed. Complex entries get two decimals.

    Rows are blocks: row j is the N entries that Stage A processes together, which is
    the orientation s_j = H h_j' + e_j' is written in. A full N x N complex matrix is
    far too wide to read at any N worth using, so this shows a corner and says so.
    """
    M = np.asarray(M)
    rows, cols = M.shape
    nr, nc = min(rows, max_rows), min(cols, max_cols)
    if np.iscomplexobj(M):
        def cell(z):
            return f"{z.real:+.2f}{z.imag:+.2f}j"
        width = 12
    else:
        def cell(v):
            return f"{int(v)}"
        width = 4

    out = [title]
    out.append("      block \\ entry " + "".join(f"{i:>{width}}" for i in range(nc)))
    for j in range(nr):
        out.append(f"      block {j:>3}:    "
                   + "".join(f"{cell(M[j][i]):>{width}}" for i in range(nc)))
    if nr < rows or nc < cols:
        out.append(f"      ... showing the top-left {nr}x{nc} of {rows}x{cols}")
    if note:
        out.append(f"      {note}")
    return "\n".join(out)


def phi_sections(ctx, msgs, exps, hvs, S, E, SE, alpha, kind, total=None):
    """The matrices the noise section operates on, before and after phi.

    Before phi a codeword is a grid of exponents in 0..p-1. After phi each exponent
    becomes a p-th root of unity. Bundling then happens on the complex side only --
    there is no exponent-domain counterpart, because a sum of roots of unity is not a
    root of unity. That is exactly why the bundle leaves the unit circle, and why
    Stage A has to lattice-decode rather than just read a table.
    """
    p, N = ctx["p"], ctx["N"]
    total = total or len(msgs)
    t = len(msgs)
    more = ([f"   ... the other {total - t} bundled codeword(s) are not shown; each "
             f"is another {N}x{N} matrix."] if total > t else [])
    out = [f"BEFORE PHI -- the Hadamard exponent grid of each bundled codeword.",
           f"   integers in 0..{p - 1}. Row j is block j: the N entries Stage A",
           f"   processes together.", ""]
    for i, (u, X) in enumerate(zip(msgs, exps), 1):
        out.append(show_matrix(f"   codeword {i} of {total}, message {u}:", X))
        out.append("")
    out += more + [""]

    out += [f"AFTER PHI -- each exponent k becomes exp(2*pi*i*k/{p}).",
            f"   every entry now sits on the unit circle.", ""]
    for i, (u, X) in enumerate(zip(msgs, hvs), 1):
        out.append(show_matrix(f"   codeword {i} of {total}, message {u}:",
                               X.reshape(N, N)))
        out.append("")
    out += more + [""]

    out += [f"THE BUNDLE -- add all {total} complex matrices, entry by entry.",
            "   There is no 'before phi' for this one: a sum of roots of unity is not",
            "   a root of unity, so the bundle has no exponent form. That is the whole",
            "   reason Stage A has to lattice-decode instead of reading a table.", "",
            show_matrix("   S = sum of the matrices above:", S.reshape(N, N),
                        note=f"|entry| ranges {np.abs(S).min():.2f} .. "
                             f"{np.abs(S).max():.2f}  (1.00 everywhere before bundling)"),
            "",
            f"THE NOISE -- e, kind '{kind}', alpha = {alpha:g}.", "",
            show_matrix("   e:", E.reshape(N, N),
                        note=f"||e_j||_2 = {np.linalg.norm(E.reshape(N, N)[0]):.3f} "
                             f"in every block; Corollary 2 permits < "
                             f"{np.sqrt(N) / 2:.3f}"),
            "",
            "THE NOISY BUNDLE -- S + e, which is what Stage A actually receives.", "",
            show_matrix("   s = S + e:", SE.reshape(N, N),
                        note=f"|entry| ranges {np.abs(SE).min():.2f} .. "
                             f"{np.abs(SE).max():.2f}"),
            "",
            "   Stage A takes row j of this, calls it s_j, and returns",
            "   round( (1/N) Re( H' s_j ) ) as the histogram column h_j."]
    return "\n".join(out)


def ks_section(ctx, message, codeword, W):
    """sidorenko_fischer.ipynb Step 1: the same codeword, one-hot.

    Shown next to the Hadamard grid because the contrast is the point -- one stores a
    phase exponent per cell, the other stores a bare 1.
    """
    q, N, p = ctx["q"], ctx["N"], ctx["p"]
    out = ["KAUTZ-SINGLETON -- the same codeword, written the way Sidorenko-Fischer",
           "wants it: one-hot each RS symbol into a column.", "",
           f"   message {message}  ->  RS codeword {fmt_list(codeword)}", "",
           ]
    out.append(show_histogram(
        ctx, W, f"   W(c), rows = symbol 0..{q - 1}, columns = position 0..{N - 1}:"))
    out += ["",
            f"   exactly one 1 per column: {bool((W.sum(axis=0) == 1).all())}",
            f"   |supp(W)| = {int(W.sum())} = N = {N}",
            "",
            f"   every entry here is 0 or 1. The Hadamard grid above stores an",
            f"   exponent in 0..{p - 1} -- a phase. W throws the phase away and keeps",
            f"   only THAT a symbol occurred, which is what turns superposition into",
            "   a set-union problem: Y is the OR of the t matrices W(c_1)..W(c_t),",
            "   and M = dY is what Algorithm 2 consumes."]
    return "\n".join(out)


def histogram_section(ctx, h, W):
    """Stage A run back on this one hypervector -- the third way to write a codeword.

    histogram_recovery.ipynb's t = 1 cell makes the point: for a single codeword the
    histogram IS the one-hot table. Bundling is the only thing that separates them.
    """
    N = ctx["N"]
    same = bool(np.array_equal(h, W))
    out = ["HISTOGRAM -- Stage A run straight back on this one hypervector.", "",
           "   h_j = round( (1/N) Re( H' s_j ) )     -- the paper's Corollary 1", "",
           show_histogram(ctx, h, "   h (counts):"),
           "",
           f"   every column sums to 1 = t, because only one codeword went in",
           f"   argmax down each column reads the codeword back: "
           f"{fmt_list(int(a) for a in h.argmax(axis=0))}",
           f"   h is identical to W(c) above: {same}",
           "",
           "   so for a SINGLE codeword the histogram is just the one-hot table.",
           "   What bundling changes is only this: the columns start summing to t,",
           "   and a cell holding 2 marks two codewords sharing a symbol there.",
           "   Throw the counts away and you are back at a 0/1 matrix -- that is Y."]
    return "\n".join(out)


def bundle_section(ctx, rows, t_limit, u_max):
    """Part 2 of the notebook: what addition does as t grows."""
    DIM = ctx["dim"]
    out = [f"bundles of t = 1 .. {t_limit}, each containing the one before:", "",
           f"   {'t':>3} {'|entry| range':>16} {'norm':>8} {'sqrt(t*dim)':>12}"]
    for r in rows:
        t = r["t"]
        if t <= 6 or t in (10, u_max, u_max + 1, t_limit):
            rng = f"{r['abs_min']:.1f} .. {r['abs_max']:.1f}"
            out.append(f"   {t:>3} {rng:>16} {r['norm']:>8.1f} "
                       f"{np.sqrt(t * DIM):>12.1f}")
    out += ["",
            f"   every bundle is still {DIM} numbers -- that is the point of a VSA:",
            "   the representation does not grow when you add more objects.",
            "   the norm grows like sqrt(t), which is why each item's share shrinks."]
    return "\n".join(out)


def by_hand(ctx, msgs, t, hist, Y, res, hvs_head=None, max_rows=None):
    """Cells 14 / 16 / 18 of the notebook: the t = 1, 2, 3 walkthroughs."""
    N, DIM = ctx["N"], ctx["dim"]
    want = sorted(msgs[:t])
    out = []

    if t == 1:
        out += [f"bundled {t} codeword: message {msgs[0]}",
                f"  RS codeword c   = {fmt_list(res['codewords'][0])}",
                "  ... but what is actually summed is the HYPERVECTOR, i.e. c after",
                f"  Hadamard and phi.  First 8 of {DIM} entries:",
                f"     {np.array2string(np.round(hvs_head[0][:8], 3), max_line_width=200)}",
                f"     dim {DIM}, |entry| = 1 everywhere: "
                f"{bool(np.allclose(np.abs(hvs_head[0]), 1.0))}",
                "  the RS codeword is shown only for reference -- it is what Stage A",
                "  will bring back out.", ""]
    else:
        out += [f"bundled {t} codewords (RS form, for reference):"]
        for u, cw in zip(msgs[:t], res["codewords"]):
            out.append(f"   message {str(u):<14} codeword {fmt_list(cw)}")
        out += ["", "the addition happens on the hypervectors, not on these. "
                    "First 6 entries:"]
        for u, h in zip(msgs[:t], hvs_head):
            out.append(f"   hv of {str(u):<14} "
                       f"{np.array2string(np.round(h[:6], 2), max_line_width=200)}")
        out.append(f"   {'sum (the bundle)':<21} "
                   f"{np.array2string(np.round(res['bundle_head'], 2), max_line_width=200)}")
        out.append(f"   |entry| of the bundle: {res['abs_min']:.1f} .. "
                   f"{res['abs_max']:.1f}  <- no longer on the unit circle")
        out.append("")

    out.append(show_histogram(ctx, hist, "STAGE A -- histogram h (counts):",
                              max_rows=max_rows))

    if t == 1:
        out += ["", f"  every column sums to {t} = t, so the data announces how many "
                    "are in there",
                "  and argmax down each column reads the codeword straight back:",
                f"     {fmt_list(int(a) for a in hist.argmax(axis=0))}"]
    else:
        out += ["", f"  max count in any cell: {int(hist.max())}  "
                    f"(a {int(hist.max())} means "
                    "that many codewords share a symbol there)"]
        coll = [j for j in range(N) if (hist[:, j] > 1).any()]
        if t == 2:
            out.append(f"  columns where the two codewords collided: {coll}")

    out += ["", "STAGE B -- Y (did this symbol appear? counts thrown away):",
            show_histogram(ctx, Y, "  Y (0/1):", max_rows=max_rows),
            "",
            f"  every entry is now 0 or 1 -- this is the OR of the {t} "
            f"Kautz-Singleton",
            f"  matrices W(c_1)..W(c_{t}), and M = dY is what Algorithm 2 consumes.",
            f"  n_P = {int(Y.sum())} observed pairs,  t*N = {t * N},  "
            f"lost to collisions: {t * N - int(Y.sum())}"]

    if res.get("error"):
        out += ["", f"  {res['error']}"]
        return "\n".join(out)

    out += ["", f"  interpolate through {res['n_P']} points (ell = {res['ell']}), "
                "factor, filter"]
    for poly, u in zip(res["polys"], res["messages"]):
        out.append(f"     f(x) = {poly:<30} ->  message {u}")
    got = res["messages"]
    if got == want:
        out += ["", f"  recovered exactly the {t} bundled message(s), "
                    "nothing spurious."]
    else:
        out += ["", f"  recovered {len(got)} message(s), NOT the {t} that went in.",
                f"     bundled:  {want}",
                f"     returned: {got}"]
    return "\n".join(out)


def gaussian_model(ctx, sigma, E, blocks_g, blocks_p, alpha_equiv):
    """The AWGN model, its covariance stated exactly, and the moments measured."""
    N = ctx["N"]
    re, im = E.real.ravel(), E.imag.ravel()
    thr = np.sqrt(N) / 2
    return "\n".join([
        f"THE MODEL -- e ~ CN(0, sigma^2 I),  sigma = {sigma:g}",
        "",
        f"   every one of the {ctx['dim']} entries, independently:",
        "      e_k = (x + iy) * sigma / sqrt(2),   x, y ~ N(0, 1) independent",
        "",
        "   so, writing I for the identity:",
        f"      e      ~ CN(0, sigma^2 I)      over C^{ctx['dim']}",
        f"      e_j    ~ CN(0, sigma^2 I)      over C^{N}, one block",
        f"      as a real vector of length {2 * ctx['dim']}:  N(0, (sigma^2/2) I)",
        "",
        "   measured against that:",
        f"      Var(Re e_k) = {re.var():.5f}      predicted sigma^2/2 = {sigma**2/2:.5f}",
        f"      Var(Im e_k) = {im.var():.5f}      predicted sigma^2/2 = {sigma**2/2:.5f}",
        f"      E|e_k|^2    = {np.mean(np.abs(E)**2):.5f}      predicted sigma^2   "
        f"= {sigma**2:.5f}",
        f"      corr(Re,Im) = {np.corrcoef(re, im)[0, 1]:+.5f}      0 -> circular",
        f"      E[e_k^2]    = {np.mean(E**2).real:+.5f}      0 -> circularly "
        f"symmetric",
        "",
        "   note the variance is sigma^2, not sigma, and the distribution is complex:",
        "   each of the real and imaginary parts carries sigma^2/2.",
        "",
        "AGAINST THE FIXED-NORM PROBE",
        "",
        f"   the noise tab pins ||e_j||_2 to the same value in every block. Gaussian",
        f"   noise does not -- and that one difference is what turns the cliff at the",
        f"   Corollary 2 threshold into a curve.",
        "",
        f"   {'model':<28} {'min':>8} {'mean':>8} {'max':>8} {'std':>8}",
        f"   {f'Gaussian, sigma={sigma:g}':<28} {blocks_g.min():>8.3f} "
        f"{blocks_g.mean():>8.3f} {blocks_g.max():>8.3f} {blocks_g.std():>8.3f}",
        f"   {f'probe, alpha={alpha_equiv:g}':<28} {blocks_p.min():>8.3f} "
        f"{blocks_p.mean():>8.3f} {blocks_p.max():>8.3f} {blocks_p.std():>8.3f}",
        "",
        f"   the probe's std is 0 by construction. Corollary 2's threshold is "
        f"{thr:.3f},",
        f"   and {int((blocks_g > thr).sum())} of the {N} Gaussian blocks are above it.",
    ])


def stage_a_derivation(ctx, rows):
    """a = Re(H'e)/N is i.i.d. normal, with the variance the derivation predicts."""
    N = ctx["N"]
    out = ["STAGE A UNDER THIS MODEL", "",
           "   H has orthogonal columns of squared norm N, so",
           "      (H' e)_i ~ CN(0, N sigma^2)",
           "   and therefore",
           "      a_i = Re(H' e)_i / N  ~  N(0, sigma^2 / 2N)   i.i.d.", "",
           f"   the 1/sqrt(N) is the transform averaging the noise down -- that is the",
           f"   processing gain that makes sigma near 1 survivable at all.", "",
           f"   {'sigma':>7} {'predicted std':>15} {'measured std':>14} "
           f"{'measured mean':>15}"]
    for sg, pred, meas, mean in rows:
        out.append(f"   {sg:>7g} {pred:>15.5f} {meas:>14.5f} {mean:>15.5f}")
    out += ["", f"   predicted std is sigma / sqrt(2N) = sigma / {np.sqrt(2*N):.3f}",
            "",
            "   Lemma 1 needs EVERY coordinate of a inside (-0.5, 0.5), and there are",
            f"   N blocks x N coordinates = {N*N} of them. They are independent, so",
            "",
            "      P(Stage A exact) = erf( sqrt(N) / 2 sigma ) ^ (N^2)",
            "",
            "   an equality, not a bound."]
    return "\n".join(out)


def closed_form_check(rows, trials):
    """Closed form against Monte Carlo."""
    out = [f"CLOSED FORM vs MONTE CARLO ({trials} trials per cell)", "",
           f"   {'N':>6} {'sigma':>8} {'P closed form':>15} {'P measured':>12}"]
    for N, sg, pred, emp in rows:
        out.append(f"   {N:>6} {sg:>8g} {pred:>15.4f} {emp:>12.4f}")
    return "\n".join(out)


def sigma_scaling(qs, sigma_half):
    """Where the usable threshold sits, against Corollary 2, as q grows."""
    out = ["THE USABLE THRESHOLD, AND HOW THE GAP GROWS", "",
           "   sigma_1/2 is where Stage A succeeds half the time. Corollary 2's",
           "   guarantee is a norm, so convert: Gaussian noise of level sigma gives a",
           "   mean block norm of about sigma*sqrt(q), i.e. a multiple 2*sigma of the",
           "   sqrt(q)/2 threshold.", "",
           f"   {'q':>6} {'sigma_1/2':>11} {'mean ||e_j||':>13} {'sqrt(q)/2':>11} "
           f"{'x Cor. 2':>10}"]
    for q in qs:
        sh = sigma_half[q]
        out.append(f"   {q:>6} {sh:>11.4f} {sh*np.sqrt(q):>13.3f} "
                   f"{np.sqrt(q)/2:>11.3f} {2*sh:>9.2f}x")
    out += ["",
            "   Corollary 2 is sufficient, never necessary, and the gap WIDENS with q.",
            "   Asymptotically sigma_1/2 ~ sqrt(q / log q): the transform averages noise",
            "   over q entries (gain sqrt(q)) while the number of coordinates that must",
            "   all stay in range grows as q^2 (cost sqrt(log q)). The gain wins."]
    return "\n".join(out)


def noise_header(ctx, alpha, kind, norms):
    """What the noise is, against the level Corollary 2 permits."""
    N = ctx["N"]
    permitted = np.sqrt(N) / 2
    verdict = ("INSIDE the guarantee" if alpha < 1 else
               "OUTSIDE the guarantee -- Corollary 2 says nothing here")
    return "\n".join([
        f"NOISE -- kind '{kind}', alpha = {alpha:g}",
        f"   Corollary 2 permits  ||e_j||_2 < sqrt(N)/2 = sqrt({N})/2 = {permitted:.3f}",
        f"   this noise has       ||e_j||_2 = {norms.max():.3f} in every one of the "
        f"{N} blocks",
        f"   that is {alpha:g}x the permitted level -> {verdict}",
    ])


def sweep_table(ctx, rows, u_max):
    """The notebook's 'Until it breaks' table."""
    N = ctx["N"]
    out = [f"{'t':>4} {'n_P':>6} {'t*N':>6} {'ell':>5} {'recovered':>10} "
           f"{'exact?':>8} {'ms':>8}", "-" * 54]
    for r in rows:
        out.append(f"{r['t']:>4} {r['n_P']:>6} {r['t'] * N:>6} {r['ell']:>5} "
                   f"{r['n_rec']:>10} {str(r['exact']):>8} {r['ms']:>8.0f}")
    ok = [r["t"] for r in rows if r["exact"]]
    out += ["", f"last t fully recovered: {max(ok) if ok else 0}      u_max = {u_max}"]
    return "\n".join(out)
