# Research plan: fractional recovery, inside Guruswami–Sudan, and spiking noise

Three threads, from the working notes. The useful discovery while scoping them is that
**they are the same thread**, and that both papers already in this repo contain most of
the machinery.

---

## 0. The connection

`core.py` currently does this, in `support_matrix`:

```python
def support_matrix(hist):
    return (hist > 0).astype(int)        # counts -> 0/1
```

That single line throws away the histogram's counts and keeps only *whether* a symbol
occurred. Everything downstream — the interpolation point set `P`, the score filter,
the `d*N` threshold — is built on that hard, binary decision.

Guruswami–Sudan §3.4 ("Weighted curve fitting", **Problem 3**) says you do not have to:

> INPUT: n points, n non-negative integer weights `w_1, ..., w_n`, parameters k and t.
> OUTPUT: All polynomials p such that `sum_{i : p(x_i) = y_i} w_i` is at least t.
>
> The algorithm of Section 2 can be modified as follows: In Step 1, find a polynomial Q
> which has a singularity of order `w_i * r` at the point `(x_i, y_i)`.

Compare `core.py:score`, which is `sum over positions of M[symbol, position]` with `M =
d*Y` binary — i.e. Problem 3 with every weight equal. **The histogram counts are the
weights the theorem is asking for**, and under noise the natural weights are softer
still (confidence from the Stage A rounding residual).

So:

| thread | what it actually is |
|---|---|
| 1. fractional success | the measurement you need before any of this is legible |
| 2. inside GS, above the bound | per-point multiplicities `w_i * r` instead of a scalar `d` |
| 3. spiking noise | why you want soft weights — **and already a theorem**, Prop. 2 |

And **Proposition 2 of `histo.pdf` is the spiking noise model**, stated and proved:

> For nonnegative integers `s, r` such that `(r+1)s < N`, our algorithm successfully
> recovers `c_1, ..., c_t` whenever (a) `t` obeys Theorem 1; and (b) every `e_i`
> satisfies `||e_i||_2 < sqrt(N)/2` **except for at most `s` many indices** `i` in
> which `e_i` is in `E_r`.

Mapping onto the proposed parameters:

| notes | paper | meaning |
|---|---|---|
| `sigma` | the `||e_i||_2 < sqrt(N)/2` condition | ordinary noise on the non-spiked blocks |
| `mu`, number of spikes | **`s`** | how many blocks are allowed to blow up |
| `nu`, spike intensity | **`r`** | how far a spike drags that block's histogram (ell_1) |
| the bound | **`(r+1)s < N`** | the spiking analogue of `t(K-1) < N` |

So thread 3 is not "invent a model". It is "instantiate Proposition 2, and find out
whether `(r+1)s < N` is tight the way `u_max` turned out not to be."

One caution the paper forces: its noise is **per block** `e_i` in `C^N` (block `i` is
coordinates `iN .. iN+N-1`), not per coordinate. "A few coordinates become very large"
is a *coordinate*-level statement that lands inside one block. Both are worth modelling,
and they are not the same experiment — see Phase 3.

---

## 1. Reading plan

### Tier 1 — before writing any code (about 3 hours)

| # | source | pages | extract |
|---|---|---|---|
| 1 | `histo.pdf` §V-A + **Proposition 2** + its proof | ~1 | the `(r+1)s < N` condition and `E_r`'s definition. The proof is short and is the whole spiking model. |
| 2 | GS paper **§3.3, Theorem 12** | ~0.5 | "uncertain receptions": `l` candidates per position, works when `l < n/k`. With `n = N = 27`, `k = K-1 = 2`, that is `l < 13.5` — **this is `u_max`, derived a completely different way.** |
| 3 | GS paper **§3.4, Problem 3** | ~1 | weighted curve fitting. The `w_i * r` multiplicity rule is the workaround for thread 2. |
| 4 | GS paper §2.2–2.3, **Lemmas 3–6, Theorem 7** | ~2 | the core algorithm and, in Lemma 6, exactly where multiplicity enters the counting. This is the algebra to re-derive by hand for `t = 14`. |

After tier 1 you should be able to answer: *why does Theorem 12 give the same 13 as
Lemma 2, when one is about candidate-list size and the other about score separation?*

### Tier 2 — the soft-decision line (about 4 hours)

| # | source | why |
|---|---|---|
| 5 | `Low-Complexity_List_Decoding_of_Reed-Solomon_Coded_Pulse_Position_Modulation.pdf` | **closest prior art in the repo.** It list-decodes Kautz–Singleton codes with GS *and Koetter–Vardy*. Same code, same one-hot-per-position structure. Read for how they build the multiplicity matrix. |
| 6 | Koetter & Vardy, *Algebraic soft-decision decoding of Reed–Solomon codes*, IEEE-IT 2003 | not in the repo — fetch. The multiplicity matrix `M` from a reliability matrix `Pi`, and the proportionality `M ~ lambda * Pi`. This is the principled answer to "how should the histogram counts become weights". |
| 7 | McEliece, *The Guruswami–Sudan decoding algorithm for Reed–Solomon codes* (JPL IPN Progress Report 42-153, 2003) | not in the repo — fetch. The clearest pseudocode anywhere, with worked numbers. Use it to check the hand derivation. |

### Tier 3 — reference, read as needed

- `linear.pdf` Theorem 2 — binding recovery. Already understood; no action.
- Sage source: `gs_decoder.py` (`gs_satisfactory`, `parameters_given_tau`,
  `guruswami_sudan_decoding_radius`) and `interpolation.py`
  (`_monomial_list`, `_interpolation_matrix_given_monomials`,
  `_interpolation_max_weighted_deg`). ~1500 lines total; only the five named functions
  matter.

---

## 2. Learning plan

Dependency order. Each item has a check — if you cannot do the check, the next item
will not land.

**(a) Weighted degree.** The `(1, K-1)`-weighted degree of `x^i y^j` is `i + j(K-1)`,
because `y` stands in for a polynomial of degree `< K`.
*Check:* say why `ell_sat(d) = floor(dN/(K-1))` and why no larger `ell` adds a monomial.

**(b) Multiplicity / singularity of order s.** `Q` vanishes to order `s` at `(x_0, y_0)`
when every Hasse derivative up to total order `s-1` vanishes there — `s(s+1)/2` linear
conditions per point.
*Check:* derive `s(s+1)/2` yourself, and explain why `core.py:attempt` passes `d` as
Sage's `s`.

**(c) The counting argument, and its limits.** Unknowns `#{(i,j) : i + j(K-1) <= dN}`
versus constraints `n_P * d(d+1)/2`. Both are quadratic in `d`.
*Check:* explain the `d = 4, t = 14` row of REPORT.md §5 — deficit of 80, yet kernel
dimension 27. **Rank, not count, is what matters**, and this is the single most
important thing in the project right now.

**(d) The two "it works" lemmas, and what sufficiency costs.** `Q(x, f(x))` has degree
`<= dN` but `> dN` roots when agreement is high, so it is identically zero, so
`(y - f)` divides `Q`. Separately, Lemma 2 caps an impostor at `t(K-1)`.
*Check:* state precisely which claim fails at `t = 14` — and note neither is "decoding
is impossible". See `GURUSWAMI_SUDAN.md` §6.

**(e) Soft decision and the multiplicity matrix.** Move from "point `(x_j, a)` is in
`P`" to "point `(x_j, a)` carries multiplicity `m_{a,j}`". Score becomes
`sum_j m_{f(x_j), j}`; the cost becomes `sum_{a,j} m_{a,j}(m_{a,j}+1)/2`.
*Check:* recover hard-decision GS as the special case `m_{a,j} = d * Y_{a,j}`.

**(f) Where fractional recovery can come from.** Stage B returns the roots of one `Q`.
Ask: can it return a strict non-empty subset of the truth? Under what corruption?
*Check:* predict the answer before running Phase 0 — then see whether you were right.

---

## 3. Coding plan

### Phase 0 — the fractional metric (half a day; do this first)

`noise_grid` already computes `n_correct` and `frac`. Nothing else does: `sweep`,
`noise_sweep`, `decode` and `gs_bound.ipynb` all report a boolean `exact`.

- `core.py`: one plain function, used everywhere.

```python
def recovery(found, truth):
    """Graded recovery, not a boolean: what fraction of the bundled codewords came
    back, and how much junk came with them."""
    R = {tuple(u) for u in found}
    T = {tuple(u) for u in truth}
    hit = len(R & T)
    return {"hit": hit, "n_true": len(T), "n_out": len(R),
            "recall": hit / len(T) if T else 1.0,
            "spurious": len(R - T),
            "exact": R == T}
```

- Thread it through `sweep`, `noise_sweep`, `decode`, and `gs_bound.ipynb`'s `measure`.
- Keep `exact` as a derived field so nothing already written breaks.
- **The first question to answer:** is `recall` ever strictly between 0 and 1? A
  histogram is decoded by one shared `Q`; if `Q` fails to exist, *everything* is lost at
  once and recall is binary. If it exists but is wrong, partial recovery is possible.
  Run `t` fixed, `sigma` scanning, and plot the recall distribution rather than its mean
  — a bimodal histogram and a graded one mean completely different things, and the mean
  hides which.

### Phase 1 — instrument the decoder (1 day)

Stop inferring the algorithm's behaviour from its output.

- `core.py`: `attempt` already returns `monomials`/`kernel`/`roots`. Add rank and the
  per-stage split.

```python
def attempt_verbose(ctx, Y, budget, ell, d=1):
    """Everything attempt() returns, plus WHY: matrix shape, rank, the rank deficiency
    that is the real reason a kernel exists, and per-stage timings."""
```

  Record `M.rank()` alongside `M.nrows()`. For `d = 4, t = 14`: 3050 rows, 2970 columns,
  kernel 27 implies rank 2943, so **107 rows are dependent**. Where do they come from?
  Collisions in `Y` (two bundled codewords sharing a symbol at a position) are the first
  hypothesis and are directly checkable — `n_P < tN` measures exactly that.
- Cross-check against Sage's own feasibility predicates:
  `GRSGuruswamiSudanDecoder.gs_satisfactory(tau, s, l)` and `parameters_given_tau`.
  They encode the *classical* conditions; find the first `(t, d)` where our setting
  succeeds and `gs_satisfactory` says no, and write down why.
- Fix the real bug this already exposed: `core.unbundle` defaults to `ell_max = 40`, but
  `ell_sat(4) = 54`. **The shipped decoder cannot reach `d >= 3` solutions at all.**
  Make `ell_max` default to `ell_sat(d)`.

### Phase 2 — weighted interpolation (2–3 days; the real workaround)

Sage's `_interpolation_matrix_given_monomials(points, s, monomials)` takes **one scalar
`s`**. Weighted curve fitting needs a multiplicity *per point*. Sage does not expose
that, so write it:

```python
def weighted_interpolation_matrix(ctx, mult, monomials):
    """GS Problem 3 / Koetter-Vardy: point (x_j, a) gets multiplicity mult[a, j],
    contributing mult*(mult+1)/2 rows instead of a uniform d*(d+1)/2."""
```

Model it on Sage's `eqs_affine` (`interpolation.py:126`) — the Hasse-derivative rows are
per point already; only the loop bound is uniform. Then:

- **Weights from counts.** `mult = d * hist` instead of `d * (hist > 0)`. Noiselessly
  this changes nothing when every count is 1, so it is a strict generalization — verify
  that first as a regression test.
- **Weights from Stage A confidence.** The rounding in
  `histograms` is `round(real(H^dagger s)/N)`; the **residual** `|x - round(x)|` is free
  reliability information currently discarded. Large residual means an unreliable cell.
  That is the `Pi` matrix Koetter–Vardy turns into `M`.
- Budget and score both change: `sum m(m+1)/2` constraints, threshold
  `sum_j m_{f(x_j), j}`. Re-derive `u_max` under weights — **it should move**, and that
  is the point.

### Phase 3 — spiking noise (2 days, plus compute)

Two models, deliberately distinct. Both in `core.py`, beside `gaussian_noise`.

```python
def spiking_noise(ctx, sigma, mu, nu, seed=0, mode="block"):
    """Ordinary CN(0, sigma^2) everywhere, plus mu spikes of scale nu.

    mode="block": mu whole blocks e_i get the extra noise -- this is Proposition 2's
    model, where mu is the paper's s.
    mode="coord": mu individual coordinates blow up -- the literal 'a few coordinates
    become really large' case, which is NOT the same experiment.
    """
```

Then measure, in this order:

1. **Calibrate `nu` to `r`.** Proposition 2's `r` is an ell_1 histogram deviation, not a
   noise scale. For a block with spike intensity `nu`, measure the induced
   `||h_hat - h||_1` empirically and build the `nu -> r` map. Without this the model is
   not connected to the theorem and nothing can be compared.
2. **Test `(r+1)s < N`.** Sweep `(mu, r)` and shade by `recall`. Is the boundary at
   `(r+1)mu = 27` sharp, or is there slack — as there was at `u_max`?
3. **Compare against plain AWGN at matched total noise power.** The interesting claim is
   that concentrated noise is *easier* than spread noise, because Stage B's §IV-B3
   tolerates a few bad blocks but Stage A's `sqrt(N)/2` is per-block. Matched power is
   the only fair comparison and is what makes this publishable rather than anecdotal.
4. **Re-check the Stage A closed form.** `erf(sqrt(N)/2sigma)^(N^2)` assumes i.i.d.
   Gaussian. Under spikes it is wrong; derive the mixture version
   (`N - mu` clean blocks, `mu` spiked) and check against Monte Carlo. This is the
   "double check the connection to the R-S formula" item.
5. Only then the correlated version — `e ~ CN(0, D)` for a general covariance `D`.
   Spikes are the special case `D` diagonal with `mu` large entries, so do not start
   here; arrive here.

### Phase 4 — the grids and the visualizer (compute-bound)

Deferred until Phases 0–3 land, because every grid below should be shaded by `recall`,
not by `exact` — regenerating them twice would be wasteful.

- `(t, sigma)` at `d = 2, 3, 4` for the layered `d` visualizer. Cost is the blocker:
  `d = 4` is ~300 s per decode, so a 15x15 grid is ~19 hours single-threaded. Use
  `noise_trials.py`'s existing multiprocessing pool, and cut `t` to a window around
  `u_max` rather than sweeping it.
- `(mu, nu)` spiking grids at fixed `t`, which are `d = 1` and therefore cheap.
- Only then the Streamlit layers.

---

## Order of work

1. **Phase 0** — half a day, and it determines what every later plot can say.
2. **Tier 1 reading** — 3 hours. Proposition 2 in particular, because it means Phase 3
   is a test of an existing theorem rather than a model-building exercise.
3. **Phase 1** — the `ell_max` bug is a real defect, and the rank question is the
   sharpest open problem in the project.
4. **Tier 2 reading** and **Phase 2** together — weighted interpolation is the largest
   genuinely new capability, and the PPM paper has already done it for this exact code.
5. **Phase 3**, then **Phase 4**.

## Open questions worth writing down now

- Is recall binary or graded? (Phase 0 settles it; predict first.)
- `d = 4, t = 14`: which 107 interpolation rows are dependent, and is their count
  predictable from `tN - n_P`?
- Theorem 12 gives `l < n/k = 13.5` and Lemma 2 gives `t(K-1) < N`, i.e. 13 both ways,
  from unrelated arguments. Is that a coincidence of these parameters, or the same
  bound in disguise? Test at another `(p, m, K)` where the two could differ.
- Under count-weighted multiplicities, does `u_max` move? If the counts are all 1 at
  distinct messages, weighting is vacuous — so does the gain only appear for *repeated*
  codewords (§IV-B3)?
- Is concentrated noise easier than spread noise at matched power?
