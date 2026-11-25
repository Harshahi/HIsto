# Why multiplicity lets you bundle past `u_max`

Deng & Raviv give `u_max = (N-1)/(K-1)` as the number of codewords histogram
recovery can unbundle. At `p = m = K = 3` that is **13**. We recover **14**, exactly
and with no spurious codewords, by raising the Guruswami–Sudan multiplicity `d` to 4
or more. This note says why, in terms that can be checked line by line.

Everything below is at `p = 3, m = 3, K = 3` unless stated.

---

## 0. What is set, what is derived, what is measured

Three different kinds of number appear in this note and they should not be confused.

**Set:** `p = 3`, `m = 3`, `K = 3`, the multiplicity `d`, and `t`, the number of
codewords bundled.

**Derived** — pure arithmetic from those:

| symbol | formula | value |
|---|---|---|
| `N` | `p^m` | 27 — codeword length and alphabet size |
| `dim` | `p^(2m)` | 729 — hypervector length |
| `w` | `K - 1` | **2** — what one power of `y` costs |
| `u_max` | `(N-1)/(K-1)` | 13 — the paper's bound |

**Measured** — the one input that is not arithmetic:

```
n_P = number of 1s in the support matrix Y
    = core.support_matrix(core.histograms(ctx, bundle)).sum()
    = 305   at t = 14
    = 285   at t = 13
```

`n_P` depends on *which* codewords were bundled, because collisions between them
leave cells of `Y` empty. Every statement below that quotes `n_P` is therefore a
statement about a particular bundle, not about `t` alone.

---

## 1. The budget

Stage A hands Stage B the support matrix `Y`: a 27x27 grid of 0s and 1s, where a 1 at
`(a, j)` means "some bundled codeword carries symbol `a` at position `j`". **Every 1
is an interpolation point** `(x_j, a)`. At `t = 14` there are 305 of them, in 729
cells — the grid is 42% full.

Guruswami–Sudan fits one bivariate curve `Q(x, y)` through all of them, vanishing to
order `d` at each, then factors it. Each bundled codeword appears as a factor
`(y - f(x))`.

The degree budget is not a free choice. It follows from Sage's formula plus one line
in `core.py`:

```
sage/coding/guruswami_sudan/interpolation.py
    _interpolation_max_weighted_deg(n, tau, s) = (n - tau) * s

streamlit_app/core.py:626
    tau = len(P) - N                    # = 305 - 27 = 278

    =>  B = (n_P - tau) * d = N * d = 27d
```

So **`B = dN`**, and Sage's bound is *exclusive* (`deg Q < B`). The monomials `Q` may
use are therefore

```
x^i y^j      with      i + w*j  <=  dN - 1
```

Two features of that region matter, and they are the two reasons `d` helps.

```
      d = 1, B = 27                      d = 2, B = 54
  j=13 |                              j=26 |#
  j=12 |#                             j=25 |##
  j=11 |##                            j=24 |###
   ...                                 ...
   j=2 |###########                    j=2 |#########################
   j=1 |############                   j=1 |##########################
   j=0 |#############                  j=0 |###########################

   height 13, area 196                height 26, area 756
```

The exclusive bound is also what makes the factor lemma fire. A codeword agreeing at
`A` positions gives `Q(x, f(x))` at least `A*d` roots against a degree of at most
`dN - 1`, so the lemma needs `A*d > dN - 1`, i.e. **`A = N`**: agreement at every one
of the 27 positions. That requirement is the same at every `d`, which is why a
codeword that loses a single symbol in Stage A is lost entirely.

---

## 2. Reason 1 — the height is the root capacity, and it *is* `u_max`

`Q` factors, and **every codeword recovered is a factor `(y - f(x))`, each eating one
degree of `y`.** A curve with `deg_y Q = 13` cannot have 14 such factors, for the same
reason a quadratic cannot have three roots.

The height of the region is

```
ell_sat = floor(dN / (K-1))
```

| `d` | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `deg_y Q <=` | **13** | 27 | 40 | 54 | 67 | 81 |

At `d = 1` the ceiling is **13**, which is exactly `u_max`. This is not a coincidence
and not a second theorem: `u_max = (N-1)/(K-1)` and `ell_sat(1) = floor(N/(K-1))` are
the same number.

So at `t = 14`, `d = 1` is not failing — it is being asked for 14 factors from a curve
with room for 13. The interpolation kernel is empty at *every* rung of the `ell`
ladder, including `ell_sat`. The decoder reports `ell = -1`: no curve exists. That is
silence, not error, and it holds at `sigma = 0` as well as under noise.

---

## 3. Reason 2 — the area has to beat the constraints

Having room for 14 roots is not enough; the curve has to exist. Vanishing to order `d`
at one point costs `d(d+1)/2` linear conditions (the Hasse derivatives). So:

```
unknowns     = number of monomials  = sum_j (B - w*j),  j = 0 .. floor((B-1)/w)
constraints  = n_P * d(d+1)/2
```

Expanding both for large `d`:

```
unknowns     ->  d^2 * N^2/(2w)        =  d^2 * 182.25
constraints  ->  d^2 * n_P/2  +  d * n_P/2   =  d^2 * 152.5  +  d * 152.5
```

Two observations.

**The quadratic coefficients favour the curve: 182.25 > 152.5.** That inequality is
just `n_P < N^2/w = 364.5`.

**But the constraints get a linear head start of `152.5 * d`.** At small `d` that term
dominates and the system is over-determined. Growing `d` is how you outgrow it:

```
constraints      n_P * (K-1)     d+1
------------  =  -----------  *  ---
  unknowns           N^2          d
```

The `d^2` cancels; the `(d+1)/d` factor is the entire gain, and it shrinks toward 1.

---

## 4. The table

At `t = 14`, `n_P = 305`:

| `d` | `B = dN` | `ell_sat` | unknowns | constraints | surplus | `+107` | **measured kernel** |
|---|---|---|---|---|---|---|---|
| 1 | 27 | 13 | 196 | 305 | −109 | −2 → 0 | **0** |
| 2 | 54 | 27 | 756 | 915 | −159 | −52 → 0 | **0** |
| 3 | 81 | 40 | 1,681 | 1,830 | −149 | −42 → 0 | **0** |
| 4 | 108 | 54 | 2,970 | 3,050 | **−80** | **27** | **27** |
| 5 | 135 | 67 | 4,624 | 4,575 | +49 | **156** | **156** |
| 6 | 162 | 81 | 6,642 | 6,405 | +237 | — | not measured |

`surplus = unknowns - constraints`. A positive surplus *guarantees* a kernel, so
counting alone would predict `t = 14` first works at `d = 5`. It works at `d = 4`.

### The same table at `t = 15`

At `t = 15`, `n_P = 323` (measured). §5's correction is `dep(15) = 171`, **now
measured directly** (see §5.1) rather than extrapolated — the earlier value of 134
came from assuming `dep` grows by 27 per step in `t`, and it does not.

| `d` | `B = dN` | `ell_sat` | unknowns | constraints | surplus | `+134` | **measured kernel** |
|---|---|---|---|---|---|---|---|
| 1 | 27 | 13 | 196 | 323 | −127 | 7 | **0** — model wrong here |
| 2 | 54 | 27 | 756 | 969 | −213 | −79 → 0 | **0** |
| 3 | 81 | 40 | 1,681 | 1,938 | −257 | −123 → 0 | **0** |
| 4 | 108 | 54 | 2,970 | 3,230 | −260 | −126 → 0 | **0** |
| 5 | 135 | 67 | 4,624 | 4,845 | −221 | −87 → 0 | **0** |
| 6 | 162 | 81 | 6,642 | 6,783 | −141 | **+30** | pending |
| 7 | 189 | 94 | 9,025 | 9,044 | **−19** | **152** | **152** |
| 8 | 216 | 108 | 11,772 | 11,628 | +144 | 315 | not measured |

Three things to read off it.

**The measured zeros at `d = 2..5` agree with the model.** All four are deeply
negative even after the `+134` correction, and all four measure 0. Capacity is *not*
the obstruction — `ell_sat >= 15` from `d = 2` onward, so every one of these curves
has room for 15 roots and still cannot be built.

**`d = 1` is where the model fails, and visibly.** It predicts a kernel of 7; the
measurement is 0. Two reasons, both known: `ell_sat(1) = 13 < 15`, so `Q` could not
carry 15 roots even with a kernel; and §5's `dep` is calibrated on `d >= 3`, where the
dependency is a constant — at `d = 1` the true value is larger and the row is
over-predicted. The `+dep` column should be read only for `d >= 3`.

**`d = 7` is confirmed: kernel 152, measured.** Pure counting would not give a
surplus until `d = 8`; the rank deficiency moves it to `d = 7`. `t = 15` is therefore
*not* a wall — it is another budget threshold, two steps up from `t = 14`'s.

Compare the shape against `t = 14`: there the deficit *shrinks* monotonically with `d`
(−109, −159, −149, −80, +49), while here it widens to −260 at `d = 4` before turning
around. The extra 18 points in `Y` cost `d(d+1)/2` each, and at small `d` that outruns
the monomials the larger budget provides. The turn comes only once the quadratic term
takes over.

That decode has now been run — see §5.1 for what it cost and what it broke.
Either outcome is worth more than the rest of the `t = 15` row, which is already five
measured zeros in agreement.

---

## 5. Why `d = 4` works although the surplus is still negative

Counting is **sufficient, not necessary**. The constraint matrix is rank-deficient:
some of its rows are linear combinations of the others and impose no new condition.

```
kernel = unknowns - rank
       = unknowns - (constraints - dependent_rows)
       = surplus + dependent_rows
```

Measured directly (`scratchpad/kernel_probe.py`), both at `t = 14`:

```
d=4:  matrix 3050 x 2970   rank 2943   =>  3050 - 2943 = 107 dependent rows
d=5:  matrix 4575 x 4624   rank 4468   =>  4575 - 4468 = 107 dependent rows
```

**Exactly 107 in both cases.** Adding 107 to the surplus column reproduces every
measured kernel dimension exactly, including the zeros at `d = 1, 2, 3`. So the real
threshold is not `surplus > 0` but **`surplus > -107`**, which falls between `d = 3`
(deficit 149, not covered) and `d = 4` (deficit 80, covered).

That `dep` is identical at `d = 4` and `d = 5` suggested it is a property of the point
set alone. **At `t = 15` that turns out to be false** -- see §5.1, where the same
quantity measures 171 at `d = 7` and 164 at `d = 6`. So `kernel = surplus + dep(t)` is
an approximation that happens to be exact at `t = 13` and `t = 14` and drifts
afterwards. Treat the constancy in `d` as a coincidence of small `t`, not a law.

### What is not explained

**Why 107.** It is identical at `d = 4` and `d = 5`, so it is a property of the point
set `Y` rather than of the multiplicity. At `t = 13` the same quantity is **80**, and
that value reproduces the measured kernels there too (51, 200, 429 at `d = 3, 4, 5`).
Between `t = 13` and `t = 14` it grows by 27 = `N`. No derivation is offered here;
`tN - n_P` (collisions, 73) and `n_P - N` (278) both fail to match.

### 5.1 `dep` measured at `t = 15`, and the rule it breaks

A single decode at `d = 7, t = 15, sigma = 0` was run to test the extrapolation
(`scratchpad/d7_kernel_dim.py`):

```
matrix        9,044 x 9,025 over GF(27)
expanded      27,132 x 27,075 over GF(3)      416s
kernel dim    152                             144s
```

Against a prediction of 115. So `dep(15) = 152 - (-19) = **171**`, not the 134 the
`+27 per t` rule gave. The sequence is

```
t = 13:  80
t = 14: 107    (+27)
t = 15: 171    (+64)
```

**`dep` is not linear in `t` — it accelerates.** Any projection in this note that
leans on `dep = 80 + 27(t-13)` is therefore wrong, and wrong in the *optimistic*
direction: the true `dep` is larger, so the `d` required at each `t` is **lower** than
§8 states. Three data points are not enough to fit a replacement, so none is offered.

A second decode, at `d = 6`, confirms that and adds a complication:

```
d=6 t=15:  matrix 6,783 x 6,642   kernel dim = 23   =>  dep(15) = 23 + 141 = 164
d=7 t=15:  matrix 9,044 x 9,025   kernel dim = 152  =>  dep(15) = 152 + 19 = 171
```

**So `t = 15` is reachable at `d = 6`, not `d = 7`** — one step lower than §8 predicted
— and the two estimates of `dep(15)` disagree, 164 against 171. At `t = 14` the same
quantity was identical (107) at two different `d`. It is not identical here, so `dep`
has a weak `d`-dependence that only shows up once `t` is large enough. The practical
consequence is that §8's `d` column should be read as an upper bound throughout.

Two further things this decode established:

**The GF(3) expansion had to be fixed to run it at all.** `kernel_vector_via_prime_field`
built the GF(3) matrix through `.tolist()`, which at this size is 735M Python ints,
about 20 GB. Handing Sage the 2-D numpy array instead keeps it in a 0.73 GB int8
buffer. All 13 cases in `tests/test_core_regression.py` pass unchanged under both
backends after the fix.

**NTL's root finder segfaults at this size.** The first attempt computed the kernel
fine and then crashed inside `_roots_univariate_polynomial` factoring a polynomial of
`y`-degree 94. So `ell_sat` growing linearly in `d` makes *factorisation*, not the
linear algebra, the next bottleneck — something to know before planning anything at
`d >= 7`.

One trap: at `d = 1` the apparent dependency is 109, not 107. That is an artefact —
with only 196 columns the rank is capped by the column count, so the true row
dependency is unobservable. The number is only meaningful once `rank < cols`, which
happens from `d = 4`.

---

## 6. Confirmation — recovery at `t = 14`

Mean recall (correct codewords returned / codewords bundled), decode counts in
parentheses. Produced by `streamlit_app/cliff_probe.py`, stored in
`streamlit_app/data/cliff_3_3_3.jsonl`.

| `d` | σ=0 | σ=1.0 | σ=1.2 | σ=1.4 | σ=1.6 |
|---|---|---|---|---|---|
| 1 | 0.00 (1) | 0.00 (20) | 0.00 (20) | 0.00 (20) | 0.00 (20) |
| 2 | 0.00 (1) | 0.00 (10) | 0.00 (10) | 0.00 (10) | 0.00 (10) |
| 3 | 0.00 (1) | — | 0.00 (5) | 0.00 (5) | — |
| **4** | **1.00** (1) | **1.00** (10) | **0.99** (10) | 0.74 (10) | 0.32 (10) |
| **5** | **1.00** (1) | — | **1.00** (3) | 0.93 (3) | — |
| **6** | **1.00** (1) | — | **1.00** (2) | — | — |

The threshold is sharp at `d = 4`, and it survives noise: 0.99 at `sigma = 1.2` over
ten independent draws. `d = 1, 2, 3` are hard zeros with `ell = -1` — no interpolant
at any rung — including at `sigma = 0`, confirming this is the bundling cliff and not
the noise cliff.

An independent check at `sigma = 0` rooted *every* kernel element rather than the one
`core.unbundle` happens to pick: 27 of 27 basis vectors plus 15 random kernel elements
at `d = 4`, and 12 of 12 plus 10 random at `d = 5`. All returned exactly the 14
bundled codewords, **zero spurious**. Correctness is not at risk here; only existence
was ever in question.

---

## 7. Where it stops

The ratio `constraints/unknowns` bottoms out at `n_P*(K-1)/N^2` and can never go below
it however large `d` gets. So multiplicity helps **if and only if**

```
n_P  <  N^2/(K-1)  =  729/2  =  364.5
```

`n_P` counts the 1s in `Y`, and `Y` has `N^2 = 729` cells, so the condition is simply:
**the support matrix must be less than half full.** At `t = 14` it is 42% full, which
is why `d = 4` gets through.

Measured `n_P` against that threshold, over random bundles:

| `t` | `n_P` (random bundles, min–max) | under 364.5 |
|---|---|---|
| 15 | 300–328 | 100% |
| 16 | 319–340 | 100% |
| 17 | 328–368 | 98% |
| 18 | 345–375 | 75% |
| 20 | 367–403 | 0% |

So the ceiling is **bundle-dependent, around `t = 18`–`19`** for a typical bundle,
rather than a single hard `t`.

`t = 15` is measured as a wall at every `d` tried (1 through 5), zero at every sigma
including 0. Extrapolating the dependent-row count (`80` at `t = 13`, `107` at
`t = 14`, `+27` per step) predicts `t = 15` first becomes reachable around `d = 7`.
**That extrapolation rests on two data points and no theory** and should be treated as
a conjecture, not a result.

---

## 8. How far can `d` be pushed before `t` stops moving?

`d` is unbounded in principle. `t` is not. Two separate things cap it, and they bite
in opposite ways: the first is a hard impossibility, the second is cost.

### The hard cap — `t = 17`

Section 7 gives the criterion: `d` helps at all only while `n_P < N^2/(K-1) = 364.5`.
Measured `n_P` for the family `core.message_list` generates:

| `t` | `n_P` | `Y` fill | capacity needs `d >=` | counting needs `d >=` | with `+dep(t)` |
|---|---|---|---|---|---|
| 13 | 285 | 39% | 1 | 4 | 3 |
| 14 | 305 | 42% | 2 | 5 | **4** |
| 15 | 323 | 44% | 2 | 8 | **6** (measured) |
| 16 | 342 | 47% | 2 | 15 | **13** |
| 17 | 357 | 49% | 2 | 45 | **43** |
| 18 | 375 | **51%** | 2 | **never** | **never** |
| 19 | 385 | 53% | 2 | never | never |
| 20 | 399 | 55% | 2 | never | never |

- *capacity* is `ceil(t(K-1)/N)`, the condition `ell_sat(d) >= t` from §2. It is
  almost never the binding constraint — `d = 2` suffices all the way to `t = 20`.
- *counting* is the smallest `d` with `surplus > 0`: guaranteed, no assumptions.
- *with `+dep(t)`* applies §5's rank-deficiency correction, extrapolating
  `dep = 80 + 27(t-13)`. **That extrapolation is now known to be wrong** — `dep(15)`
  measures 171, not 134 (§5.1). The true values are larger, so every entry in this
  column is an **over**-estimate of the `d` actually needed. `t = 15` is confirmed at
  `d = 7`, where the column predicted 7; `t = 16` and `t = 17` should be read as upper
  bounds only.

**At `t = 18` the grid passes half full and the column reads `never`** — not "very
large `d`", but no `d` at all, because the asymptotic ratio `n_P(K-1)/N^2` exceeds 1
and no amount of multiplicity can bring it back.

Two caveats. `n_P` is bundle-dependent (§0), so for a *typical* random bundle the
`never` row falls nearer `t = 19`; the 51% at `t = 18` here is one particular family.
And the `+dep` column is calibrated on `d >= 3`: at `d = 1, 2` the true dependency is
larger (90 and 102 at `t = 13` rather than 80), so the column is **conservative at low
`t`** — it predicts `d = 3` for `t = 13`, where `d = 1` in fact works. It was correct
at the one point it has been tested against, `t = 14` → `d = 4`.

### The practical cap — `t = 15`

Long before the hard cap, cost ends the argument. The measured ladder at `t = 13`
(one decode, this machine):

```
d=1: 2s   d=2: 8s   d=3: 57s   d=4: 154s   d=5: 356s   d=6: 530s
```

which fits `cost ~ d^3.1`. Past `u_max` the `ell` ladder must climb all the way to
`ell_sat` instead of stopping at `ell = t`, and that was measured at **6.2x** more
(`d = 5`: 356s at `t = 13` against 2204s at `t = 14`). Extrapolating:

| `d` | one decode at `t = 13` | one decode past `u_max` | interpolation matrix at `ell_sat` |
|---|---|---|---|
| 4 | 2 min | 15 min | 3,050 x 2,970 |
| 5 | 5 min | 31 min | 4,575 x 4,624 |
| 6 | 9 min | 55 min | 6,405 x 6,642 |
| **7** (for `t = 15`) | 14 min | **89 min** | 8,540 x 9,025 |
| **13** (for `t = 16`) | 1.6 hr | **10 hr** | 27,755 x 30,976 |
| **43** (for `t = 17`) | 3 days | **18 days** | 288,530 x 337,561 |

So the honest ceiling is:

- **`t = 14`** — done, at `d = 4`, 15 minutes a decode. Routine.
- **`t = 15`** — **done**: kernel non-empty at `d = 6` (dim 23) and `d = 7` (dim 152),
  §5.1. About 55 and 90 minutes respectively for the interpolation alone. Note that
  NTL's root finder segfaults at this `ell_sat`, so the kernel exists but extracting
  the codewords from it needs a different factorisation path.
- **`t = 16`** — needs `d = 13`, ~10 hours a decode on a 28k x 31k matrix over
  GF(27). Possible in principle, not on one machine.
- **`t = 17`** — needs `d = 43`. A 288k x 338k matrix and ~18 days per decode. The
  theory says yes; nothing else does.
- **`t = 18`** — impossible at any `d`, for this bundle.

`t = 15` is therefore where "raise `d`" stops being a strategy and becomes a thought
experiment. The gap between `t = 15` and the true ceiling at `t = 17`-`18` is real but
unreachable with dense linear algebra over an extension field; closing it would need
the interpolation itself to change (Koetter's iterative algorithm, `O(n^2 l)`, is the
candidate — see `RESEARCH_PLAN.md`).

---

## 9. Spiking noise

AWGN perturbs every coordinate a little. **Spiking** noise hits a few coordinates
hard. The natural parameters are `sigma` (the ordinary floor), `mu` (how many
coordinates are hit) and `nu` (how hard). This section asks whether the paper's
guarantee covers that, and what actually happens.

All of it is at `d = 1`, from `streamlit_app/spike_grid.py`, 618 decodes recorded in
`data/spikes_3_3_3.jsonl`.

### 9.1 The paper already states a spiking model

The hypervector is `N` blocks of `N`, one block per codeword position, and Stage A
decodes each block independently. **Proposition 2** (§V-A of histo.pdf):

> for non-negative `s, r` with `(r+1)s < N`, recovery succeeds whenever every `e_i`
> has `||e_i||_2 < sqrt(N)/2` **except at most `s`** indices, in which `e_i` lies in
> `E_r`.

That is a spiking model in the paper's own variables: **`s` is `mu`** (how many
coordinates are badly hit), **`r` is `nu`** (how badly), and `(r+1)s < N` is the
budget they share. Note the shape — a few severe hits cost the same as many mild
ones.

`E_r` is a union of open balls of radius `sqrt(N)/2` around the lattice points `Hv`
with `||v||_1 <= r`. Membership means "Stage A returns the right histogram displaced
by `v`, and `||v||_1 <= r`". `E_0` is exactly Corollary 2.

### 9.2 Proposition 2 does not cover spiking noise

Over 618 decodes, 578 had at least one block exceeding `sqrt(N)/2 = 2.598`:

```
spiked blocks:                              3,200
of those, actually inside E_r:                  2      (0.06%)
decodes where Prop 2's hypotheses hold:         2 of 578
```

This is structural, not bad luck. `H` has unit-magnitude entries, so every lattice
point `Hv` is spread across all 27 coordinates, while a spike is concentrated in one.
A spike can sit in the Voronoi cell of `Hv` — decoding to a known offset — and still
be outside `B(Hv, sqrt(N)/2)`, hence in `E_r` for no `r` at all.

The ball is conservative in a second way too: Lemma 1 gives the Voronoi cell as
`Hv + H(-1/2,1/2)^N + i*H*R^N`, and that last term is an **unbounded** real
`N`-dimensional subspace along which noise does nothing whatsoever. Measured: noise
of norm **25,166** in the `i*H*R^N` direction — nearly 10,000x the ball radius —
leaves the histogram exactly right and all codewords recovered.

**So the guarantee is silent here, and the decoder works anyway.**

### 9.3 What actually governs damage

Stage A computes `round((1/N)Re(H^dag s_j))` and then `Y = (hist > 0)`. So the only
thing that matters is whether a perturbation drives a histogram entry **across zero**:

| perturbation | effect on `Y` | effect on recall |
|---|---|---|
| raises an entry `0 -> positive` | **adds** a point | none — every codeword still agrees everywhere |
| lowers an entry `positive -> 0` | **removes** a point | that column's list loses a symbol; codewords using it die |
| moves an entry without crossing 0 | none | none |

That asymmetry is the whole story. Spiking noise is dangerous only in so far as it
*erases* support, and a large fraction of it does not.

It also explains why `s_pred` — the count of blocks past Corollary 2 — so badly
overstates the harm:

| blocks past Cor. 2 | decodes | histograms actually wrong | mean recall |
|---|---|---|---|
| 0 | 40 | 0.0 | 1.00 |
| 1–2 | 289 | 0.6 | 0.90 |
| 3–6 | 132 | 1.9 | 0.64 |
| 7–30 | 157 | 6.1 | 0.47 |

Seven to thirty blocks outside the guarantee produce, on average, **six** wrong
histograms — and still return just under half the codewords.

### 9.4 The `(mu, nu)` grid

Mean recall, `t = 5`, `width = 1`, no background noise, 5 seeds per cell:

| `mu` \ `nu` | 2 | 5 | 10 | 14 | 20 | 40 |
|---|---|---|---|---|---|---|
| 1 | 1.00 | 1.00 | 1.00 | 0.92 | 0.64 | 0.64 |
| 2 | 1.00 | 1.00 | 1.00 | 0.80 | 0.52 | 0.36 |
| 3 | 1.00 | 1.00 | 1.00 | 0.96 | 0.32 | 0.16 |
| 5 | 1.00 | 1.00 | 0.96 | 0.48 | 0.08 | 0.08 |
| 8 | 1.00 | 1.00 | 0.76 | 0.52 | 0.16 | 0.00 |
| 13 | 1.00 | 1.00 | 0.88 | 0.12 | 0.00 | 0.00 |
| 20 | 1.00 | 1.00 | 0.64 | 0.24 | 0.00 | 0.00 |

`t = 10` reads better at light damage (`mu=1, nu=20`: 0.80 against 0.64) and the same
or worse at heavy damage (`mu=8, nu=20`: 0.14 against 0.16). The two grids are not
measuring different noise -- the **histograms damaged are identical** at both `t`, for
every `(mu, nu)`, because the noise does not know what `t` is. What differs is the
consequence, and two effects pull opposite ways:

- **Normalisation.** `recall = (t - lost)/t`. At `mu=1, nu=20` the absolute loss is
  1.8 codewords at `t=5` and 2.0 at `t=10` -- nearly the same, divided by a larger
  `t`. This dominates when damage is light.
- **Shared cells cost more.** Erasing `hist[a][j]` kills every codeword using symbol
  `a` at position `j`, and that count grows with `t`. At `mu=3, nu=20` the loss is 3.4
  against 6.4 -- roughly doubling with `t`. This dominates when damage is heavy, and
  the two curves converge (`mu=8, nu=20`: 0.16 vs 0.14; `mu=20`: 0.00 at both).

The `t`-independent view is **histograms damaged**, which is identical across the two
and is the cleanest evidence that the noise is doing the same thing either way.

**Everything to the left of `nu = 10` is untouched, at every `mu`.** The threshold is
predictable: a spike of magnitude `A` in one coordinate moves each of that block's 27
histogram entries by at most `A/N`, and rounding only flips at

```
A/N > 1/2      =>      A > N/2 = 13.5
```

Corollary 2 gives up at `sqrt(N)/2 = 2.598`, so the gap is `sqrt(N)`.

### The sqrt(N) gap, verified as a scaling law

That ratio is a *prediction about every N*, not an observation about 27, so it was
tested across seven codes by bisecting on the spike magnitude until Stage A's
histogram first changes -- 40 random phases each, Stage A only, no decoding:

| `p,m,K` | `N` | Cor. 2 = `sqrt(N)/2` | onset, median | onset, **min** | `N/2` |
|---|---|---|---|---|---|
| 2,2,2 | 4 | 1.00 | 2.60 | **2.0** | 2.0 |
| 2,3,2 | 8 | 1.41 | 5.21 | **4.0** | 4.0 |
| 3,2,2 | 9 | 1.50 | 4.76 | **4.5** | 4.5 |
| 2,4,2 | 16 | 2.00 | 10.42 | **8.0** | 8.0 |
| 5,2,2 | 25 | 2.50 | 12.69 | **12.5** | 12.5 |
| 3,3,3 | 27 | 2.60 | 14.14 | **13.5** | 13.5 |
| 2,5,2 | 32 | 2.83 | 20.84 | **16.0** | 16.0 |

**The minimum onset equals `N/2` exactly in all seven**, so the worst-case ratio to
Corollary 2 is exactly `sqrt(N)`. The constant is 1, not "about 1".

The spread above the minimum is the spike's **phase**. A spike contributes
`Re(A e^{i theta} conj(H_jk))/N` to histogram entry `k`, so its damage is
`A |cos(theta - arg H_jk)| / N`; the worst case over `k` gives `A/N` and hence the
onset `N/2`, while unluckier phases need more magnitude. That also explains why the
`p = 2` rows spread much wider: there `H` is **real**, every entry shares a phase, and
a spike at `theta ~ pi/2` does nothing at all (the 128 at `N = 32` is the bisection
cap, not a measurement). For `p >= 3` the columns of `H` span several phases, so some
`k` is always nearly aligned -- `p=5, N=25` ranges only over `[12.5, 13.1]`.

So the honest statement is: **damage begins at `N/2` in the worst case over phase, and
Corollary 2 abandons the block at `sqrt(N)/2` -- a gap of exactly `sqrt(N)`, verified
from `N = 4` to `N = 32`.**

### 9.5 `(r+1)s < N` as a predictor

Using the measured `r` and `s`:

| | exact recovery | failure |
|---|---|---|
| `(r+1)s < N` holds | **331** | 64 |
| `(r+1)s < N` fails | 29 | 194 |

84% of decodes satisfying the condition recover exactly, against 13% of those that do
not. So it carries real information — but it is **not** behaving as a guarantee here,
which is expected: hypothesis (b) of Proposition 2 (membership in `E_r`) essentially
never holds for spikes, so the condition is being applied outside its domain. The 29
decodes that recover exactly while violating it are the usual direction of error:
reality more generous than the bound.

### 9.6 A claim I had to withdraw: spike *width*

From three coarse probes I earlier concluded that spreading a spike over `w` adjacent
coordinates makes it `sqrt(w)` times more damaging, so that thin spikes are safest.
**The swept data does not support that.** Mean recall at `t = 5`, one spike, the same
block energy in every row, first `nu` at which recall drops below 1.0:

| width | 1 | 2 | 3 | 5 | 9 | 27 |
|---|---|---|---|---|---|---|
| measured onset | 14 | 16 | 12 | 10 | 16 | **never** |
| `N/(2*sqrt(w))` predicted | 13.5 | 9.5 | 7.8 | 6.0 | 4.5 | 2.6 |

There is no monotone trend, the scatter is within what 3 seeds can resolve, and
`width = 27` never fails at all up to `nu = 30`. Only `w = 1` matches its prediction,
which is the case the `sqrt(w)` argument reduces to.

The `w = 27` row is not noise, though — it is structural, and it illustrates §9.3.
With `width = N` every entry of the block receives the same value, so

```
e_i = c * (1, ..., 1) = c * H[:,0]
```

because the first column of `H` is all ones. That is a **lattice direction**.
Verified: `(1/N)Re(H^dag e_i)` is non-zero in exactly one coordinate, the histogram
changes in exactly one entry, and at `nu = 30` that entry moved by `-1` without
crossing zero — so `Y` lost nothing and recall stayed at 1.00.

So the honest statement is: **what matters is not how wide a spike is but whether it
drives a support cell across zero**, and a spike aligned with a lattice direction
concentrates its damage into a single histogram entry where it usually does not.

### 9.7 Spikes on a Gaussian floor

`sigma = 1.0` underneath, `t = 5`:

| `mu` | `nu = 5` | `nu = 14` | `nu = 20` |
|---|---|---|---|
| 1 | 1.00 | 0.93 | 0.80 |
| 3 | 1.00 | 0.73 | 0.47 |
| 8 | 1.00 | 0.40 | 0.00 |

At `sigma = 1.0` alone, recall is 1.00. The floor does not change the `nu = 5` column
at all, and it compounds roughly multiplicatively further right. Nothing new in kind:
the background noise adds its own erasures to the spikes'.

### 9.8 On the correlation matrix

The original plan was to model spikes through a correlation matrix `D`. That will not
work. Spikes at random positions give covariance `(sigma^2 + p*nu^2) * I` — diagonal
and isotropic, indistinguishable from plain Gaussian noise of matched variance.
Second-order statistics cannot see sparsity; distinguishing them needs the fourth
moment, or an explicit mixture.

This is precisely why Proposition 2 is stated **per block** rather than per
coordinate. The block structure, not the covariance, is where the information lives.

### 9.9 Summary

- The paper's Proposition 2 **is** a spiking model, with `s = mu` and `r = nu`.
- It almost never applies: **3,198 of 3,200** spiked blocks fall outside `E_r`.
- The real threshold for a single spike is `N/2 = 13.5`, against Corollary 2's
  `sqrt(N)/2 = 2.6` — conservative by `sqrt(N)`.
- Damage is governed by whether a histogram entry crosses zero, not by block norm.
  Additions to `Y` are free; erasures are fatal.
- Spike **width** has no clean effect, contrary to what single probes suggested.
- A correlation matrix cannot represent this model.

---

## Summary

| | why `d` helps |
|---|---|
| **capacity** | `deg_y Q <= dN/(K-1)`; at `d = 1` that is 13, which *is* `u_max` |
| **existence** | `constraints/unknowns ∝ (d+1)/d`, so the system loosens as `d` grows |
| **not agreement** | a codeword must still match `Y` at all `N` positions, at every `d` |

`u_max = 13` is where `d = 1` stops, not where the code stops. `t = 14` is reachable
from `d = 4`. The true ceiling is set by `Y` filling past half, near `t = 18`.
