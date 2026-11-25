# `streamlit_app` — histogram_recovery.ipynb with the parameters on sliders

The notebook fixes `p = 3, m = 3, K = 3` and walks through encode → bundle → decode
once. This is the same four parts with `(p, m, K, d)` moved into the sidebar, so you
can drag the parameters and watch the cliff move.

## Running it

```bash
uv sync --group app --group sage
uv run --group app --group sage streamlit run streamlit_app/app.py
```

The `sage` group is ~200 MB and supplies the Guruswami–Sudan decoder; the `app` group
is streamlit + pandas.

## The tabs

| Tab | What it shows |
|---|---|
| **1 · Encode** | eq. (1) one layer at a time — digits, the RS evaluation, the Hadamard exponent grid, φ, and the hypervector. Move the `u[i]` sliders and the whole printout updates. |
| **2 · Bundle** | the fixed message list, and what addition does to `|entry|` and the norm as `t` grows. |
| **3 · Decode** | the notebook's **by hand** walkthrough at any `t` — every step printed, histogram included. |
| **4 · Noise** | the same walkthrough with noise, up to and past the level §IV-A permits. See below. |
| **5 · Gaussian sweep** | the physical noise model, plus the precomputed sweep over every prime power `q <= 100`. See below. |
| **6 · In versus out** | the whole sweep `t = 1 …`, plotted. Contains the "can a bigger ℓ rescue it?" probe. |
| **7 · Repeats** | §IV-B3 — bundle the same codeword several times and peel the multiset apart round by round. |
| **Bounds** | `d_min`, the unique radius, Johnson, `μ`, `l_max`, `u_max`, and why the wall lands exactly at `u_max`. |

Output is formatted as the notebook formats it: the same aligned monospace tables, the
same `show_histogram`, the same `f(x) = ... -> message [...]` lines. `report.py` holds
those formatters. Charts appear only where the notebook has one.

The Encode tab shows the codeword **three ways**, because the differences between them
are the whole algorithm:

| | each cell holds | what it is for |
|---|---|---|
| **Hadamard** grid (eq. 1) | a phase exponent in `0..p-1` | this is what gets bundled |
| **Kautz-Singleton** `W(c)` | `0` or `1` — *did* this symbol occur | what Algorithm 2 decodes |
| **histogram** `h` | a count — *how many* carried it | what Stage A hands back |

For a single codeword the last two coincide — Stage A run straight back on one
hypervector returns exactly `W(c)`, which the tab asserts. That is the cleanest way to
see what bundling does: columns start summing to `t`, a cell holding 2 marks a
collision, and throwing the counts away gives `Y`, the OR of `W(c_1)..W(c_t)`. The
Decode tab prints `h` and `Y` in the same layout.

## The noise tab

The notebook is noiseless. The paper is not — §IV-A writes block `j` of the received
vector as `s_j = H h_j' + e_j'`, and **Corollary 2** says the rounding still returns
`h_j` whenever `||e_j||_2 < sqrt(N)/2`. The tab sets `alpha` = the multiple of that
permitted level, so `alpha < 1` is inside the guarantee and `alpha >= 1` is outside it.

What makes it worth looking at is that the bound is about the **2-norm** while Lemma 1
only constrains the **real part, coordinate-wise** — so how far past `alpha = 1` you
get depends entirely on the direction of the noise. Measured at `p=3, m=3, K=3, t=3`:

| direction | `e_j` | Stage A exact to | recovery exact to |
|---|---|---|---|
| **real** | `H a'`, one signed coordinate | `alpha = 0.99` | `alpha = 0.99` |
| **isotropic** | complex Gaussian | `alpha = 2` | `alpha = 3` |
| **imaginary** | `i H b'` | never breaks | never breaks |

`real` is the direction Lemma 1 actually constrains: `||a||_inf = ||a||_2 = alpha/2`
crosses `0.5` exactly when `alpha` hits 1, so **Corollary 2 is tight there** — perfect
at 0.99, wrong at 1.00. `imaginary` never breaks at any `alpha`, because Lemma 1's
Voronoi cell is `H h' + H(-0.5,0.5)^N + i H R^N`, unbounded in that direction. That is
the paper's own remark that decoding "succeeds even for some unbounded error vectors",
and it is why the norm bound is sufficient but nowhere near necessary.

A collapsed expander on that tab prints the matrices themselves — each bundled
codeword's Hadamard exponent grid (**before φ**, integers in `0..p-1`), the same grids
**after φ** (every entry on the unit circle), their sum, the noise `e`, and `S + e`,
which is what Stage A actually receives. Rows are blocks, matching `s_j = H h_j' + e_j'`.
There is deliberately no "before φ" for the bundle: a sum of roots of unity is not a
root of unity, which is precisely why Stage A has to lattice-decode rather than read a
table off.

### Both cliffs at once

The α ladder fixes `t` and varies α; the **In versus out** tab fixes α = 0 and varies
`t`. They are two slices of one picture, and the **Both cliffs at once** section builds
the whole grid — `t` up the side, α across the bottom, one panel per noise direction.
The rendered version is [`figures/noise_grid.png`](../figures/noise_grid.png).

What it shows, at `p=3, m=3, K=3` (cells recovered exactly, out of each region):

| direction | `t ≤ u_max`, `α < 1` | `t ≤ u_max`, `α ≥ 1` | `t > u_max` |
|---|---|---|---|
| **real** | **39/39** | 5/65 | 0/16 |
| **isotropic** | **39/39** | 43/65 | 0/16 |
| **imaginary** | **39/39** | 65/65 | 0/16 |

Three things fall out of it:

- **Where both papers' guarantees hold, recovery is perfect** — 39/39 in every
  direction. Nothing in the left-hand block ever fails.
- **The two limits are independent.** Past `u_max` nothing comes back at *any* noise
  level, including α = 0: the bundling cliff is absolute and noise is irrelevant there.
  That is the horizontal boundary. The vertical one is α = 1 and only bites below it.
- **Past α = 1 the direction is everything** — 5/65 for real against 65/65 for
  imaginary. The largest α that still recovers every `t ≤ u_max` is 0.9 for real, 2.0
  for isotropic, and ≥ 5 for imaginary.

The `real` panel also shows partial failure rather than collapse: past α = 1 roughly
half the bundled codewords still come back for most `t`, and at `t = 1` it never breaks
at all — one codeword leaves so much redundancy that a badly corrupted histogram still
decodes.

Note the two stages fail separately: at `alpha = 3` isotropic, Stage A is wrong in 12
of 27 columns yet recovery is still exact — Stage B's list decoding has its own slack
and absorbs the damage.

## Why there are three files and not one

**`core.py`** is the notebook's logic as plain functions with `(p, m, K)` lifted into
arguments. It does not re-implement the encoder — it reads `encode_m.ipynb` at import
time and pulls the functions out of the resulting namespace, the same trick
`tests/conftest.py` uses, so the app cannot drift from the notebook it mirrors.

**`worker.py`** is a subprocess, and it exists for one reason: **Sage cannot be called
from Streamlit's script thread.** Streamlit runs the app on a `ScriptRunner` worker
thread, and Sage's PARI/NTL backends are not thread-safe — the Guruswami–Sudan root
finder segfaults the entire server, taking the session with it. A larger thread stack
does not help, and pre-importing Sage on the main thread makes it worse (it activates
cysignals' handlers, which then misfire on the worker). So everything that touches
Sage is pushed out to a fresh process that talks JSON Lines over stdout.

**`app.py`** keeps all the numpy work in-process — encoding, bundling, and the Stage A
histogram are pure linear algebra — and shells out to the worker only for Stage B.
That is why dragging the sliders stays immediate while decoding shows a spinner.

## The Gaussian noise tab

Everything else in the app is noiseless. This tab adds

    e ~ CN(0, sigma^2 I)      added to the bundle before Stage A

to one fixed code (`p=3, m=3, K=3, N=27`) — white noise, covariance a scalar times the
identity, every entry independent. Note `sigma^2` not `sigma` (it is a variance) and
`CN` not `N` (each entry's real and imaginary parts carry `sigma^2/2` each). `sigma` is
in units of the signal, since a codeword entry has `|entry| = 1`.

Sliders for `sigma` and `t`, the histogram diffed against the noiseless one, the
before/after-phi matrices in an expander, and a decode button — the same shape as the
by-hand walkthrough in the Decode tab.

### The grid

`noise_trials.py` builds it:

```bash
uv run --group sage --group app python streamlit_app/noise_trials.py
```

1000 independent noise draws per `(t, sigma)` cell, 17 values of `t` x 8 of `sigma`.
The codewords are fixed and only the noise is redrawn, so the result isolates the
noise rather than mixing it with a change of code. The tab has sliders for the `t` and
`sigma` ranges and a toggle between Stage A and full recovery.

Two things make 136,000 decodes feasible. One worker per core. And an `ell` bracket:
`unbundle` walks `ell` upward from 2 and succeeds at `ell = t`, so the first `t-2`
interpolations are wasted work — passing a bracket around `t` is 5-6x faster and gives
identical results, which is checked rather than assumed.

## The parameter space## The parameter space

Every slider's range is derived from the ones above it, so **an unusable combination is
unreachable** — there is no "too big" error to run into. `m`'s ceiling comes from `p`
(the hypervector is `p^2m` complex numbers, capped at 500,000), and `K`'s comes from
`N = p^m` — `K` is capped at `N - 1`, because `K = N` gives a Reed–Solomon code with
`d_min = 1`, which is no code at all and which Sage refuses to decode.

| `p` | `m` | largest `N` | largest `K` |
|---|---|---|---|
| 2 | 2–4 | 16 | 8 |
| 3 | 1–4 | 81 | 8 |
| 5 | 1–4 | 625 | 8 |
| 7 | 1–3 | 343 | 8 |
| 11 | 1–2 | 121 | 8 |
| 13 | 1–2 | 169 | 8 |

`p = 2, m = 1` is excluded: `N = 2` leaves no room for a `K` that is both ≥ 2 and ≤ N-1.

All 51 reachable `(p, m, K)` combinations render, and every one small enough to decode
quickly recovers its bundle exactly, with the cliff landing on `u_max` each time.

## Cost

Encoding is free everywhere — even `N = 625`, `dim = 390,625` builds in under 3 seconds
and encodes a hypervector in 10 ms. Bundling streams, so memory stays at one
hypervector however many you add.

**Decoding is the expensive part, and it is driven by `t`, not by `N`.** Measured:

| | `t = 3` | `t = 8` |
|---|---|---|
| `N = 27` | 1.6 s | — |
| `N = 49` | 1.8 s | 5.0 s |
| `N = 81` | 2.3 s | 15.8 s |
| `N = 121` | 3.8 s | — |

Since `u_max = (N-1)/(K-1)` grows with `N`, a *full* sweep at large `N` is hundreds of
decodes with the late ones dominating. The sweep therefore has its own ceiling slider
and prints an estimate before you commit.

A failed `t` is slower than a successful one — `unbundle` walks `ell` upward and only
gives up at `ell_max`, so every attempt past the wall costs the full ladder. Lower
**ell_max** in the sidebar to fail faster.

Sweep results are cached per parameter set in session state, so returning to a
combination you have already run is instant.
