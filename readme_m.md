# What changed to support `m > 1`

[`encode.ipynb`](encode.ipynb) implements the encoder of Deng and Raviv, *Efficient
Vector Symbolic Architectures from Histogram Recovery* (arXiv:2511.01838, local copy
`histo.pdf`) for the special case `m = 1`: the Reed–Solomon alphabet is the prime field
`F_p` and all arithmetic is plain integer arithmetic mod `p`.

[`encode_m.ipynb`](encode_m.ipynb) does the general case over `F_{p^m}`. Set `m = 1` and
it reproduces `encode.ipynb` exactly — asserted in the notebook and again in
[`tests/test_encode_m.py`](tests/test_encode_m.py).

```python
# encode.ipynb
histo_encode([2, 1], 5)         # -> 25 entries

# encode_m.ipynb
histo_encode([2, 1], 5, 1)      # -> 25 entries, identical values
histo_encode([2, 1], 2, 3)      # -> 64 entries, m = 3
```

## The one-line summary

`N = p` (a prime) becomes `N = p^m` (a prime power). Everything else follows from that.

## Quantities

| Quantity | `encode.ipynb` (`m = 1`) | `encode_m.ipynb` (general `m`) |
|---|---|---|
| RS alphabet | `F_p`, integers `0..p-1` | `F_{p^m}`, integers `0..p^m-1` |
| Block length `N` | `p` — must be **prime** | `p^m` — any **prime power** |
| Hypervector dimension | `p²` | `p^{2m}` = `N²` |
| Code size | `p^K` | `p^{mK}` = `N^K` |
| Field arithmetic | `% p` | `ADD` / `MUL` lookup tables |
| `phi` range | `p`-th roots of unity | unchanged — still `p`-th roots of unity |

## Signature changes

| `encode.ipynb` | `encode_m.ipynb` |
|---|---|
| `phi(a, p)` | `phi(a, p)` — identical |
| `reed_solomon_encode(message, p)` | `reed_solomon_encode(message, p, m, tables=None)` |
| `hadamard_matrix(p)` | `hadamard_matrix(p, m, tables=None)` |
| `histo_encode(message, p)` | `histo_encode(message, p, m, tables=None)` |
| `l_max(p, K)` | `l_max(N, K)` — argument renamed, since `N` is no longer `p` |
| `next_prime(n)` | `next_prime_power(n, max_p=None)` |
| `min_p(K, l)` | `min_N(K, l)` |
| `choose_params(l, n_items, K_max)` | `choose_params(l, n_items, K_max, max_p=None)` |
| — | `gf_tables`, `poly_mul`, `poly_mod`, `is_irreducible`, `find_irreducible` |
| — | `hadamard_exponents`, `d_min`, `u_max`, `prime_powers`, `dim_for` |

`tables` is an optional pre-built `gf_tables(p, m)` result. Default behaviour rebuilds per
call, exactly as `encode.ipynb` rebuilds its Hadamard matrix on every `histo_encode`; the
sweeps pass it in to avoid ~14 000 redundant table builds.

## What did *not* change

Worth stating explicitly, because it is more than you might expect.

**`phi` is untouched.** Its domain is `F_p`, not `F_{p^m}` — the inner Hadamard code emits
symbols in `F_p` whatever `m` is. So codeword entries are always `p`-th roots of unity.

**Every figure of merit is a function of `N` alone**, so the formulas carry over verbatim
under the substitution `N = p^m`:

```
d_min = N - K + 1        mu = (K-1)/N        l_max = (2N - d_min) / (2(N - d_min))
```

The notebook's third graph measures this: `p=2, m=4` (`N=16`) and `p=17, m=1` (`N=17`)
produce separation curves agreeing to within ~0.01 at every `l`, despite alphabets of 2
and 17.

**`histo_encode`'s body is structurally identical** — encode, index the Hadamard matrix by
the codeword, transpose, flatten.

## Function by function

### New prerequisite: the field

`m = 1` let us skip this entirely. `F_{p^m}` is built as `F_p[x]/(f)` for a monic
irreducible `f` of degree `m`; an element is a plain integer `0..p^m-1` whose base-`p`
digits are its polynomial coefficients.

```python
vec, ADD, MUL = gf_tables(p, m)
```

- `vec[a]` — the base-`p` digit vector of `a`. This is the paper's "representing each
  `F_{p^m}` entry as a vector in `F_p^m`" (§III), and the map is an isomorphism of
  *additive* groups, which is the only structure `f_Had` and `phi` depend on.
- `ADD`, `MUL` — the `q × q` field operation tables.

At `m = 1` these degenerate to `vec[a] == [a]`, `ADD == (a+b) % p`, `MUL == a*b % p`,
which is exactly why `encode.ipynb` never needed them.

Supporting helpers: `poly_mul`, `poly_mod` (remainder mod a *monic* polynomial, so no
coefficient inverse is needed), `is_irreducible`, `find_irreducible`.

### Reed–Solomon: the biggest change

Three substitutions. Evaluation points become all `N = p^m` field elements; the
Vandermonde powers `beta^i` are built with `MUL`; the accumulation uses `ADD`. None of it
can be expressed with `%`.

```python
# encode.ipynb -- integer Vandermonde
points = np.arange(p)
powers = points[:, None] ** np.arange(len(message))
return (powers @ message) % p

# encode_m.ipynb -- field Vandermonde
points = np.arange(N)
c = np.zeros(N, dtype=int)
power = np.ones(N, dtype=int)          # beta^0; the field one is the integer 1
for a in message:
    c = ADD[c, MUL[int(a), power]]
    power = MUL[power, points]         # beta^i -> beta^(i+1)
```

### Hadamard: same value, different reason

```python
# encode.ipynb
inner = np.outer(idx, idx) % p

# encode_m.ipynb
(vec @ vec.T) % p
```

These agree at `m = 1` and diverge for every `m > 1`. See the traps below.

### Parameter selection: primes become prime powers

`N` is no longer confined to primes, which is a strictly larger menu, so the required
dimension can only fall. Prime-power factorisation is unique, so a chosen `N` fixes
`(p, m)` unambiguously — no search over `p` and `m` is needed.

`prime_powers`, `next_prime_power` and `dim_for` replace `next_prime` and the primes-only
search. `choose_params` gains **`max_p`**, a cap on the alphabet that forces a larger `m`:
`max_p=2` puts `phi`'s range at `{+1, -1}`, so hypervectors are real ±1 vectors, at the
cost of rounding `N` up to a power of two.

The algorithm is otherwise unchanged — search `K`, derive `N`, factor `N`:

```
for K = 1 .. K_max:
    N <- smallest prime power >= max(2, K, (K-1)(2l-1))     # RS validity + bundling
    while N^K < n_items:                                    # vocabulary
        N <- smallest prime power >= N+1
return the (N², -K) minimiser, with (p, m) read off from N
```

## Two things that silently break if you generalise naively

Both are asserted in the notebook rather than merely warned about, because both produce a
plausible-looking wrong answer rather than an error.

### 1. The Hadamard exponent is a dot product, not field multiplication

The paper defines `f_Had(a) = (a b^T)_b`, and that `a b^T` is the **`F_p`-bilinear form on
the coefficient vectors** — a dot product mod `p` — *not* multiplication in `F_{p^m}`.

At `m = 1` the two coincide, both reducing to `a·b mod p`. That is why
`np.outer(idx, idx) % p` is correct in `encode.ipynb`, and why reaching for `MUL` when
generalising looks reasonable and stays invisible until `m > 1`.

```python
assert (hadamard_exponents(p, 1) == np.outer(idx, idx) % p).all()   # m = 1: agree
assert not (hadamard_exponents(p, m) == MUL).all()                  # m > 1: differ
```

### 2. Binding addition is `ADD`, not `(a + b) % p**m`

Binding is still the point-wise product and still corresponds to addition in the message
space — but that addition is `F_{p^m}` addition, coefficient-wise mod `p`, which at
`p = 2` is XOR. `(a + b) % p**m` returns a perfectly valid message, so the failure is
silent.

Concretely at `p=2, m=3`: `ADD[3, 1] = 2`, while `(3 + 1) % 8 = 4`. The notebook reports
that the naive version breaks binding on **22 of 30** random pairs.

## Not ported

`encode.ipynb`'s object/scene demo — `name_to_message`, `atom`, `encode_object`, the
codebook and the similarity read-back — has no counterpart in `encode_m.ipynb`. Porting it
needs two changes: base-`p^m` digits instead of base-`p` in `name_to_message`, and `ADD`
instead of `%` wherever bindings are combined.

Note also that `n_items` for a bound representation must cover the number of distinct
**objects**, not the number of attribute *values* — objects are products of atoms, and it
is objects that get compared against the bundle.

## How the equivalence is verified

Three asserts inside `encode_m.ipynb`:

1. `gf_tables(p, 1)` equals plain mod-`p` arithmetic.
2. `hadamard_exponents(p, 1)` equals `np.outer(idx, idx) % p`.
3. `histo_encode(u, p, 1)` reproduces `encode.ipynb`'s `histo_encode(u, p)` entry for entry.

And in the test suite, which execs both notebooks' own cells so the tests cannot drift
from the code they check:

- `test_m_equals_1_reproduces_encode_notebook_exactly` — `histo_encode(u, p, 1)` and
  `reed_solomon_encode(u, p, 1)` against `encode.ipynb`'s versions, for five `(p, K)`
  pairs.
- `test_m_equals_1_collapses_to_plain_mod_p_arithmetic`
- `test_hadamard_uses_the_dot_product_not_MUL`
- `test_allowing_m_above_1_never_costs_dimension` — the general search is never worse
  than `encode.ipynb`'s primes-only `min_p`, and is strictly better somewhere.
- `test_l_max_matches_between_the_two_notebooks`

```
$ .venv/bin/python -m pytest
352 passed
```

## Cost

`gf_tables` builds `q × q` arrays with a Python double loop for `MUL`, so the practical
ceiling is `q = p^m` in the low thousands. `GF(2, 10)` (`q = 1024`) is fine; `GF(3, 10)`
(`q = 59049`) would want a 3.5-billion-entry table. Well beyond anything the notebook
explores, but worth knowing before reaching for large `m`.

## Why bother with `m > 1`

Dimension is always `N²`, and `N = p^m`, so choosing `m` is choosing how to **factor** your
target `N`:

- `m = 1` forces `p = N`, meaning `N` distinct complex phase values per coordinate. Primes
  are dense, so you hit your target `N` tightly.
- `p = 2` with large `m` gives the same `N` with only two phase values — real ±1
  hypervectors, the conventional VSA setting of the paper's footnote 1 — but `N` must round
  up to a power of two, and squaring makes that expensive. At `l = 20`: `N = 41`
  (dim 1681) unconstrained versus `N = 64` (dim 4096) forced binary.

There is also a small free lunch: prime powers are denser than primes low down
(2, 3, **4**, 5, 7, **8**, **9**, 11, 13, **16**), so allowing `m > 1` sometimes reduces
the dimension outright, with no downside.

It only shows up when the required `N` lands just above a prime, though — which in
practice means when the *vocabulary* constraint is the binding one. At `l = 2` with no
vocabulary requirement, both searches return `N = 3` and there is nothing to gain. Add
`n_items = 10` and bundling wants `N ≥ 3` while naming wants `N ≥ 4`: prime powers take
`N = 4 = 2²` (dim 16) where primes are forced up to `N = 5` (dim 25).

The test `test_allowing_m_above_1_never_costs_dimension` asserts the general search is
never worse than `encode.ipynb`'s primes-only `min_p`, and is strictly better somewhere.
