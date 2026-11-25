"""Check every number in GURUSWAMI_SUDAN.md.

Pure Python, no Sage: the example is over F_5 with n = 5, so brute force is instant
and the arithmetic in the write-up can be followed line by line against this.

    uv run python verify_gs_example.py
"""

import math
from itertools import product

p, k = 5, 2
xs = [0, 1, 2, 3, 4]


def ev(coeffs, x):
    """Evaluate a polynomial given low-degree-first coefficients, over F_p."""
    return sum(c * pow(x, i, p) for i, c in enumerate(coeffs)) % p


def codeword(coeffs):
    return [ev(coeffs, x) for x in xs]


# --- section 1: the radii ---------------------------------------------------------
n = len(xs)
d_min = n - k + 1
unique_radius = (d_min - 1) // 2
johnson = n - math.sqrt(n * (n - d_min))
assert (n, k, d_min, unique_radius) == (5, 2, 4, 1)
assert abs(johnson - 2.7639) < 1e-4

# --- section 2: send and corrupt --------------------------------------------------
sent = codeword([1, 2])                       # f(x) = 1 + 2x
received = [1, 4, 0, 1, 4]                    # errors at positions 1 and 3
assert sent == [1, 3, 0, 2, 4]
errors = [j for j in range(n) if sent[j] != received[j]]
assert errors == [1, 3]
assert len(errors) > unique_radius            # genuinely past unique decoding
assert len(errors) < johnson                  # but inside Johnson

# --- section 3: the interpolation system ------------------------------------------
# Q = a + bx + cx^2 + dy + exy + hy^2, the (1,1)-weighted monomials of degree <= 2
monomials = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]
rows = [[pow(x, i, p) * pow(y, j, p) % p for i, j in monomials]
        for x, y in zip(xs, received)]
assert len(monomials) == 6 and len(rows) == 5      # 6 unknowns, 5 constraints

kernel = [v for v in product(range(p), repeat=6)
          if any(v) and all(sum(a * b for a, b in zip(r, v)) % p == 0 for r in rows)]
assert len(kernel) == p - 1                        # one curve, up to scaling
Q_coeffs = next(v for v in kernel if v[5] == 1)    # normalise h = 1 (monic in y)
assert Q_coeffs == (3, 2, 2, 1, 2, 1), Q_coeffs    # the write-up's Q


def Q(x, y):
    a, b, c, d, e, h = Q_coeffs
    return (a + b * x + c * x * x + d * y + e * x * y + h * y * y) % p


assert all(Q(x, y) == 0 for x, y in zip(xs, received))   # vanishes at all five

# --- section 4/5: factor, and the list --------------------------------------------
roots = [list(f) for f in product(range(p), repeat=k)
         if all(Q(x, ev(list(f), x)) == 0 for x in xs)]
assert sorted(roots) == [[1, 2], [3, 1]]           # f = 1 + 2x and f = 3 + x

for f in roots:
    w = codeword(f)
    agree = sum(1 for j in range(n) if w[j] == received[j])
    assert agree == 3 and n - agree == 2           # both at distance exactly 2

assert sent in [codeword(f) for f in roots]        # the true codeword is in the list

# --- section 6: the lemma ---------------------------------------------------------
# deg Q(x, f(x)) <= 2, yet it has 3 roots when f agrees in 3 places -> identically 0
for f in roots:
    assert all(Q(x, ev(f, x)) == 0 for x in xs)

# --- section 6: why the budget cannot simply be raised ----------------------------
# At B = 2 the kernel is one curve and the lemma forces it to contain every f agreeing
# in 3+ places. At B = 3 the lemma never fires, and most kernel curves are useless.
def monomials_upto(B):
    return [(i, j) for j in range(B + 1) for i in range(B + 1) if i + j <= B]


def kernel_curves(mons):
    rows = [[pow(x, i, p) * pow(y, j, p) % p for i, j in mons]
            for x, y in zip(xs, received)]
    return [v for v in product(range(p), repeat=len(mons))
            if any(v) and all(sum(a * b for a, b in zip(r, v)) % p == 0 for r in rows)]


def y_roots(v, mons):
    return [f for f in product(range(p), repeat=k)
            if all(sum(c * pow(x, i, p) * pow(ev(list(f), x), j, p)
                       for c, (i, j) in zip(v, mons)) % p == 0 for x in xs)]


budget_stats = {}
for B in (2, 3):
    mons = monomials_upto(B)
    curves = kernel_curves(mons)
    hit = sum(1 for v in curves if [1, 2] in [list(f) for f in y_roots(v, mons)])
    none = sum(1 for v in curves if not y_roots(v, mons))
    budget_stats[B] = (len(mons), len(curves) // (p - 1),
                       hit // (p - 1), none // (p - 1))

assert budget_stats[2] == (6, 1, 1, 0), budget_stats[2]
assert budget_stats[3] == (10, 781, 156, 419), budget_stats[3]

print(f"n={n} k={k} d_min={d_min}  unique radius={unique_radius}  "
      f"Johnson={johnson:.3f}")
print(f"sent      {sent}   f(x) = 1 + 2x")
print(f"received  {received}   errors at {errors}")
print(f"Q(x,y) = {Q_coeffs[0]} + {Q_coeffs[1]}x + {Q_coeffs[2]}x^2 "
      f"+ {Q_coeffs[3]}y + {Q_coeffs[4]}xy + {Q_coeffs[5]}y^2")
print("list decoded:")
for f in sorted(roots):
    w = codeword(f)
    agree = sum(1 for j in range(n) if w[j] == received[j])
    tag = "  <- the one that was sent" if w == sent else ""
    print(f"   f(x) = {f[0]} + {f[1]}x  ->  {w}  distance {n - agree}{tag}")
print("\nall assertions in GURUSWAMI_SUDAN.md verified")
