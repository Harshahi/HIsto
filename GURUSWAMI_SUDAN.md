# Guruswami–Sudan, worked by hand

A complete list-decoding example small enough to do with pencil and paper, then a map
from each step to the code in this repo. Every number below is machine-checked by
`verify_gs_example.py`.

---

## 1. The setup

Work over $\mathbb{F}_5 = \{0,1,2,3,4\}$, all arithmetic mod 5. Take the Reed–Solomon
code of dimension $k = 2$ evaluated at **all five** field elements:

$$x_0,\dots,x_4 \;=\; 0,1,2,3,4, \qquad
f(x) = m_0 + m_1x, \qquad
c = \bigl(f(0),\,f(1),\,f(2),\,f(3),\,f(4)\bigr)$$

So $n = 5$, $k = 2$, and by the Singleton bound

$$d_{\min} = n - k + 1 = 4 .$$

The three radii:

| | formula | value |
|---|---|---|
| unique decoding | $\left\lfloor \tfrac{d_{\min}-1}{2}\right\rfloor$ | $\mathbf{1}$ |
| Johnson | $n - \sqrt{n(n-d_{\min})} = 5 - \sqrt 5$ | $2.764$ |

We will correct **two** errors — strictly past unique decoding, strictly below Johnson.

## 2. Send, and corrupt

Take $f(x) = 1 + 2x$:

$$c = (1,\,3,\,0,\,2,\,4)$$

Corrupt positions 1 and 3 ($3 \to 4$ and $2 \to 1$):

$$r = (1,\,4,\,0,\,1,\,4)$$

| $j$ | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| $c_j$ | 1 | **3** | 0 | **2** | 4 |
| $r_j$ | 1 | **4** | 0 | **1** | 4 |
| | ok | ✗ | ok | ✗ | ok |

Two errors. Unique decoding handles one. **It cannot answer this** — and not because
it is weak, but because the answer genuinely is not unique, as §5 shows.

## 3. Step one — interpolation

Find a non-zero $Q(x,y)$ vanishing at all five points
$(0,1), (1,4), (2,0), (3,1), (4,4)$, of $(1, k-1) = (1,1)$-weighted degree at most 2.
Weighted, because $y$ stands for a degree-$\le 1$ polynomial, so $y$ costs the same as
$x$. The monomials with $i + j \le 2$ are

$$1,\quad x,\quad x^2,\quad y,\quad xy,\quad y^2$$

— six unknowns. Write

$$Q(x,y) = a + bx + cx^2 + dy + exy + hy^2 .$$

Each point gives **one linear equation**, so five equations in six unknowns. More
unknowns than equations ⟹ a non-zero solution must exist. That is the whole trick.

$$
\begin{aligned}
(0,1):&\quad a + d + h &= 0\\
(1,4):&\quad a + b + c + 4d + 4e + h &= 0\\
(2,0):&\quad a + 2b + 4c &= 0\\
(3,1):&\quad a + 3b + 4c + d + 3e + h &= 0\\
(4,4):&\quad a + 4b + c + 4d + e + h &= 0
\end{aligned}
$$

(Using $x^2 \bmod 5$: $3^2 = 9 \equiv 4$, $4^2 = 16 \equiv 1$; and $y^2$: $4^2 \equiv 1$.)

### Solving it

From the first equation, $a = -(d+h) = 4d + 4h$. Substitute into the rest; the $a$
column clears and the $h$ terms collapse ($5h \equiv 0$):

$$
\begin{aligned}
b + c + 3d + 4e &= 0 &&(2')\\
2b + 4c + 4d + 4h &= 0 &&(3')\\
3b + 4c + 3e &= 0 &&(4')\\
4b + c + 3d + e &= 0 &&(5')
\end{aligned}
$$

Now subtract and reduce:

- $(2') - (5')$: $\;-3b + 3e = 0 \Rightarrow 2b + 3e = 0 \Rightarrow \mathbf{b = e}$
- $(4')$ with $e = b$: $\;6b + 4c = b + 4c = 0 \Rightarrow \mathbf{b = c}$
- $(2')$ with $c = e = b$: $\;6b + 3d = b + 3d = 0 \Rightarrow \mathbf{b = 2d}$
- $(3')$ with $b = c = 2d$: $\;6d + 4h = d + 4h = 0 \Rightarrow \mathbf{d = h}$

One free parameter, as expected. Set $h = 1$, which makes $Q$ monic in $y$ and so
lets §4 factor it by inspection:

$$d = 1,\quad b = c = e = 2,\quad a = 4d + 4h = 8 \equiv 3$$

$$\boxed{\,Q(x,y) \;=\; 3 + 2x + 2x^2 + y + 2xy + y^2\,}$$

### Check it vanishes

$$
\begin{aligned}
Q(0,1) &= 3 + 1 + 1 = 5 \equiv 0\\
Q(1,4) &= 3 + 2 + 2 + 4 + 8 + 16 = 35 \equiv 0\\
Q(2,0) &= 3 + 4 + 8 = 15 \equiv 0\\
Q(3,1) &= 3 + 6 + 18 + 1 + 6 + 1 = 35 \equiv 0\\
Q(4,4) &= 3 + 8 + 32 + 4 + 32 + 16 = 95 \equiv 0
\end{aligned}
$$

All five. ✓

## 4. Step two — factor

Collect $Q$ as a polynomial in $y$:

$$Q = y^2 + (1 + 2x)\,y + (2x^2 + 2x + 3)$$

It is monic of degree 2 in $y$, so if it splits as $(y - f_1)(y - f_2)$ then

$$f_1 + f_2 = -(1+2x) = 4 + 3x, \qquad f_1 f_2 = 2x^2 + 2x + 3 .$$

Try $f_1 = 1 + 2x$. Then $f_2 = (4+3x) - (1+2x) = 3 + x$, and

$$(1+2x)(3+x) = 3 + x + 6x + 2x^2 = 2x^2 + 7x + 3 \equiv 2x^2 + 2x + 3 \quad\checkmark$$

$$\boxed{\,Q = \bigl(y - (1+2x)\bigr)\bigl(y - (3+x)\bigr)\,}$$

Equivalently, substituting $y = 1+2x$ into $Q$ gives $10x^2 + 10x + 5 \equiv 0$.

## 5. The list — and why uniqueness really does fail

| $f$ | codeword | vs $r = (1,4,0,1,4)$ | agree | distance |
|---|---|---|---|---|
| $1 + 2x$ | $(1,3,0,2,4)$ | ✓·✗·✓·✗·✓ | 3 | **2** |
| $3 + x$ | $(3,4,0,1,2)$ | ✗·✓·✓·✓·✗ | 3 | **2** |

**Two distinct codewords sit at distance exactly 2 from $r$.** No decoder can pick
between them from $r$ alone — the information is not there. Unique decoding is not
merely failing to find the answer; there is no single answer to find.

Guruswami–Sudan returns *both*, and the true one is guaranteed to be among them. That
is what list decoding buys: it trades a unique answer for a short, complete one.

## 6. Why the method works

Write $B$ for the **degree budget**: the bound on $\deg_{1,k-1} Q$ that Step 1
interpolates under. In this example $B = 2$ — that is where the "at most 2" in §3 came
from. Guruswami--Sudan is, from end to end, a fight over this one number.

### The lemma

Suppose $f$ has $\deg f < k$ and agrees with $r$ in more than $B$ positions. Consider
the univariate *composite* (unrelated to §3's coefficient $h$ — this $g$ is a
polynomial in $x$, not a number)

$$g(x) \;=\; Q\bigl(x, f(x)\bigr).$$

- $\deg g \le B$ — this is exactly what the weighting bought. $y$ was charged $k-1$
  precisely so that substituting a degree-$<k$ polynomial cannot exceed the budget.
- $g(x_j) = Q(x_j, r_j) = 0$ at every position where $f$ agrees with $r$, because $Q$
  was interpolated through those points.

So $g$ has degree at most $B$ and more than $B$ roots. The only such polynomial is the
zero polynomial. Hence $g \equiv 0$, and by the factor theorem in
$(\mathbb{F}_q[x])[y]$,

$$(y - f(x)) \;\bigm|\; Q(x,y).$$

Above: 3 agreements against $B = 2$, and $3 > 2$. §4's substitution lets you watch the
collapse happen — $10x^2 + 10x + 5$, every coefficient $\equiv 0 \bmod 5$.

### The squeeze: $B$ is pulled both ways

This is the part to slow down on, because $B$ has to satisfy two conditions that pull in
**opposite** directions.

**$B$ must be large enough for $Q$ to exist.** Step 1 needs more unknowns than
equations:

$$\underbrace{\#\{\,(i,j) : i + j(k-1) \le B\,\}}_{\text{unknowns}}
\;>\;
\underbrace{n \cdot \tfrac{s(s+1)}{2}}_{\text{constraints}}$$

Raising $B$ admits more monomials, so this pushes $B$ **up**.

**$B$ must be small enough for the lemma to fire.** The lemma needs agreement $> B$, and
you can agree in at most $n$ positions. Tolerating $e$ errors means $n - e > B$. This
pushes $B$ **down**.

What is left between them is the decoding radius:

$$e \;\le\; n - B - 1$$

So you want the **smallest** $B$ that still guarantees a curve. This is why "just raise
the budget so interpolation is easier" is self-defeating: it buys existence by spending
radius. The budget *is* the threshold.

### Concretely, on this example

With $k - 1 = 1$ the monomials under budget $B$ number $(B+1)(B+2)/2$, against $n = 5$
points at $s = 1$:

| $B$ | monomials | $> 5$, so $Q$ guaranteed? | agreement needed | errors correctable |
|---|---|---|---|---|
| 1 | 3 | no | --- | --- |
| **2** | **6** | **yes** | **3** | **2** |
| 3 | 10 | yes | 4 | 1 |
| 4 | 15 | yes | 5 | 0 |

$B = 2$ is forced from below and optimal from above: the smallest budget that guarantees
a curve, and every larger one corrects strictly fewer errors.

What goes wrong at $B = 3$ is worth stating precisely, because the failure is
probabilistic rather than absolute. Interpolation still succeeds — it succeeds *more*
easily, 10 monomials against 5 equations, leaving a 5-dimensional kernel. But 3
agreements no longer exceed the budget, so the lemma never fires, and **nothing forces
the curve you happen to find to contain the answer**. Counting every curve in each
kernel:

| | $B = 2$ | $B = 3$ |
|---|---|---|
| monomials | 6 | 10 |
| kernel dimension | 1 | 5 |
| distinct curves, up to scaling | **1** | **781** |
| containing the true $f = 1+2x$ | 1 (**100%**) | 156 (**20%**) |
| with no degree-$<2$ root at all | 0 | 419 (**54%**) |

At $B = 2$ there is exactly one curve, and the lemma compels it to contain every $f$
agreeing in 3 or more places — which is why §4 recovers both codewords from it. At
$B = 3$ there are 781, four fifths of which miss the answer and over half of which
factor into nothing at all. Some still work: any multiple of the $B = 2$ curve by a
low-degree factor remains in the kernel. But the algorithm cannot tell those apart, and
takes whichever basis vector the solver returns.

So a larger budget does not make the decoder wrong so much as **unguaranteed** — it
trades a certainty for a one-in-five chance. That distinction returns at scale: it is
exactly the situation §5 of `REPORT.md` finds at $t = 14$, where the guarantee is gone
and each answer has to be certified after the fact rather than promised in advance.

(The same effect at full scale, measured in `histogram_recovery.ipynb`: a budget past
$Nd$ gives non-empty kernels and **zero** roots.)

Note that $e \le 2$ here matches the Johnson radius $5 - \sqrt 5 = 2.76$ from §1, so at
these parameters $s = 1$ already reaches the optimum.

### What multiplicity changes

Now require $Q$ to vanish to order $s$ at each point. Two things move at once:

| | $s = 1$ | general $s$ |
|---|---|---|
| cost of one point | 1 equation | $s(s+1)/2$ equations |
| worth of one agreement | a simple root of $g$ | a root of multiplicity $s$ |
| lemma fires when | $t > B$ | $s\,t > B$, i.e. $t > B/s$ |

Cost grows like $s^2/2$, but the threshold is $B/s$ — so the two do **not** scale
together, and that asymmetry is the entire point. Balancing unknowns against
constraints gives

$$B \;\approx\; \sqrt{n(k-1)\,s(s+1)},
\qquad
t \;>\; \frac{B}{s} \;=\; \sqrt{n(k-1)\Bigl(1 + \tfrac{1}{s}\Bigr)}$$

The agreement you need **falls** as $s$ rises: $\sqrt{2n(k-1)}$ at $s = 1$, decreasing
to $\sqrt{n(k-1)}$ as $s \to \infty$ — which is exactly the Johnson radius
$n - \sqrt{n(n-d)}$. That is why multiplicity exists, and why it is the knob
`core.py` calls `d`.

(Those are large-parameter approximations. At $n = 5$ the exact counts in the table
above are what govern.)

## 7. Where this lives in the code

| step above | code |
|---|---|
| build the point set | `core.py:unbundle` — `P = [(points[j], points[a]) ... if Y[a,j]]` |
| interpolate | `gs_interpolation_linalg(P, tau, (d, ell), K-1)` |
| the degree budget | `tau = len(P) - N`, which makes Sage's budget $d\cdot N$ |
| factor | `ctx["root_finder"](Q, maxd=K-1)` |
| the raw linear system | `core.py:attempt` — `_monomial_list`, `_interpolation_matrix_given_monomials`, `.right_kernel()` |
| off-the-shelf version | `guruswami_sudan.ipynb` — `GRSGuruswamiSudanDecoder(CODE, tau=...)` |

Two differences between this example and histogram recovery are worth naming.

**One point per position, versus many.** Classical GS receives $r$ and gets $n$ points
$(x_j, r_j)$. Histogram recovery receives $Y$ — *which symbols appeared* at each
position — and so gets one point per observed symbol: $n_P = \sum_j |S_j|$, which at
$t = 3,\ N = 27$ is 81 points rather than 27. There is no single "received word"; the
point cloud is what several codewords pass through at once, and you want all of them.

**A score filter afterwards.** Steps 1–2 are Algorithm 2 as Sidorenko–Fischer wrote it.
`histo.pdf` adds a third step: keep only candidates scoring the maximum $dN$. A genuine
bundled codeword scores $dN$ by construction; an impostor is capped at $d\,t(k-1)$,
since two distinct degree-$<k$ polynomials agree in at most $k-1$ places. Thresholding
at $dN$ separates them exactly while $t(k-1) < N$ — which is where
$u_{\max} = \lfloor (N-1)/(k-1)\rfloor$ comes from.
