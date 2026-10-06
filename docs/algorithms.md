# Algorithms

Every method repeats one step on data $X = \{x_i\}_{i=1}^N \subset \mathbb{R}^D$
and prototypes $V = \{v_c\}_{c=1}^K$ under the same stopping rule. All
clustering methods and two of the maps use the standard alternating step
(the online SOM's step is one epoch; `kmeans` computes the same step with
Hamerly's bounds, which skip only distances that cannot change a label):

$$
d_{ic}^2 = \lVert x_i - v_c \rVert^2
\;\longrightarrow\;
U = \mathrm{assign}(D)
\;\longrightarrow\;
v_c = \frac{\sum_i w(u_{ic})\, x_i}{\sum_i w(u_{ic})}
$$

Inputs must be finite with row norms below $10^{150}$ and, unless all rows are
equal, some feature spanning at least $10^{-150}$, so that squared distances
neither overflow nor all vanish; other inputs raise an error, so standardize
data first. A prototype with no mass keeps its previous position. The loop
stops when no prototype coordinate moves more than `tol` times the RMS radius
of the centered data (`tol = 0` means an exact fixed point), or after
`max_iter` iterations; stopping says nothing about the quality of the
partition (see "Degenerate solutions").

The data are an input of every iteration, so a run can go on with new data,
for example points that move. The prototypes keep their positions, and the
method keeps what belongs to its model: the schedule and random state of
the maps, and the memberships of `som_olp`, which give each point its latent
position. Points are known only by their rows, so `som_olp` keeps the
memberships while the number of rows stays the same, taking row $i$ to be
the same point as before; when it changes, the next iteration starts
without them, as the first does. Rows that are other points, such as
batches of equal size, must therefore not be sent to `som_olp`, whose next
iteration would give them the latent positions of the old points. What
belongs to the old data only, such as Hamerly's bounds or the neighborhood
graph of `rmcm`, is built again. The run then ends when it converges on unchanged data, or after
`max_iter` iterations (`None` in Python and `Infinity` in JavaScript for no
limit). For maps that follow moving data, a constant neighborhood width
(`sigma = sigma_end`) keeps the schedule from narrowing over time.

The default initialization, shared by the clustering methods, is greedy
k-means++ seeding (each seed is the best, by the resulting sum of squared
distances, of $2 + \lfloor \ln K \rfloor$ points drawn in proportion to the
squared distance to the nearest seed) followed by one k-means step: each seed
moves to the mean of the points nearest to it. Seeds are data points, and a
prototype exactly on a data point gives that point membership 1; with the
column-scaled weights of `fcm` the point then holds almost all of the
prototype's weight for large $m$, and the prototype stalls there (with
$m = 50$ in the benchmark data the prototypes moved less than `tol` and the
run stopped after one iteration). After the step a prototype sits on a data
point only when its cell holds that point alone. A cell is empty only when
seeds coincide, which needs fewer distinct points than $K$; such seeds stay
where they are, and coincident prototypes stay together.

| Method | `assign(D)` | weight $w(u)$ |
|---|---|---|
| `kmeans` | $u_{ic} = 1$ for $c = \arg\min_j d_{ij}$ (lowest index on ties) | $u$ |
| `fcm` | $u_{ic} \propto d_{ic}^{-2/(m-1)}$ | $u^m$ |
| `efcm` | $u_{ic} \propto \exp(-d_{ic}^2/\tau)$ | $u$ |
| `rcm` | $u_{ic} = \mathbf{1}[c \in A_i] / \lvert A_i \rvert$ | $u$ |
| `rmcm` | $R = P H$ | $u$ |
| `batch_som` | $u_{ij} = h_t(\mathrm{bmu}_i, j)$ | $u$ |
| `som_olp` | $p_{ij} \propto \exp\!\big(-(d_{ij}^2 + \gamma \lVert v_i - r_j \rVert^2)/\lambda\big)$ | $p$ |

## Fuzzy c-means (`fcm`)

Minimizes $J_m = \sum_{i,c} u_{ic}^m d_{ic}^2$ subject to $\sum_c u_{ic} = 1$
(Bezdek, 1981). The membership update is evaluated as
$u_{ic} \propto (d_{i,\min}^2 / d_{ic}^2)^{1/(m-1)}$, the softmax of
$-\log d_{ic}^2 / (m-1)$ without logarithms: the ratios lie in $(0, 1]$, so it
is stable for every $m > 1$ and tends to k-means as $m \to 1$. It is the only
membership rule that divides by a distance: squared distances are clipped
at the smallest positive normal number, so a point on a prototype gets
membership 1 there, shared equally among prototypes exactly on it, which is
the limit of the formula (Bezdek's rule for $d_{ic} = 0$). Each
column of $u^m$ is evaluated as $(u_{ic} / \max_j u_{jc})^m$, which leaves the
weighted means unchanged and cannot underflow to all zeros for large $m$.
`history` holds $J_m$ per iteration, computed with the memberships as
$\sum_i d_{i,\min}^2\, s_i^{1-m}$, where
$s_i = \sum_c (d_{i,\min}^2 / d_{ic}^2)^{1/(m-1)}$ is the sum that normalizes
them; this equals $J_m$ at the memberships of the same distances, since
$u_{i,\max} = 1/s_i$.

## Entropy-regularized FCM (`efcm`)

Minimizes $\sum u_{ic} d_{ic}^2 + \tau \sum u_{ic} \log u_{ic}$ (Miyamoto &
Mukaidono, 1997), giving a softmax of $-d^2/\tau$ with linear center weights.
The softmax is evaluated as $\exp(-(d_{ic}^2 - d_{i,\min}^2)/\tau)$,
normalized: every exponent lies in $[-\infty, 0]$, so a $\tau$ however small
against the distances gives the hard memberships of its limit, k-means,
rather than an overflow. `history` holds the objective, computed with the
memberships as the sum of the soft minima
$d_{i,\min}^2 - \tau \log \sum_c \exp(-(d_{ic}^2 - d_{i,\min}^2)/\tau)$,
from the row minima and sums that the softmax computes; this equals the
objective at the memberships of the same distances (and likewise for
`som_olp` with its costs and $\lambda$).

## Degenerate solutions (`fcm`, `efcm`)

With every prototype at the mean $\bar x$ of the data, all distances from a
point are equal, the memberships are $1/K$ and the update returns $\bar x$:
the mean is a fixed point for every $m$ and $\tau$. Linearizing the update
around it (Yu, Cheng & Huang, 2004) shows that differences $\delta$ between
prototypes evolve as $\delta' = \frac{2m}{m-1} M \delta$ for `fcm`, with
$M = \frac1N \sum_i y_i y_i^\top / \lVert y_i \rVert^2$ and
$y_i = x_i - \bar x$, and as $\delta' = \frac{2}{\tau} S \delta$ for `efcm`,
with $S$ the covariance of the data. The collapsed state therefore attracts
the iteration when

$$
m \ge m^* = \frac{1}{1 - 2\lambda_{\max}(M)} \;\;(\lambda_{\max}(M) < \tfrac12),
\qquad \tau \ge \tau^* = 2\lambda_{\max}(S),
$$

the second being the critical temperature of deterministic annealing (Rose,
1998), which also applies to $\lambda$ in `som_olp`. The eigenvalues of $M$
sum to 1, so in two dimensions $\lambda_{\max}(M) \ge \frac12$ and `fcm`
never collapses, while for $D$ roughly isotropic features $m^* \approx
D/(D-2)$: already 1.14 for $D = 16$, so the default $m = 2$ can collapse from
about five dimensions on (Winkler, Klawonn & Kruse, 2011). Above $m^*$ the
collapse is possible, not certain: the iteration converges to a local
minimum or saddle point of the objective, and which one depends on the start.

A lower objective is not a better partition. Past $m^*$ the collapsed state
has the lowest $J_m$ yet labels no better than chance (ARI 0.03 to 0.19 in
the benchmark data), and `converged` only says that the prototypes stopped
moving. Keep $m$ below $m^*$ (often 1.2 to 1.5 in high dimensions) and
$\tau$ below $\tau^*$, and check that the prototypes have not merged, for
example by their spread around the mean, or compare with known labels.

## Rough c-means (`rcm`)

Cluster $c$ is admissible for $x_i$ when

$$
d_{ic}^{\,p} \le (\alpha\, d_{i,\min})^p + \beta^p ,
$$

and $x_i$ shares unit membership equally among its admissible set $A_i$.
$p = 1$ is RCM; other $p$ give ExRCM. $\alpha \ge 1$ and $\beta \ge 0$ widen the
upper approximation; $\alpha = 1, \beta = 0$ is k-means. The radius is computed
in scaled form to avoid overflow and underflow of the powers.

## Rough membership c-means (`rmcm`)

$P$ is the row-normalized adjacency of $\lVert x_i - x_j \rVert \le \delta$
(self included) and $H$ the one-hot nearest-center assignment; memberships are
$R = P H$, the fraction of each point's neighborhood assigned to each cluster.
`max_edges` refuses graphs that would not fit in memory before building them.

## Self-organizing maps

Units $j$ sit at grid coordinates $r_j$ (unit $j$ = `row * cols + col` at
`(row, col)`). Unless `init` is given, prototypes start on the plane of the two
leading principal axes (grid rows along the first), spread over
$\pm$`pca_scale` standard deviations (2 for `som` and `batch_som`). Each axis
is oriented so that its first clearly nonzero component is positive, which
stays well defined when components tie in magnitude, as for standardized
2-D data. Python takes the axes from the smaller of $X^\top X$ and $X X^\top$;
JavaScript uses Rayleigh–Ritz on a Krylov basis of at most 64 vectors, which
is exact for $D \le 64$ and never forms the $D \times D$ covariance.
The neighborhood is $h_t(b, j) = \exp(-\lVert r_b - r_j \rVert^2 / 2\sigma_t^2)$
with $\sigma_t$ decaying geometrically from `sigma` (half the grid extent) to
`sigma_end`. It is evaluated by dividing by $\sigma_t$ twice (or squaring
$\lVert r_b - r_j \rVert / \sigma_t$), never by $\sigma_t^2$, which underflows
for a tiny width: such a width gives its limit, the winner alone.

- `som`: per sample, $w_j \leftarrow w_j + \eta_t h_t(\mathrm{bmu}, j)(x - w_j)$
  with $\eta_t$ decaying geometrically from `lr` to `lr_end`.
- `batch_som`: per epoch, the weighted mean above. The Gaussian is separable on
  a rectangular grid, so smoothing costs $O(K(\text{rows}+\text{cols})D)$.
- `som_olp` (Ubukata): minimizes
  $\sum_{ij} p_{ij}\big(\lVert x_i - w_j\rVert^2 + \gamma\lVert v_i - r_j\rVert^2\big) + \lambda \sum_{ij} p_{ij}\log p_{ij}$
  with latent positions $v_i = \sum_j p_{ij} r_j$ taken from the previous
  memberships. `embedding` is $V$, `history` the objective.

## Metrics

- `ari`: adjusted Rand index (Hubert & Arabie, 1985), from exact integer pair counts.
- `ami`: adjusted mutual information (Vinh, Epps & Bailey, 2010) with the exact
  hypergeometric expectation; `average` selects the entropy normalization.
  Each expectation sum runs over $n_{ij}$ within $\sqrt{35\min(a_i, b_j)}$ of
  its mean $a_i b_j / N$; by Hoeffding's inequality the omitted probability
  mass is below $2e^{-70}$, so the value equals the full sum to rounding. The
  log-factorials come from one table of $\log t!$, $t = 0..N$.
- `trustworthiness(X, Y, k)` (Venna & Kaski, 2001):
  $1 - \frac{2}{Nk(2N-3k-1)} \sum_i \sum_{j \in U_k(i)} (r_{ij} - k)$, where
  $U_k(i)$ are the $k$ nearest neighbors of $i$ in $Y$ that are not among its
  $k$ nearest in $X$ and $r_{ij}$ is the rank of $j$ in $X$.
  `continuity(X, Y, k)` is `trustworthiness(Y, X, k)`. Distances use direct
  differences and ties are ordered by sample index, so grid embeddings are exact.

## TPE

Multivariate Tree-structured Parzen Estimator (Bergstra et al., 2011; Falkner
et al., 2018). After `n_startup` random trials, the best `gamma` fraction and
the rest are modeled by Parzen mixtures with one component per trial (all
dimensions together) plus a wide prior. Numeric kernels are truncated normals
on $[0, 1]$ whose width is the larger gap to the neighboring centers;
categorical kernels put $1 - \varepsilon$ on the observed option. The best of
`n_candidates` samples from $l(x)$ by $l(x)/g(x)$ is proposed.
