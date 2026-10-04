# Entropy-regularized fuzzy c-means

`entropyFcm(input, options)` and the generator `entropyFcmSteps(input, options)`
fit the entropy-regularized objective

\[
J(U,V)=\sum_{i=1}^{N}\sum_{c=1}^{K}u_{ic}\lVert x_i-v_c\rVert_2^2
+\tau\sum_{i=1}^{N}\sum_{c=1}^{K}u_{ic}\log u_{ic},
\quad u_{ic}\ge0,\quad\sum_c u_{ic}=1,\quad\tau>0.
\]

Here `0 log 0 = 0`. The entropy term can make the objective negative.
This model uses linear membership weights, without an FCM fuzzifier `m`.
It has its own updates and does not use SOM-OLP's latent positions or grid cost.

For fixed memberships, differentiating the objective gives

\[
v_c=\frac{\sum_i u_{ic}x_i}{\sum_i u_{ic}}.
\]

For fixed centers, the simplex Lagrange multiplier gives a softmax. With
`d_ic = ||x_i-v_c||²` and `a_i = min_c d_ic`, the implementation evaluates

\[
u_{ic}=\frac{\exp(-(d_{ic}-a_i)/\tau)}
{\sum_j\exp(-(d_{ij}-a_i)/\tau)}.
\]

Subtracting the minimum **before** dividing by `tau` is essential for very small
temperatures, provided squared distances remain representable. Coinciding with a center does not force a
hard assignment. Larger `tau` generally produces softer memberships; `tau` is
in squared-coordinate units. No standardization is performed automatically.

```js
import { entropyFcm, run } from 'ubukit-js';

const input = {
  data: Float64Array.of(0, 2), nSamples: 2, nFeatures: 1
};
const options = {
  initMembership: Float64Array.of(0.8, 0.2, 0.2, 0.8),
  tau: 2, maxIterations: 1, returnHistory: true
};
const result = entropyFcm(input, options);
// centers ≈ [0.4, 1.6]; objective ≈ -0.7331298693521248
const same = run('entropy-fcm', input, options);
```

## Options and initialization

Input follows `normalizeInput`: a finite numeric TypedArray in sample-major
order, with positive integer `nSamples` and `nFeatures`.

| Option | Default | Meaning |
| --- | --- | --- |
| `nClusters` | inferred from initialization | Positive cluster count; can exceed sample count |
| `tau` | `1` | Finite positive Number, including subnormal temperatures |
| `initMembership` | seeded random rows | Numeric TypedArray of `N*K` finite nonnegative values, with positive row sums |
| `initCenters` | absent | Numeric TypedArray of `K*D` finite values, used to initialize memberships by softmax |
| `seed` | `0` | Safe integer for the existing JavaScript Mulberry32 generator |
| `maxIterations` | `100` | Positive safe integer |
| `tolerance` | `1e-5` | Nonnegative absolute Frobenius membership-change threshold |
| `returnHistory` | `false` | Include an owned `objectiveHistory` Float64Array |
| `blockRows` | `512` | Positive row-block size for membership updates |
| `maxMemoryBytes` | `512 * 1024**2` | Conservative primary-array and scalar-workspace estimate, not a total JS heap limit |

Membership rows are normalized on an owned copy using row-maximum scaling.
The initializers are mutually exclusive; supplying both raises `RangeError`.
Both require valid shapes and finite values. With neither initializer, set `nClusters`.
JavaScript and Python use different random generators, so cross-language
comparisons must supply explicit memberships. JavaScript also supports
`initCenters`; Python's initial public API uses memberships.

Each iteration first updates centers from the old memberships, then updates
memberships from those centers. An empty column retains its previous center;
the initial fallback for every column is the float64 data mean. `initCenters`
only seeds memberships and does not change this empty-column fallback.
Memberships that round to zero have zero weight on later iterations; no hidden
positive weights or probability floors are introduced.

Convergence means `delta < tolerance`. Setting `tolerance: 0` runs exactly
`maxIterations` iterations. This membership-only test does not certify that
another center update would be stationary or that the nonconvex problem has
reached a global minimum. Try several initializations when local minima matter.

## Results and numerical behavior

The result contains owned flat Float64Arrays `centers` (`K*D`) and `membership`
(`N*K`, `membershipLayout: 'samples-clusters'`), plus owned Int32Array `labels`
(`N`). Labels select the first maximum membership. It also contains `objective`,
`fpc`, `tau`, `delta`, `iterations`, `converged`, dimensions, and
`backend: 'javascript-float64'`. Optional history evaluates the center/membership
pair returned at each completed iteration; its last value equals `objective`.

All center updates, squared distances, softmax and objective accumulation use
ordinary float64 (`Number`) arithmetic. There is no BigInt, exact-reference,
higher-precision, or SOM mean fallback. `numericalMode: 'float64'` records this
contract. Supplying a backend other than `'javascript-float64'` is rejected.

Minimum subtraction stabilizes softmax normalization, but it cannot restore a
distance gap already lost when squared costs were rounded. For example, costs
`1e16` and `1e16 + 1` both round to `1e16`, yielding equal memberships. Weighted
means and objectives can lose cancellation residuals. Subnormal terms and tiny
memberships may underflow to zero; no hidden probability floor is introduced.

Nonfinite center, squared-distance, or objective intermediates raise `RangeError`.
This includes cases where extended-range arithmetic formerly returned a result.
Positive softmax ratios may overflow to infinity, yielding valid zero weights.
No overflowed cost is silently treated as a valid distance or clipped result.
Rescale coordinates and temperature together where appropriate.

On success, `objectiveRepresentation` is `'finite'`; `objectiveSign` is `-1`, `0`,
or `1`, and `objectiveLogAbs` is the log of the rounded absolute objective or
`-Infinity` for rounded zero. These diagnostics no longer distinguish exact zero
from underflow or cancellation to zero. The former extended-range status and
exact-gap guarantees have been removed.

Memory is `O(ND + NK + KD + K)` plus optional iteration history; each iteration
is `O(NKD)` with ordinary float64 arithmetic.

## Execution scope

`run('entropy-fcm', ...)`, `steps('entropy-fcm', ...)`, `runAsync`, and the existing
one-shot Worker client route through the same algorithm registry.
`signal`, `shouldCancel`, and `onProgress` follow the core API. The generator
yields between center columns and membership row blocks; an individual scalar
mean or row is synchronous, so cancellation latency can grow with its size.

Stateful `createSession` and the realtime Worker are **not supported** for
`entropy-fcm` in this first release. No browser, GPU, or cross-platform speed
claim is made by the Node correctness tests. Tests use analytic known outputs and float64 tolerances. Retained historical
fixture constants are read without executing or regenerating a higher-precision
oracle. They cover linear weights, coincident centers, asymmetric
multidimensional data, and an initially empty cluster.
