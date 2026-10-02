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
temperatures and extreme coordinates. Coinciding with a center does not force a
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
the initial fallback for every column is the robust data mean. `initCenters`
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

Finite coordinate differences, squared costs and signed objective products use
exact binary integer accumulation with extended exponent range. Softmax uses
ordinary binary64 `Math.exp`; entropy uses `Math.log` of the actual returned
membership. The scalar `somStableMean` helper supplies robust convex means;
no SOM model update is reused. This does not make the whole solver an
arbitrary-precision optimizer.

Unrepresentable objectives are reported without NaN or clipping:

- `objectiveRepresentation`: `'finite'`, `'overflow'`, or `'underflow'`
- `objectiveSign`: `-1`, `0`, or `1`, including the sign of an underflowed value
- `objectiveLogAbs`: log of the absolute extended-range objective, or `-Infinity`
  for exact zero
- `objective`: a finite Number, signed infinity on overflow, or signed zero on
  underflow

`numericalMode: 'exact-binary-reference'` describes the scalar arithmetic even
when the one-row cost cache is used. The cache only reuses the squared costs
just computed for that same membership row, before any yield or callback. It
does not change summation order or introduce a distance matrix. Memory is
`O(ND + NK + KD + K)` plus optional iteration history; each iteration is
`O(NKD)`, with additional integer-arithmetic cost for wide exponent ranges.

## Execution scope

`run('entropy-fcm', ...)`, `steps('entropy-fcm', ...)`, `runAsync`, and the existing
one-shot Worker client route through the same algorithm registry.
`signal`, `shouldCancel`, and `onProgress` follow the core API. The generator
yields between center columns and membership row blocks; an individual scalar
mean or row is synchronous, so cancellation latency can grow with its size.

Stateful `createSession` and the realtime Worker are **not supported** for
`entropy-fcm` in this first release. No browser, GPU, or cross-platform speed
claim is made by the Node correctness tests. The shared 80-digit Decimal
fixtures use explicit memberships and cover linear weights, coincident
centers, asymmetric multidimensional data, and an initially empty cluster.
