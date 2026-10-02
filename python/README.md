# UbuKit for Python

**PRIVATE PREVIEW. Not published to PyPI, TestPyPI, or any registry.**
The provisional distribution remains `ubukit-bundled-local-preview`; this is not
an approved public name. Project contributions use the [MIT License](LICENSE);
see [license scope](LICENSE-SCOPE.txt) for retained third-party terms.

Start with [installation](#installation-boundary),
[clustering and metric examples](#core-clustering-and-metric-examples),
[online and batch self-organizing maps (SOM)](SOM.md), or [parameter optimization](OPTIMIZATION.md).

## Installation boundary

Use a **new virtual environment** for this private trial. The previous aggregate
installed overlapping top-level modules; upgrading/uninstalling it in a mixed
legacy environment can remove files another distribution owns. The
[install-environment guard](tools/check_install_environment.py) checks the target environment read-only.
It blocks actual foreign `ubukit` ownership/imports and an old layout of this
same aggregate. Unrelated standalone legacy module names no longer collide with
the new package and do not by themselves block installation. No package is
removed or repaired automatically. `pip check` alone does not detect file-owner
collisions.

For a repository checkout, follow the [repository installation commands](../docs/getting-started.md#python)
from the repository root. They build a wheel and install it in a fresh environment
with `python/constraints/constraints-namespace-verified.txt`, the tested Linux /
Python 3.12 base stack. Built wheels are not committed. These pins are not a
tested matrix for every declared Python version. Stop if the environment guard
reports a conflict.

The public import is `import ubukit`. Former top-level imports
`portable_accel`, `ubukit_fcm`, `rough_cmeans`, `ubukit_rmcm`, `external_metrics`,
`_numba_kernel`, and `_external_metrics_numba` are not installed or aliased.
Use existing `ubukit` exports instead. Paths under `ubukit._impl` are unsupported
implementation details, not replacement public imports.

The `ubukit` root import is lazy and imports neither NumPy nor Numba.

## Saved objects and optional compiler caches

The namespace move changes implementation classes/functions' module paths.
Cross-version pickle compatibility is not promised. Do not patch pickle bytes
or load untrusted pickle files; retain the original isolated environment
when an existing serialized object requires its old module paths. Recreate a
model from supported data/parameters where possible. This migration adds no
model persistence format or pickle migration tool.

Old Numba `.nbc`/`.nbi` caches are not distributed or reused as compatibility
artifacts. Use a fresh candidate-specific cache for validation; first use may
compile again. Numba remains optional and is loaded only when a selected path
needs it. No global module aliases recreate old names.

## Results: samples first

| Family | Entry point | Result contract |
| --- | --- | --- |
| k-means: one cluster per sample | `fit_kmeans(X, init, ...)` | dict; labels `(N,)` |
| Fuzzy c-means (FCM): degrees of cluster membership | `fit_fcm(X, n_clusters, ...)` | dict; membership `(N,K)` |
| Rough c-means (RCM): overlapping cluster assignments | `fit_rcm(X, n_clusters, ...)` | result adapter; memberships `(N,K)` |
| Extended rough c-means (ExRCM): rough clustering with adjustable exponent `p` | `fit_exrcm(X, n_clusters, ...)` | result adapter; memberships `(N,K)` |
| Rough membership c-means (RMCM): membership from fixed-radius neighborhoods | `fit_rmcm(X, n_clusters, delta=..., ...)` | result object; memberships `(N,K)` |
| Self-organizing maps with optimized latent positions (SOM-OLP): continuous sample positions on a supplied map | `fit_som_olp(X, R, ...)` | dict; `V` `(N,latentD)`, `P` `(N,M)` |
| Neighborhood quality | `joint_quality(X, Y, ks=..., ...)` | list of quality records |

`N` means samples and `K` means clusters. Every clustering
membership matrix is sample-major. For RCM/ExRCM, both `.memberships` (float64)
and `.upper_memberships` (bool) have shape `(N,K)`. `assign_rcm` still returns a
2-tuple with both arrays `(N,K)`. Rows of normalized memberships sum to 1.

FCM remains a dict with the singular key `["membership"]`. RMCM keeps its result
object and plural `.memberships` field. Their container types and membership axes retain the sample-major convention.
Container types and field names differ across algorithms. Use `PreparedSOM`
and `PreparedRMCM` for repeated fits; construct `PreparedRMCM` with `prepare_rmcm`.

The RCM/ExRCM result is a thin attribute-forwarding adapter, not a
`rough_cmeans.RoughCMeansResult` or dataclass. All non-membership metadata
(including centers, convergence/cycle information, initialization indices and
backend) is forwarded without changing values. Other arrays retain their
legacy ownership semantics. Each adapted membership array is a writable,
C-contiguous, separately owned transpose-copy, even when `N == K` or an axis
has length 1. It is never a view into the legacy output. `return_memberships=False`
still returns `None` for both membership fields. The private legacy result is
retained so metadata is available; the copies add O(NK) time and storage
(approximately 9NK bytes for float64 memberships plus bool upper memberships).

## Supported environments

The installation constraints describe checked Linux x86-64 / CPython 3.12
stacks. Other operating systems, architectures and Python versions, including
the declared minimum, are not a tested support matrix. Existing acceleration
measurements exclude the RCM/ExRCM membership-copy cost; they do not establish
end-to-end API speedups.

## ARI and AMI

Adjusted Rand index (ARI) and adjusted mutual information (AMI) measure
chance-adjusted agreement between two sets of cluster labels. Use
`ubukit.adjusted_rand_score`, `ubukit.adjusted_mutual_info_score`, or
`ubukit.adjusted_scores`. The joint helper returns
`{"ari": ..., "ami": ...}`. AMI accepts `average_method="arithmetic"` (default),
`"geometric"`, `"min"`, or `"max"`, and `backend="numpy"` (default) or `"numba"`.

The complete-support mathematical definition is unchanged. Ordinary-path
NumPy/Numba AMI is floating-point arithmetic, not a bitwise-equality guarantee.
Singular and conservatively screened ill-conditioned/high-cluster-density
cases call the installed scikit-learn implementation to preserve its numerical
convention. These calls can be slower. Test-panel tolerance is not a universal
error bound. Optional Numba is imported only when its path is needed.

Exact external-metrics attribution is in `NOTICE-EXTERNAL-METRICS.txt` and
`THIRD_PARTY_LICENSES.txt`. References to `NOTICE.txt` within the unchanged BSD
notice refer to the original external-metrics notice now supplied under that
longer filename. The project MIT selection does not replace these scoped terms.

## Core clustering and metric examples

Use these examples with the [installed package](#installation-boundary). The
examples share the following data and imports:

```python
import numpy as np
import ubukit as uk

X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
centers = X[[0, 3]].copy()
policy = uk.ExecutionPolicy(threads=1)
```

### k-means and fuzzy c-means

```python
km = uk.fit_kmeans(X, centers, backend="numpy", max_iter=10, policy=policy)
print(km["labels"])                    # One cluster index per sample

fcm = uk.fit_fcm(X, 2, random_state=4, backend="scipy", max_iter=10, threads=1)
print(fcm["membership"])               # Shape (samples, clusters)
```

`fit_fcm_numpy` exposes the NumPy reference entry point with the same membership
shape. Use `uk.prepare(X)` or `uk.PreparedData(X)` for an owned, reusable data
snapshot. Set optional Numba backends only when Numba is installed.

### Rough clustering

```python
rcm = uk.fit_rcm(X, 2, init=centers, backend="numpy", max_iter=10)
exrcm = uk.fit_exrcm(X, 2, init=centers, alpha=1.1, beta=0., p=1.,
                   backend="numpy", max_iter=10)
membership, upper = uk.assign_rcm(X, rcm.centers, backend="numpy")
print(rcm.memberships, exrcm.memberships)
print(membership, upper)

rmcm = uk.fit_rmcm(X, 2, delta=1.5, init=centers, threads=1)
print(rmcm.memberships)
```

`fit_rmcm_numpy` provides the NumPy reference entry point. To reuse an RMCM
neighborhood graph, call `graph = uk.prepare_rmcm(X, delta=1.5)`, then
`graph.fit(2, init=centers, threads=1)`. This factory returns `PreparedRMCM`;
fits return `RMCMResult`.

### Self-organizing maps

```python
online = uk.som(X, grid_shape=(4, 3), epochs=2, random_state=1)
batch = uk.som_batch(X, grid_shape=(4, 3), epochs=2, random_state=1)
print(online["embedding"], batch["embedding"])

grid = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
som_olp = uk.fit_som_olp(X, grid, gamma=.1, lam=1., max_iters=3, policy=policy)
print(som_olp["V"])                    # Sample positions on the supplied grid
```

For explicit SOM-OLP initialization, use
`W0, P0 = uk.initialize_som_olp(X, grid, lam=1., policy=policy)` and
`uk.run_som_olp(X, grid, W0, P0, gamma=.1, lam=1., max_iters=3, policy=policy)`.
For repeated fits, use `prepared = uk.PreparedSOM(X, threads=1)` and
`prepared.fit(grid, gamma=.1, lam=1., max_iters=3)`.
See [SOM.md](SOM.md) for online and batch schedules, state and result contracts.

### Evaluation metrics

```python
quality = uk.joint_quality(X, som_olp["V"], ks=[1, 2], backend="numpy", policy=policy)
for score in quality:
    print(score.k, score.trustworthiness, score.continuity)

truth = np.array([0, 0, 0, 1, 1, 1])
predicted = np.array([0, 0, 1, 1, 2, 2])
print(uk.adjusted_rand_score(truth, predicted))
print(uk.adjusted_mutual_info_score(truth, predicted, average_method="arithmetic"))
print(uk.adjusted_scores(truth, predicted))  # {"ari": ..., "ami": ...}
```

For parameter search, see [OPTIMIZATION.md](OPTIMIZATION.md). The complete
[public-API check](../examples/python/all_methods.py) also covers the reference
entry points, preparation helpers, result types and optional Numba.

## Dependency compatibility

Install with one of the supplied tested constraint files. They preserve the two
verified stacks; the unconstrained declared minimum versions are not a support
matrix. In particular, old scikit-learn releases without a NumPy upper bound can
be combined by a resolver with incompatible NumPy 2.x. A general dependency-support
floor has not been validated. Do not infer that a successful unconstrained
resolution establishes runtime compatibility.

## Extreme-range FCM

The finite-m FCM equations and default membership-only stopping rule are
unchanged. Ordinary `1.0001 <= m <= 32` runs retain the existing vectorized/compiled
kernels. Large or near-one fuzzifiers, erased initialization/coordinate values, unsafe
squared distances, or an unrepresentable objective activate a shared
scaled/log-domain fallback. A late range failure restarts from the original
initialization. NumPy, SciPy, BLAS and both Numba selections use that same
fallback; requesting a Numba backend still requires Numba to be installed.

The fallback keeps relative center log-weights derived from distances, not from
rounded public memberships. It removes the common `-m*log(K)` analytically,
uses `expm1`/`log1p` for near-uniform weights, and normalizes each cluster before
its weighted mean. Exact coordinate coincidences split memberships only among
exactly coincident centers; a tiny nonzero norm is never made into a zero.
Scaled squared-norm mantissas/exponents handle distances below/above float64's
square range. Exceptional near-one fuzzifiers additionally preserve low parts
of squared norms to avoid amplifying a rounded distance tie. Compensated means
and guarded weighted-product recovery protect large/mixed-sign and tiny
centroid contributions. This is finite-m stabilization, not uniformization,
clipping, or an m-to-infinity approximation.

Exceptional results add `numerical_diagnostics` without changing existing
result names or their types. Important fields are:

- `arithmetic`, `trigger`, `requested_backend`, `effective_backend`: indicate
  that the shared `scipy_scaled_log` fallback was used
- `center_relative_delta`: largest coordinate displacement in the last center
  update divided by the maximum absolute working-data coordinate; zero for
  all-zero data. A lossless translation is used when available
- `membership_convergence_only`: membership tolerance was met while the above
  center displacement remained at least `tol`. `converged` still means exactly
  the historical membership test; it does not certify stationary centers
- `membership_resolution_lost_rows`: nonuniform positive-distance memberships
  rounded to a uniform public row. Internal center weights retain their
  distance-derived differences. Labels still use the public membership argmax
- `membership_frame='working_coordinates'` and `published_centers_rounded`:
  memberships were calculated at the working-coordinate centers. Restoring a
  large origin can round those centers; in that case a fresh membership update
  at the published center coordinates can differ substantially. U is not
  silently recalculated, because that would change the iteration/stop contract
- `objective_status`: `finite`, `exact_zero`, `underflow` or `overflow`.
  A positive objective outside float64's scalar range is returned as 0 or
  infinity, explicitly paired with `log_objective`; it is not clipped and does
  not invalidate finite centers/memberships. The logarithm itself can be
  negative infinity if its magnitude is outside float64's range
- `recovered_distance_pairs`: number of exceptional pair evaluations, including
  objective evaluations. With `return_history=True`, diagnostics also includes
  `log_objective_history`

The returned objective remains evaluated at the published rounded center and
membership pair. `tol=0` remains the existing way to request exactly `max_iter`
updates; inspect the center diagnostic when working with very large m. There
is no new stopping parameter or silent change to `converged`. A separate call
initialized from public rounded memberships cannot reconstruct lost log-weight
information from an earlier call. The fallback uses more computation and O(NK)
working memory; no extreme-path speedup is claimed. It does not supply arbitrary
precision or promise identical results across all floating-point environments.

Near-coincident centers can turn tiny center changes into large membership
changes. Stabilization does not guarantee that every membership is closer to an
arbitrary-precision trajectory. Returned-pair consistency and full-trajectory
accuracy are distinct: restoring a large origin can round the published centers
while memberships retain the working-coordinate update. Inspect the diagnostics.

The [SOM-OLP numerical limits](NUMERICAL_LIMITS.md) describe its exceptional-range
contract. [Parameter optimization](OPTIMIZATION.md) uses only the standard
library; importing that API does not load the scientific dependencies.

## Online SOM and BatchSOM

Use `ubukit.som(X, epochs=10)` for online updates and
`ubukit.som_batch(X, epochs=10)` for batch updates (BatchSOM), holding each
sample's best-matching unit (BMU) fixed throughout an epoch.
Both default to a 16×16 rectangular map with arbitrary input dimension.
Detailed API, examples, limits and sources: [SOM.md](SOM.md).

The separately checked SOM stack uses NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0,
and threadpoolctl 3.7.0 without Numba. Use `constraints/constraints-som-verified.txt` from this `python/` directory. These constraints apply to that stack; they do not qualify every resolver outcome.
