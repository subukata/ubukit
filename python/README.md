# UbuKit for Python

Clustering, self-organizing maps, evaluation metrics and parameter search.
This alpha uses distribution and import name `ubukit`, version `0.1.0a3`.
Project contributions use the MIT License; `LICENSE-SCOPE.txt` describes retained
third-party terms. License and notice files accompany both wheel and source archive.

Start with [installation](#installation-boundary),
[clustering and metric examples](#core-clustering-and-metric-examples),
[online and batch self-organizing maps (SOM)](#online-som-and-batchsom), or
`OPTIMIZATION.md` in the source archive.

## Installation boundary

Use a **new virtual environment** when moving from earlier previews. The old
`ubukit-bundled-local-preview` distribution installed overlapping top-level
modules; upgrading or uninstalling it in a mixed environment can remove files
another distribution owns. `pip check` alone does not detect file-owner collisions.

At source preparation on 2026-10-03, this alpha.3 candidate had not completed
fresh exact-artifact qualification or initial registry publication. This is a
dated preparation record; check the maintainer’s release record for later results
and exact archive hashes. Install a supplied local wheel with the CPython 3.12 base stack
used for the separate alpha.2 qualification:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install ./ubukit-0.1.0a3-py3-none-any.whl "numpy==2.3.5" "scipy==1.17.0" "scikit-learn==1.8.0" "threadpoolctl==3.6.0"
.venv/bin/python -c "import ubukit; print(ubukit.__version__)"
```

Use the actual local path to your wheel. These commands do not require a registry
release. Python >=3.10 is declared; the pins above are not a tested matrix for
every Python version or platform. Alpha.2 frozen wheel/sdist artifacts passed the
defined checks on Linux X64, Windows X64 and macOS ARM64. The extended full Python
regression was Linux-only, and optional Numba was absent on every OS. Those results
do not certify this changed alpha.3 candidate. On Windows use `Scripts/python.exe`.

A repository checkout additionally provides `python/tools/check_install_environment.py`.
Run it with the target environment's Python before installation. This read-only
guard blocks foreign `ubukit` ownership/imports and legacy aggregate layouts;
unrelated standalone module names do not by themselves block installation.
Stop on a conflict. The guard does not remove or repair packages and is not
included in the wheel or source archive.

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
needs it. Keep compiler caches private to a trusted account and never reuse an
attacker-writable cache. No global module aliases recreate old names.

## Results: samples first

| Family | Entry point | Result contract |
| --- | --- | --- |
| k-means: one cluster per sample | `fit_kmeans(X, init, ...)` | dict; labels `(N,)` |
| Fuzzy c-means (FCM): degrees of cluster membership | `fit_fcm(X, n_clusters, ...)` | dict; membership `(N,K)` |
| Entropy-regularized fuzzy c-means: temperature-controlled memberships | `fit_entropy_fcm(X, n_clusters, tau=1.0, ...)` | dict; membership `(N,K)` |
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

Existing local correctness evidence covers Linux x86-64 / CPython 3.12.
Other operating systems, architectures and Python versions, including the
declared minimum, are not a tested support matrix. Optional Numba requires
separate qualification for this alpha. Existing acceleration
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

`PreparedData.X`, `feature_bounds()`, `centered()`, and `norms()` return
read-only array views with independent headers and shared snapshot/cache bytes.
Array and bounds-tuple object identities are not guaranteed: changing a
returned array's shape or dtype does not change the prepared snapshot or later
results. Repeated `prepare(prepared)` and cached `as_dtype()` calls still reuse `PreparedData`
objects. Use an array's `.copy()` when you need writable values.

### Entropy-regularized fuzzy c-means

```python
efcm = uk.fit_entropy_fcm(X, 2, tau=0.5, random_state=4, return_history=True)
print(efcm["membership"], efcm["objective"])
```

The objective is `sum(u * squared_distance) + tau * sum(u * log(u))`,
with nonnegative memberships summing to one per sample and `0*log(0)=0`.
Centers are weighted by `u`. There is no fuzzifier `m`.

`X` must be a nonempty finite real `(N,D)` matrix. `tau` must remain finite and
positive after float64 conversion; larger values produce softer memberships.
`init` accepts an `(N,K)` nonnegative membership matrix with positive row sums;
it is normalized on an owned copy and can supply `K` when `n_clusters` is omitted.
Otherwise `random_state` initializes memberships with NumPy's generator.
`max_iter=300`, `tol=1e-5`, `backend="numpy"`, and `return_history=False` are defaults.
Only `backend="numpy"` is supported. The previous `backend="reference"` now
raises `ValueError`. EFCM uses ordinary float64 arithmetic throughout, with no
arbitrary-integer, higher-precision, or exact-reference fallback. Distances are
rounded before minimum subtraction: a small gap on top of a large common cost
can be lost, producing a tie. Weighted-mean and objective cancellation can lose
small residuals, and subnormal terms can underflow. Rescale data and temperature
when appropriate; scaling cannot restore information already rounded away.

The result contains owned `centers (K,D)`, `membership (N,K)`, `labels (N,)`,
`objective`, `fpc`, `n_iter`, `converged`, `delta`, `tau`, `backend`, and
`numerical_diagnostics`; `objective_history` is added when requested.
Each iteration updates centers from the previous membership, then membership
from the returned centers. The objective evaluates that returned pair.
Empty clusters retain their previous center, initially the data mean.
Convergence means the absolute Frobenius membership change is below `tol`;
`tol=0` runs exactly `max_iter`, and membership convergence alone does not certify
stationary centers. Nonfinite center, distance, or objective intermediates raise
`ValueError` instead of returning an invented or extended-range result. Successful
results report `arithmetic="float64"`, `used_reference_fallback=False`, and
`objective_status="finite"`; sign and log diagnostics describe only the rounded
float64 objective. They do not distinguish exact zero from underflow or
cancellation to zero. The membership update is a softmax of
negative squared distance divided by `tau`, with the minimum distance subtracted
before division. If coordinates scale by `s`, scale `tau` by `s**2` to preserve
the mathematical memberships. Positive `tau` does not force hard assignments at
zero distance; tiny float64 memberships can still underflow to zero.

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
See `SOM.md` in the source archive for online and batch schedules, state and result contracts.

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

For parameter search, see `OPTIMIZATION.md` in the source archive. The repository
also includes `examples/python/all_methods.py`, covering reference entry points,
preparation helpers, result types and optional Numba.

## Dependency compatibility

Use the pinned base stack in the installation example. Repository constraint
files preserve separately checked historical stacks; they are not bundled in
the package archives. Unconstrained declared minimum versions are not a support
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

The source archive includes `NUMERICAL_LIMITS.md` for SOM-OLP's exceptional-range
contract and `OPTIMIZATION.md` for parameter search. The optimization API uses
only the standard library; importing it does not load scientific dependencies.

## Online SOM and BatchSOM

Use `ubukit.som(X, epochs=10)` for online updates and
`ubukit.som_batch(X, epochs=10)` for batch updates (BatchSOM), holding each
sample's best-matching unit (BMU) fixed throughout an epoch.
Both default to a 16×16 rectangular map with arbitrary input dimension.
Detailed API, examples, limits and sources are in `SOM.md` in the source archive.

The separately checked SOM stack uses NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0,
and threadpoolctl 3.7.0 without Numba. The repository records it in
`python/constraints/constraints-som-verified.txt`. Those pins apply to that stack;
they do not qualify every resolver outcome.
