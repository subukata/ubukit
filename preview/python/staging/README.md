# UbuKit: private accelerated sample-major membership candidate

**PRIVATE PREVIEW. Not published to PyPI, TestPyPI, or any other registry.**
The provisional distribution name `ubukit-bundled-local-preview` is not an
approved release name. No project-wide license has been selected. The retained
SOM third-party MIT notice applies to its original/derived code, not the whole
project. No published-release or general performance claim is made.

This private dev4 candidate is an additive copy of the validated Python dev3
bundle. It retains the 31 prior exports and adds seven traditional SOM exports
(38 total): `som`, `som_batch`, their `fit_` aliases, both initializers, and
`SOMState`. All existing numerical source files remain byte-identical to dev3.
Only the two lazy export facades, version/package documentation, and new SOM
module are changed. The prior dev3 remains untouched. See [SOM.md](SOM.md) for
formulas, schedules, per-sample/per-epoch semantics, and numerical limits.

`SOURCE_MANIFEST.json` compares this candidate to dev3. Validation/measurement
logs in the companion distinguish this addition's tests from prior historical
checks. This is version `0.0.0.dev4`, not an approved public release.

## Installation boundary

Use a **new virtual environment**. Do not install this alongside
`portable-accel-restart`, `ubukit-fcm`, `ubukit-exrcm-research`, `ubukit-rmcm`, or
the prior `ubukit-local-preview` aggregate. Their RECORDs own overlapping paths;
uninstalling either distribution can delete the other's runtime files.
`pip check` does not detect this conflict. The review companion's
`tools/check_install_environment.py` is a pre-install check, not an automatic
pip hook, and intentionally performs no uninstall or repair.

The legacy top-level module `_numba_kernel` is retained because
`rough_cmeans` imports it directly. An unrelated distribution or local module
using that name can shadow/collide with the optional RCM backend. Relocating it
requires a separate compatibility review; this prototype does not relocate it.
The same environment-isolation rule applies to every preserved top-level name.

```sh
python -m venv .venv
.venv/bin/python -m pip install -c /path/to/constraints-som-verified.txt /path/to/ubukit_bundled_local_preview-0.0.0.dev4-py3-none-any.whl
# Optional: install the same local wheel with the [numba] extra.
```

A clean environment has only one UbuKit distribution owning these files. All
original Python imports remain valid:

```python
import ubukit
import portable_accel
import ubukit_fcm
import rough_cmeans
import ubukit_rmcm
```

The `ubukit` root import is lazy. It does not import NumPy or Numba. Its 38
exports comprise 35 aliases and three thin adapters: `fit_rcm`,
`fit_exrcm`, and `assign_rcm`. Accessing one of these adapters loads the numerical
runtime; importing `ubukit` alone still does not.

## Facade membership axes: samples first

| Family | Facade entry point | Facade result contract |
| --- | --- | --- |
| k-means | `fit_kmeans(X, initial_centers, ...)` | dict; labels `(N,)` |
| FCM | `fit_fcm(X, n_clusters, ...)` | dict; membership `(N,K)` |
| RCM | `fit_rcm(X, n_clusters, ...)` | result adapter; memberships `(N,K)` |
| ExRCM | `fit_exrcm(X, n_clusters, ...)` | result adapter; memberships `(N,K)` |
| RMCM | `fit_rmcm(X, n_clusters, delta=..., ...)` | result object; memberships `(N,K)` |
| Online / Batch SOM | `som(X, ...)`, `som_batch(X, ...)` | dict; centers `(M,D)`, labels `(N,)`, embedding `(N,2)` |
| SOM-OLP | `fit_som_olp(X, grid, ...)` | dict; `V` `(N,latentD)`, `P` `(N,M)` |
| Neighborhood quality | `joint_quality(X, embedding, ks=..., ...)` | list of quality records |

`N` means samples and `K` means clusters. In the new facade, every clustering
membership matrix is sample-major. For RCM/ExRCM, both `.memberships` (float64)
and `.upper_memberships` (bool) have shape `(N,K)`. `assign_rcm` still returns a
2-tuple, now with both arrays `(N,K)`. Rows of normalized memberships sum to 1.

FCM remains a dict with the singular key `["membership"]`. RMCM keeps its result
object and plural `.memberships` field. Their container types and membership axes retain the sample-major convention.
The corrections do not unify container types or field names. `PreparedSOM` and
`PreparedRMCM` keep their public types; see the correction report for validation fixes.

The new RCM/ExRCM result is a thin attribute-forwarding adapter, not a
`rough_cmeans.RoughCMeansResult` or dataclass. All non-membership metadata
(including centers, convergence/cycle information, initialization indices and
backend) is forwarded without changing values. Other arrays retain their
legacy ownership semantics. Each adapted membership array is a writable,
C-contiguous, separately owned transpose-copy, even when `N == K` or an axis
has length 1. It is never a view into the legacy output. `return_memberships=False`
still returns `None` for both membership fields. The private legacy result is
retained so metadata is available; the copies add O(NK) time and storage
(approximately 9NK bytes for float64 memberships plus bool upper memberships).

All original imports, including `rough_cmeans.fit_rcm`, `rough_cmeans.fit_exrcm`
and `rough_cmeans.assign`, retain `(K,N)` arrays and their original return types.
Use those entry points when exact legacy object identity/types or axes are
required. The prior single-wheel review remains unchanged in its own directory.

The established public entrypoints and sample-major membership axes remain.
See the companion correction report for the specific numerical and input-validation
changes. The runtime archive does not include JavaScript, datasets, benchmarks,
tests or logs.

## Scope and release gates

Local packaging tests are separate from numerical release approval. This private
candidate has no public release authorization. Multi-OS/Python and browser
checks, minimum dependency versions, final name/version policy, project license,
final public API/dependency policy, and explicit publication approval remain open.
This is a Python-only package; the matching JavaScript dev5 source and its
separate validation suite are alongside it in this repository.

Declared minimum versions are resolver metadata, not a tested version matrix.
See the companion integration report for the exact tested environment and
passed, skipped, excluded, and unrun checks. Existing acceleration measurements
did not include the facade transpose-copy cost; no end-to-end API speedup is
claimed here.

## ARI and AMI additions

`ubukit.adjusted_rand_score`, `ubukit.adjusted_mutual_info_score`, and
`ubukit.adjusted_scores` are lazy aliases of the unchanged functions in the
preserved top-level module `external_metrics`. The joint helper returns
`{"ari": ..., "ami": ...}`. AMI accepts `average_method="arithmetic"` (default),
`"geometric"`, `"min"`, or `"max"`, and `backend="numpy"` (default) or `"numba"`.

The complete-support mathematical definition is unchanged. Ordinary-path
NumPy/Numba AMI is floating-point arithmetic, not a bitwise-equality guarantee.
Singular and conservatively screened ill-conditioned/high-cluster-density
cases call the installed scikit-learn implementation to preserve its numerical
convention. These calls can be slower. Test-panel tolerance is not a universal
error bound. Optional Numba is imported only when its path is needed.

The generic top-level names `external_metrics` and `_external_metrics_numba`
are retained to preserve the frozen absolute import. They add potential name
collisions; the supplied read-only preflight checks both names. The facade
avoids duplicating their implementation. No user environment is repaired or
uninstalled automatically.

Exact external-metrics attribution is in `NOTICE-EXTERNAL-METRICS.txt` and
`THIRD_PARTY_LICENSES.txt`. References to `NOTICE.txt` within the unchanged BSD
notice refer to the original external-metrics notice now supplied under that
longer filename. Its `source_audit.json` is included in the verification
companion, under `verification/external_provenance/`. No project license is
chosen. Historical performance results are not new measurements of this wheel.

## Tested environment and installation example

This private trial was checked on Linux x86_64 with CPython 3.12.14 only.
The inherited dev3 integration environments used NumPy 2.3.5, SciPy 1.17.0 and threadpoolctl
3.6.0. Both scikit-learn 1.7.2 with optional Numba 0.63.1, and scikit-learn
1.8.0 with optional Numba 0.67.0 were checked. No Windows/macOS, ARM, browser,
or minimum-supported-version claim follows from these tests. Matching historical
version numbers does not carry the old benchmark results over to this facade.

From the repository `preview/python/` directory after building local artifacts,
run the following in a fresh
environment. The preflight is read-only and is not a pip install hook. Stop on
a blocked result; do not co-install or automatically uninstall another owner.

```sh
python3 -m venv .venv-ubukit-preview
.venv-ubukit-preview/bin/python -I -B tools/check_install_environment.py
.venv-ubukit-preview/bin/python -m pip install -c constraints-som-verified.txt artifacts/ubukit_bundled_local_preview-0.0.0.dev4-py3-none-any.whl
.venv-ubukit-preview/bin/python -I -B examples/all_methods.py
# Optional Numba in this same isolated environment:
.venv-ubukit-preview/bin/python -m pip install -c constraints-final-stack.txt 'artifacts/ubukit_bundled_local_preview-0.0.0.dev4-py3-none-any.whl[numba]'
.venv-ubukit-preview/bin/python -I -B examples/all_methods.py --with-numba
```

Install the local wheel with the supplied companion constraints for this private
trial. They pin the tested stack rather than claiming all resolver outcomes
were validated. The source archive includes this README; the example script,
preflight, constraints and full verification harness are in the repository
`preview/python/` directory. Optional Numba checks above describe a runnable
workflow; the final dev4 integration replay uses the base stack without Numba.

## Runnable examples for each facade entrypoint

The following complete example is also `examples/all_methods.py`. It covers
the seven original families, three ARI/AMI entrypoints, preparation helpers,
and the public result/preparation types. `PreparedRMCM` is constructed with
`prepare_rmcm`, as its API recommends. Set optional Numba only when installed.

```python
"""Executable README examples for the private UbuKit dev3 trial.

Run with the installed virtualenv: python -I -B examples/all_methods.py
Add --with-numba only after installing this local wheel's [numba] extra.
"""
import json
import sys
import numpy as np
import ubukit as uk

X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
centers = X[[0, 3]].copy()
policy = uk.ExecutionPolicy(threads=1)

# Owned reusable data, via either constructor or convenience function.
snapshot = uk.PreparedData(X)
data = uk.prepare(X)
assert isinstance(data, uk.PreparedData)
np.testing.assert_array_equal(snapshot.X, data.X)

# k-means: labels (N,), centers (K,D).
km = uk.fit_kmeans(data, centers, backend="numpy", max_iter=10, policy=policy)
assert km["labels"].shape == (6,)

# Fuzzy c-means: the dict uses the singular key "membership", shape (N,K).
fcm = uk.fit_fcm(X, 2, random_state=4, backend="scipy", max_iter=10, threads=1)
fcm_reference = uk.fit_fcm_numpy(X, 2, random_state=4, max_iter=10, threads=1)
assert fcm["membership"].shape == fcm_reference["membership"].shape == (6, 2)

# Rough c-means and extended rough c-means: facade arrays are (N,K).
rcm = uk.fit_rcm(X, 2, init=centers, backend="numpy", max_iter=10)
exrcm = uk.fit_exrcm(X, 2, init=centers, alpha=1.1, beta=0., p=1.,
                   backend="numpy", max_iter=10)
membership, upper = uk.assign_rcm(X, rcm.centers, backend="numpy")
assert membership.shape == upper.shape == exrcm.memberships.shape == (6, 2)

# Rough membership c-means: one-shot, NumPy reference, and prepared graph.
rmcm = uk.fit_rmcm(X, 2, delta=1.5, init=centers, threads=1)
rmcm_reference = uk.fit_rmcm_numpy(X, 2, delta=1.5, init=centers, threads=1)
graph = uk.prepare_rmcm(X, delta=1.5)
assert isinstance(graph, uk.PreparedRMCM)  # Use the factory, not private graph internals.
rmcm_repeated = graph.fit(2, init=centers, threads=1)
assert isinstance(rmcm, uk.RMCMResult)
assert rmcm.memberships.shape == rmcm_reference.memberships.shape == rmcm_repeated.memberships.shape == (6, 2)

# SOM-OLP: one-shot, explicit initialization/run, and reusable preparation.
grid = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
som = uk.fit_som_olp(X, grid, gamma=.1, lam=1., max_iters=3, policy=policy)
W0, P0 = uk.initialize_som_olp(X, grid, lam=1., policy=policy)
som_from_init = uk.run_som_olp(X, grid, W0, P0, gamma=.1, lam=1.,
                              max_iters=3, policy=policy)
prepared_som = uk.PreparedSOM(X, threads=1)
som_repeated = prepared_som.fit(grid, gamma=.1, lam=1., max_iters=3)
assert som["V"].shape == som_from_init["V"].shape == som_repeated["V"].shape == (6, 2)

# Neighborhood quality: records contain trustworthiness and continuity.
Q = np.random.default_rng(73).normal(size=(12, 3))
quality = uk.joint_quality(Q, Q, ks=[1, 2], backend="numpy", policy=policy)
assert all(q.trustworthiness == 1 and q.continuity == 1 for q in quality)

# ARI, AMI, and a joint helper, all accepting raw label arrays.
truth = np.array([0, 0, 0, 1, 1, 1])
predicted = np.array([0, 0, 1, 1, 2, 2])
ari = uk.adjusted_rand_score(truth, predicted)
ami = uk.adjusted_mutual_info_score(truth, predicted, average_method="arithmetic")
scores = uk.adjusted_scores(truth, predicted)
assert scores == {"ari": ari, "ami": ami}
for method in ["arithmetic", "geometric", "min", "max"]:
    assert np.isfinite(uk.adjusted_mutual_info_score(truth, predicted, average_method=method))
if "--with-numba" in sys.argv:
    compiled = uk.adjusted_scores(truth, predicted, backend="numba")
    np.testing.assert_allclose(list(compiled.values()), list(scores.values()), atol=1e-8, rtol=0)

print(json.dumps({"status": "passed", "version": uk.__version__,
                  "facade_exports": len(uk.__all__), "optional_numba_example": "--with-numba" in sys.argv,
                  "ari": ari, "ami": ami}))
```

## Dependency safety boundary for this private correction

Install with one of the supplied tested constraint files. They preserve the two
verified stacks; the unconstrained declared minimum versions are not a support
matrix. In particular, old scikit-learn releases without a NumPy upper bound can
be combined by a resolver with incompatible NumPy 2.x. This correction does not
silently change the historical support floor or cap NumPy below 2, which would
discard the tested stacks. Choosing and testing a general dependency-support
floor remains a release gate. Do not infer that a successful unconstrained
resolution establishes runtime compatibility.

## Extreme-input acceptance

The earlier dev2 centroid correction exposed a squared-distance-underflow
rejection for an m=1000, D=128 fixture. The guarded finite-m implementation
below now supports the original and row-zero-shift versions of that fixture on
all five FCM backend selections. This supersedes the rejection-only policy; it
is not a promise of universal float64 accuracy or an unchanged extreme-input
trajectory. See the companion numerical report for verification and limits.

### Guarded extreme-range FCM arithmetic

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

Extreme-path trajectories are intentionally not bit-compatible with the prior
rounded-membership implementation. Near-coincident centers can turn tiny center
changes into large membership changes. The stabilization is not a guarantee
that every final membership is closer to a fully arbitrary-precision trajectory;
a 400-digit reference panel found improved centers but not uniformly improved
membership accuracy. Returned-pair consistency in that reference fixture and full-trajectory
accuracy are distinct checks. Membership/center consistency is not universal:
restoring a large origin can round the published centers while U retains the
working-coordinate update; the diagnostic flags this situation. See the accompanying numerical report and reference evidence.


## Combined private preview

The inherited dev3 implementation adds the optional standard-library-only optimization API. See OPTIMIZATION.md. The existing scientific dependencies are unchanged; importing the optimization facade does not require them. The bundled SOM-OLP exceptional-range contract is documented in NUMERICAL_LIMITS.md.

## Traditional SOM additions

Use `ubukit.som(X, epochs=10)` for online updates and
`ubukit.som_batch(X, epochs=10)` for true frozen-BMU batch updates.
Both default to a 16×16 rectangular map with arbitrary input dimension.
Detailed API, examples, limits and sources: [SOM.md](SOM.md).

This SOM addition was tested with NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0,
and threadpoolctl 3.7.0 without Numba. Use `constraints-som-verified.txt` from the repository
`preview/python/` directory. Earlier environment listings above describe inherited checks.
