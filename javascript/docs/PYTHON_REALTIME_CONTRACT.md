# Python parity and the update contract

This feature slice adds the retained session API to **JavaScript**. It does not
introduce or claim an equivalent Python `createSession`/`step` class. Existing
Python batch/prepared APIs remain unchanged. In particular, repeatedly invoking
Python `fit(..., max_iter=1)` is not presented as a complete persistent solver:
that would reset convergence/cycle histories and may reinitialize empty-cluster
state. Python stateful execution requires a separate implementation/validation
slice rather than a cosmetic wrapper.

## Existing correspondences

| JavaScript | Existing Python contract | Array layout |
|---|---|---|
| FCM `initMembership`, `m`, `maxIterations`, `tolerance` | `fit_fcm(X, init=U, m=..., max_iter=..., tol=...)` | U is N×K in both |
| RCM/ExRCM `initCenters`, `alpha`, `beta`, `p` | rough-c-means `fit(..., init=centers, alpha=..., beta=..., p=...)`; RCM fixes p=1 | JS U N×K; Python ExRCM U C×N |
| RMCM `prepareRMCM(X,{delta,...}).fit(...)` | `prepare_rmcm(X, delta, ...).fit(n_clusters, init=centers, ...)` | JS `membership`; Python result `memberships`, both N×K |
| SOM `initialPrototypes`, `initialMemberships` | explicit initial W and P in the Python prepared kernel | W M×D, P N×M, V N×Q |

The existing checked-in Python fixtures continue to test the historical JS
results. The new session tests additionally require exact complete-object
agreement between segmented and uninterrupted JavaScript runs at the same
explicit initialization. Together these preserve the established fixture parity
for the unchanged numerical contracts. This is not a claim of universal bitwise
Python/JavaScript parity: NumPy/BLAS reduction order, PCA implementations and
platform math can differ. Supply explicit initialization, compare with the
established tolerances, and compare identical stopping/update order.

Numerical random seeds do not synchronize NumPy's generator and JavaScript's
Mulberry32. SOM's Jacobi PCA is not NumPy/LAPACK SVD bit parity. Neighborhood
metrics use rooted-distance then sample-index ties; do not assume exact sklearn
tie-order parity on tied distances.

## Requirements for a future Python stateful API

These describe the validated JS behavior to preserve, not currently available
Python methods:

1. Keep the actual old/new membership, centers, workspace, convergence state and
   finite cycle history across steps. Segmenting unchanged inputs must reproduce
   one uninterrupted fit, including retained empty centers
2. At every update, atomically validate a candidate configuration. Drop any
   partial iteration and resume only from a complete compatible state. Changed
   X/order/N resets row memberships; unchanged D may retain centers. Changed D/K
   or incompatible grid invalidates model-shaped state
3. A changed FCM m permits a previous normalized U warm start, but resets the
   objective/history/convergence comparison. Retain empty-cluster centers
   explicitly; initializing from U alone in a new batch call may use the data
   mean instead
4. Reuse RMCM preparation only for identical X and delta/backend. New X or delta
   requires new exact neighborhoods and PᵀX/Pᵀ1. Reusing a graph after point motion
   changes the model and is not an exact acceleration
5. Preserve output timing. RMCM labels produced the returned centers, FCM
   centers use U_old while returned membership is U_new, and SOM V/W use the
   pre-update P. A detected RMCM cycle must never be described as convergence
6. Keep metric debounce/cancellation/content caching separate from iterative
   fitting. Expose copied snapshots and bounded memory explicitly

No Python publication, new Python distribution, or Python runtime measurement is
implied by this JavaScript feature slice. Existing Python performance evidence
must be reported separately from the JavaScript/Node timings.
