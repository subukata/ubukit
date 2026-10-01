# FCM comparator review

Research date: 2026-10-01. Source inspection only; no benchmark runs or imported third-party code.

## Recommendation and evidence

Use **scikit-fuzzy 0.5.0** as the primary external comparator, specifically `skfuzzy.cluster.cmeans`. It is a long-lived SciPy-oriented toolkit with public documentation, tests and releases since 2015. Its official repository snapshot displayed 877 stars and 295 forks. These are modest, observable interest indicators, not market-share proof. PyPI currently identifies 0.5.0, released 2024-08-22, as latest; do not call it rapidly maintained. Its license is BSD-3-Clause.

- Version/history: https://pypi.org/project/scikit-fuzzy/
- Project/reputation indicators: https://github.com/scikit-fuzzy/scikit-fuzzy
- License: https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/LICENSE.txt
- Tests: https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/cluster/tests/test_cmeans.py

Use **fuzzy-c-means 2.0.3**, imported as `fcmeans.FCM`, only as an optional second comparator. It is a dedicated implementation with a public citation, tests, release history since 2019, and MIT license. The repository snapshot displayed 193 stars and 44 forks. Latest PyPI release is 2026-09-13; recent release activity is verified, but broad adoption is not established.

- Version/history/license: https://pypi.org/project/fuzzy-c-means/
- Project: https://github.com/omadson/fuzzy-c-means

## scikit-fuzzy algorithm details

Data shape is `(features, samples)`; initial and returned membership shape is `(clusters, samples)`. Supply shared `init` for reproducibility: passing the same seed to different packages is insufficient. The default RNG is legacy NumPy, with a global seed side effect.

Each iteration normalizes old memberships, floors them to float64 epsilon, computes weighted centers with NumPy dot, gets Euclidean distances through SciPy cdist, floors distances, records the objective using old memberships, and updates memberships. Stopping is **absolute Frobenius membership change < error**. Version 0.5.0 executes up to exactly `maxiter` updates and returns the actual count. Older 0.4.2 online docs have `maxiter - 1`, fixed in 0.5.0.

Returned centers and objective are based on the preceding membership matrix. Returned membership is one update newer; centers are not recomputed at the end. Objective-history construction and final partition-coefficient computation are part of public API runtime; the latter forms U @ U.T.

- Release-pinned source: https://raw.githubusercontent.com/scikit-fuzzy/scikit-fuzzy/v0.5.0/skfuzzy/cluster/_cmeans.py
- Fix history: https://github.com/scikit-fuzzy/scikit-fuzzy/commits/master/

Normalization explicitly casts distances to float64, divides each sample's distances by its maximum, floors relative values at float64 epsilon, divides by the minimum for negative exponents, raises to `-2/(m-1)`, then normalizes. Thus coincidence handling approximates limiting zero-distance behavior, with tiny memberships outside coincident centers rather than exact zeros. Distances below epsilon relative to the largest are intentionally clipped. Default runs produce float64 memberships. Do not advertise a float32-versus-float64 speedup as equivalent precision.

- Normalization: https://raw.githubusercontent.com/scikit-fuzzy/scikit-fuzzy/v0.5.0/skfuzzy/cluster/normalize_columns.py

## fuzzy-c-means 2.0.3 details

The official PyPI source distribution was downloaded and its SHA256 verified against PyPI: `caded82238be65cae71b59a8a185bf0da00a09dc1e8cec9f8fee7bd7a6f8d8c6`. That distribution was inspected for comparison; its source files are not included in UbuKit.

Initialization is float64 `default_rng(seed).uniform((N,C))`, row-normalized. There is no public init argument. Generate this same initialization independently for the other implementations. Fit stops on absolute Frobenius membership change, contrary to its relative-center-change docstring. It allows max_iter 1–1000 and error >=1e-9, so normal construction cannot disable stopping with zero tolerance. It returns no iteration count or objective. Centers precede final membership by one update.

Membership uses inverse powers without clipping or an exact-zero branch; coincident points can produce NaNs. Euclidean distances create an N*C*D broadcast temporary. A float32 X does not create a float32 fit, because U initializes float64. The package requires NumPy >=1.21.1,<2, plus pydantic, tqdm, typer and tabulate; isolate this optional comparison rather than downgrade the project environment.

- Exact official distribution: https://files.pythonhosted.org/packages/b7/1e/1be787f5f90e25ac75d03d54f0f60d082f218e863b188aab30887a3dfe58/fuzzy_c_means-2.0.3.tar.gz

## Threading and fair measurement

Neither FCM API exposes a worker count. Both call NumPy linear algebra, so effective threading depends on the installed BLAS. NumPy documents BLAS thread controls and threadpoolctl: https://numpy.org/doc/stable/reference/global_state.html

Use the same float64 data, shared membership initialization, Euclidean metric, fuzzifier, stopping rule and output convention. For fixed-iteration tests use scikit-fuzzy with zero tolerance and record exact update count; do not silently patch fuzzy-c-means for a headline comparison. Separately report convergence time, observed iterations, and numerical quality. Pin all versions and record threadpool metadata; compare single-threaded first, then separately describe optional multicore behavior. Label end-to-end public-API timing separately from a kernel benchmark. Include JIT compilation as a cold-start result and exclude it only from explicitly warmed steady-state timings. Recompute a common objective at each returned center/membership pair when assessing quality, because library objective histories use a different state.

Test coincident points, identical data, duplicate centers, near-zero distances, m near 1, non-default m, and truncated runs independently. A stricter exact-zero policy may legitimately disagree slightly with scikit-fuzzy's clipping policy and should be documented instead of forced to numerical identity.
