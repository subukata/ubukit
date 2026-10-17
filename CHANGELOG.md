# Changelog

While UbuKit is 0.x, minor versions may change numerical results and the API;
every such change is listed under "Changed results" or "Changed API". Patch
versions keep the API and change results only to fix bugs, listed under
"Fixed".

## (unreleased)

### Changed API

- Python: the internal modules have an underscore (`ubukit._cluster`,
  `_som`, `_metrics`, `_tpe`); import every name from the public modules,
  `ubukit`, `ubukit.steps`, `ubukit.apply` and `ubukit.steps.apply`.
  `from ubukit.cluster import fcm` gave the generator, not `ubukit.fcm`.
  These paths were never promised, so no alias is kept.
- `Result.converged` is replaced by `stop_reason` (`stopReason`), the
  stopping rule that holds at the result's iteration: `"tol"` (no prototype
  coordinate moved more than `tol` times the RMS radius of the data),
  `"schedule"` (`som`, `batch_som`: every epoch has run) or `"limit"`
  (`max_iter` reached first), and `None` (`null`) in a per-iteration result
  where none holds yet. `converged` was also true once a map's schedule
  completed. Read `r.stop_reason == "tol"`, or
  `r.stop_reason in {"tol", "schedule"}` for the old meaning (not
  `if r.stop_reason`, which is true for `"limit"` too).
- `rmcm` takes `delta` by name, required, as `som_olp` takes `lam` and
  `gamma`: `ub.rmcm(X, k, delta=0.5)`, `ub.rmcm(X, k, { delta: 0.5 })` in
  JavaScript. It was the third positional argument.
- `TPE`'s `n_startup`, `n_candidates` and `gamma` (`nStartup`,
  `nCandidates`, `gamma`) are read-only: a value set later skipped the
  checks, and `n_candidates = 0` made the next proposal fail.

### Added

- `apply` (generators in `steps.apply`) applies trained prototypes to new
  data for every method, the prototypes fixed and kept exactly as given:
  `ub.apply.fcm(X_new, centers, m=...)`,
  `ub.apply.som_olp(X_new, centers, grid, lam=..., gamma=...)`, with the
  parameters they were trained with. Each method assigns once
  (`"schedule"`); SOM-OLP iterates its memberships until they settle
  (`"tol"`, or `"limit"` after `max_iter`, 1000 by default).
- `Neighborhood(X)` holds the pairwise distances of `X`, computed once;
  `trustworthiness` and `continuity` take it in place of `X` or `Y` with the
  same values, so a search scoring many embeddings of one `X` no longer
  recomputes them. Both also take a list of `k` and return one value per
  entry (an array; a `Float64Array` in JavaScript).
- The Results of `som`, `batch_som` and `som_olp` hold `unit_coordinates`
  (`unitCoordinates`): the (K, Q) coordinates of the units as the run used
  them, `centers[j]` being unit j's prototype; `None` (`null`) for
  clusterings.

### Fixed

- Results that held `NaN` or `-inf` yet reported `"tol"` are correct or
  raise an error. `som_olp` computes its latent costs from centered unit
  coordinates, so units far from the origin no longer overflow them through
  rounding (results move in their last digits), and refuses a `gamma` whose
  costs overflow (`gamma` times the squared extent of the units above
  1e300). A `pca_scale` whose start leaves the scale contract, and a `tau`
  or `lam` whose objective overflows, raise an error. JavaScript's `rcm`
  with a `p` near 1e-320 gives Python's limit.
- JavaScript: options that are not an object raise `TypeError`; a number
  read as no options, so `minimize(f, space, 20)`, Python's positional
  `n_trials`, ran 100 trials.
- Python: values must be numbers, as in JavaScript since 0.1.1: booleans
  (among numbers in lists too), numeric strings, complex numbers and `None`
  raise `TypeError`.
- JavaScript: labels must be numbers, bigints, strings or booleans (arrays
  or objects were compared by identity), and sparse arrays as rows, labels
  or a list of `k` are refused before anything is allocated for their
  length. Python's error for labels that do not sort, such as `[0, None]`,
  names them.
- `k`, a named `init` and `seed` are checked before any work on the data;
  Python refuses `seed=True`, which it took as 1.
- TPE: a range too narrow to search, such as
  `loguniform(1000, 1000.0000000000001)`, raises when made (it later gave
  `NaN` or `ZeroDivisionError`). JavaScript keeps checked copies of the
  dimensions, which are now frozen, as Python's are. Python's `choice` does
  not take `True` for `1`, and finds options such as arrays as `ask` gave
  them.
- Python: `som` and `som_olp` copy the unit coordinates passed as `grid`;
  changing that array during a run changed the run.
- JavaScript: rows of unit coordinates may be typed arrays, as `matrix`
  reads them.
- Types: a Python search space kept in a variable type-checks, and `TPE`
  accepts any mapping; `minimize` declares the options it passes to `TPE`.
  JavaScript's declarations accept typed-array rows and boolean or bigint
  labels, and make dimensions read-only.

## 0.1.1 (2026-10-07)

### Fixed

- JavaScript: rows given as arrays must hold numbers. `null` was read as 0,
  so a missing value entered the fit silently (Python rejects `None`), and
  booleans and numeric strings were converted.
- JavaScript: the TypeScript declarations of `minimize` accept the options
  of `TPE` (`seed`, `nStartup`, `nCandidates`, `gamma`), which it passes on;
  they allowed only `nTrials`.
- JavaScript: an unknown option throws `TypeError`, as an unknown keyword
  does in Python. It was ignored, so a misspelled option, such as the Python
  name `max_iter` for `maxIter`, left the default in force.
- JavaScript: the package exports the types `Progress` and `TPEResult`, as
  Python exports the classes.
- Python: the internal `stepwise` is no longer a name of the package; it was
  never in `__all__`.

## 0.1.0 (2026-10-06)

First release of the rewritten library, for Python (PyPI) and JavaScript
(npm): k-means, fuzzy c-means, entropy-regularized FCM, rough c-means
(RCM/ExRCM), rough membership c-means, online and batch SOM, SOM-OLP, ARI,
AMI, trustworthiness, continuity and multivariate TPE. Every fitting
function has a generator in `steps` that yields each iteration's Result and
accepts new data between iterations, so learning can follow data that move.
In Python, the online SOM, trustworthiness, continuity and AMI can run as
Numba kernels (`engine="numba"`, with `pip install "ubukit[numba]"`).
