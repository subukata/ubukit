# Changelog

While UbuKit is 0.x, minor versions may change numerical results and the API;
every such change is listed under "Changed results" or "Changed API". Patch
versions keep the API and change results only to fix bugs, listed under
"Fixed".

## (unreleased)

### Changed API

- Python: the internal modules have an underscore (`ubukit._cluster`,
  `_som`, `_metrics`, `_tpe`); `ubukit`, `ubukit.steps`, `ubukit.apply` and
  `ubukit.steps.apply` are the only public modules. `from ubukit.cluster
  import fcm` gave the generator, not the function `ubukit.fcm`; import
  every name from the public modules. These paths were never promised, so
  no alias is kept.
- `Result.converged` is replaced by `stop_reason` (`stopReason` in
  JavaScript), the stopping rule that holds at the result's iteration:
  `"tol"` (no prototype coordinate moved more than `tol` times the RMS
  radius of the data), `"schedule"` (`som` and `batch_som`: every epoch has
  run) or `"limit"` (`max_iter` reached without the tolerance), and `None`
  (`null`) in a per-iteration result where none holds yet. `converged` was
  also true when a map's schedule completed, after even one epoch, and false
  in both other cases. Where code read `r.converged`, read
  `r.stop_reason == "tol"`, or `r.stop_reason in {"tol", "schedule"}` for
  the old meaning across all methods (not `if r.stop_reason`, which is true
  for `"limit"` too).

### Added

- `Neighborhood(X)` holds the pairwise distances of `X`, computed once, and
  `trustworthiness` and `continuity` take it in place of `X` or `Y` with the
  same values, so scoring many embeddings of one `X`, as a search does, no
  longer computes the distances of `X` in every call.
- `trustworthiness` and `continuity` take a list of `k` and return one value
  per entry, in its order (an array; a `Float64Array` in JavaScript), for
  one count of ranks.
- `apply` (with its generators in `steps.apply`) applies trained prototypes
  to new data for every method, the prototypes fixed and kept exactly as
  given: `ub.apply.fcm(X_new, centers, m=...)`,
  `ub.apply.som_olp(X_new, centers, grid, lam=..., gamma=...)`, and so on,
  with the parameters the prototypes were trained with. Each method assigns
  once (`stop_reason` `"schedule"`); SOM-OLP iterates its memberships until
  they settle (`"tol"`, or `"limit"` after `max_iter`, 1000 by default). Any
  number of points, including fewer than the prototypes.
- The Results of `som`, `batch_som` and `som_olp` hold `unit_coordinates`
  (`unitCoordinates`): the (K, Q) coordinates of the units as the run used
  them, not normalized, `centers[j]` being the prototype of unit j. `None`
  (`null`) for clusterings.

### Fixed

- Python: data must hold numbers, as in JavaScript since 0.1.1. Booleans
  and numeric strings were converted, the real part of complex numbers
  kept, and `None` refused only as a missing value; they now raise
  `TypeError`.
- Python: `som` and `som_olp` read the unit coordinates from the array
  passed as `grid`, so changing that array during a run changed the
  iterations that followed, and the Results that a saved Progress of an
  earlier iteration built afterwards. The run now copies it; JavaScript
  already did.
- A numeric range too narrow to search, such as
  `loguniform(1000, 1000.0000000000001)`, whose logarithms coincide, raises
  an error when it is made; once TPE used its model, it gave `NaN` in
  JavaScript and `ZeroDivisionError` in Python. A non-finite objective
  value reads "value must be a finite number" (it read ">= -inf").
- JavaScript: a `TPE` keeps checked copies of the dimensions of its space,
  and `uniform`, `loguniform`, `integer` and `choice` return frozen objects,
  as Python's are: changing a dimension after the `TPE` was made changed
  its proposals, and a dimension written by hand was not checked.
- Python: `True` is not the option `1` of a `choice` (in Python `True == 1`),
  as JavaScript compares options with `===`.
- Python: a search space kept in a variable type-checks. Type checkers
  accepted `TPE` and `minimize` only with the space written as a dict in
  the call: a variable holding numeric dimensions only, or numeric and
  categorical ones, was rejected. Both now take any mapping of dimensions,
  at run time too.
- Python: the type declarations of `minimize` name the options it passes
  to `TPE` (`n_startup`, `n_candidates`, `gamma`), as JavaScript's do since
  0.1.1, so a type checker reports a misspelled or mistyped option.

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
