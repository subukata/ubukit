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
- `rmcm` and `apply.rmcm` take `delta`, the neighborhood radius, by name and
  require it, as `som_olp` takes `lam` and `gamma`:
  `ub.rmcm(X, k, delta=0.5)` and `ub.apply.rmcm(X, centers, delta=0.5)` in
  Python, `ub.rmcm(X, k, { delta: 0.5 })` in JavaScript. `rmcm` took it as
  its third positional argument, the only method parameter passed by
  position; `apply.rmcm` is new.
- The settings of a `TPE`, `n_startup`, `n_candidates` and `gamma`
  (`nStartup`, `nCandidates`, `gamma`), are read-only. They were plain
  attributes, so a value set after construction skipped the checks:
  `n_candidates = 0` made the next proposal fail.

### Added

- `Neighborhood(X)` holds the pairwise distances of `X`, computed once, and
  `trustworthiness` and `continuity` take it in place of `X` or `Y` with the
  same values, so scoring many embeddings of one `X`, as a search does, no
  longer computes the distances of `X` in every call.
- `trustworthiness` and `continuity` take a list of `k`, of any length, and
  return one value per entry, in its order (an array; a `Float64Array` in
  JavaScript), for one count of ranks.
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

- JavaScript: options that are not an object throw `TypeError`. A number
  or boolean in their place has no keys, so it read as no options:
  `minimize(f, space, 20)`, Python's positional `n_trials`, ran 100
  trials, `integer(1, 8, true)` searched on a linear scale, and
  `fcm(X, 3, 1.5)` used `m = 2`.
- Python: data must hold numbers, as in JavaScript since 0.1.1. Booleans,
  alone or among numbers in lists, and numeric strings were converted, the
  real part of complex numbers kept, and `None` refused only as a missing
  value; they now raise `TypeError`.
- `k`, a named `init` and `seed` are checked before any work on the data,
  as the other arguments are: a wrong one was reported only after
  centering, `rmcm`'s neighborhood graph or the PCA start. Python refuses
  `seed=True`, which it took as 1, as JavaScript does.
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
  as JavaScript compares options with `===`. An option is found first as
  the object `ask` gave, so options without a truth value of `==`, such as
  arrays, can be told back; all but the first raised `ValueError`.
- Python: a search space kept in a variable type-checks. Type checkers
  accepted `TPE` and `minimize` only with the space written as a dict in
  the call: a variable holding numeric dimensions only, or numeric and
  categorical ones, was rejected. The space is now typed as a mapping of
  dimensions, and at run time `TPE` accepts any mapping too, where it
  required a dict.
- JavaScript: labels must be numbers (bigints too), strings or booleans, as
  Python's form a 1-D array: arrays or objects as labels were compared by
  identity, each a cluster of its own. Python's error for labels that do
  not sort, such as `[0, None]`, says so; NumPy's named neither.
- JavaScript: rows of unit coordinates may be typed arrays, as `matrix`
  reads them; they were refused as a malformed `[rows, cols]`.
- Python: the type declarations of `minimize` name the options it passes
  to `TPE` (`n_startup`, `n_candidates`, `gamma`), as JavaScript's do since
  0.1.1, so a type checker reports a misspelled or mistyped option.
- JavaScript: the type declarations say what the functions accept: rows
  may be typed arrays, labels booleans and bigints, and the properties of
  search-space dimensions are read-only, as the objects are frozen.
- `som_olp` and `apply.som_olp` refuse a `gamma` too large for the unit
  coordinates (`gamma` times their squared extent above 1e300, the bound
  of the data's squared norms): the latent costs overflowed, and the run
  gave `NaN` memberships and reported `"tol"`.
- JavaScript: `rcm` with a `p` whose reciprocal overflows (about 1e-320)
  gave `NaN` memberships and reported `"tol"`, where Python gives the
  limit; it now computes what Python does (`1 ** Infinity` is `NaN` in
  JavaScript, 1 in IEEE arithmetic).
- `som_olp` computes its latent costs from centered unit coordinates, so
  units far from the origin no longer overflow them through rounding (equal
  coordinates at 1e20 with a large `gamma` gave `NaN`); results move in
  their last digits only. A `pca_scale` whose start leaves the scale
  contract, and a `tau` or `lam` so large that the objective overflows,
  raise an error; they gave `NaN` memberships or `-inf` history with
  `"tol"`.
- JavaScript: sparse arrays (with holes) as rows, labels or a list of `k`
  are refused before anything is allocated for their length: `every()`
  skipped the holes, so `matrix([new Array(n)])` and `ari` of such labels
  allocated for n values before refusing them.

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
