# Changelog

While UbuKit is 0.x, minor versions may change numerical results and the API;
every such change is listed under "Changed results" or "Changed API". Patch
versions keep the API and change results only to fix bugs, listed under
"Fixed".

## 0.1.1 (unreleased)

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

## 0.1.0 (2026-10-06)

First release of the rewritten library, for Python (PyPI) and JavaScript
(npm): k-means, fuzzy c-means, entropy-regularized FCM, rough c-means
(RCM/ExRCM), rough membership c-means, online and batch SOM, SOM-OLP, ARI,
AMI, trustworthiness, continuity and multivariate TPE. Every fitting
function has a generator in `steps` that yields each iteration's Result and
accepts new data between iterations, so learning can follow data that move.
In Python, the online SOM, trustworthiness, continuity and AMI can run as
Numba kernels (`engine="numba"`, with `pip install "ubukit[numba]"`).
