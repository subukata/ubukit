# Design

UbuKit is a small library. These rules keep it that way; changes that break
one need a reason written in the pull request.

## Principles

1. **One engine.** Every partitional method is the loop in `_core.alternate`
   (`core.js` in JavaScript): distances, then `assign`, then a weighted mean.
   A new method should be an `assign` function, not a new loop.
2. **One implementation per algorithm.** Vectorized NumPy/SciPy in Python,
   plain loops in JavaScript. No alternative backends, no Numba, no WASM, and
   no code paths selected by library versions, CPU models or data ranges.
3. **One numerical contract.** Inputs are finite float64 data of ordinary
   scale (standardize first). Stability comes from the formulation
   (centering, log-domain softmax, scaled radii), not from rescue paths.
   Anything outside the contract raises a clear `ValueError`/`RangeError`.
4. **Same API, same math, not the same bits.** Python and JavaScript share
   names, parameters and equations (`docs/algorithms.md`). Each uses its own
   random generator; results agree to ~1e-9 when the initialization is given.
   `fixtures/` holds Python results that the JavaScript tests reproduce.
5. **Minimal dependencies.** Python: NumPy and SciPy. JavaScript: none.
6. **Correctness from mathematics and references.** Tests check invariants
   (row-stochastic memberships, non-increasing objectives, fixed points),
   transcribed reference loops, and scikit-learn where it defines the same
   quantity.
7. **Speed from algorithms.** BLAS products, sparse matrices, separable
   kernels and partial sorts. The target is the same order as scikit-learn,
   measured, not guessed.

## Non-goals

- Bit-for-bit agreement across languages, platforms or library versions.
- Exact or extended-range arithmetic for subnormal or 1e300-scale data.
- Stateful sessions, workers, schedulers or other application frameworks.
- Provenance manifests, artifact hashes or custom release machinery.

## Layout

```
python/src/ubukit/  _core.py cluster.py som.py metrics.py tpe.py
js/src/             core.js  cluster.js som.js metrics.js tpe.js index.js
fixtures/           generate.py -> fixtures.json (cross-language reference)
docs/algorithms.md  the equations; code mirrors them
```

## Workflow

- Python: `cd python && pip install -e . --group dev && ruff check . && ruff format --check . && pytest`
- JavaScript: `cd js && npm ci && npm test && npm run check`
- After changing Python numerics: `python fixtures/generate.py`, then run both test suites.
- Release: bump `python/src/ubukit/__init__.py` and `js/package.json` together,
  then push a `v*` tag; `.github/workflows/release.yml` publishes both.
