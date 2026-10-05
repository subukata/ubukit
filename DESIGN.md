# Design

UbuKit is a small library. These rules keep it that way; changes that break
one need a reason written in the pull request.

## Principles

1. **One engine.** Every iterative method runs in `_core.iterate` (`core.js`
   in JavaScript), which owns the control: the stopping rule, the iteration
   limit, the objective history and, in JavaScript, progress and
   cancellation. Methods supply the step. Nearly all use the standard step
   `lloyd`: distances, then `assign`, then a weighted mean. A new method
   should be an `assign` function; a step of its own is for iterations of
   another shape (the online SOM's epoch, an exact accelerated k-means), and
   no method gets a loop of its own.
2. **One definition per algorithm, at most one accelerated kernel.**
   Vectorized NumPy/SciPy is the reference in Python, plain loops in
   JavaScript. An optional Numba kernel (installed with `ubukit[numba]`,
   chosen per call with `engine="numba"`) may replace a hot step when the
   benchmark shows a large gain; it must reproduce the reference iterates and
   pass the same tests. Nothing is selected automatically by library
   versions, CPU models, data ranges or installed packages, and there are no
   other backends (no WASM, no GPU). No Numba kernel exists yet.
3. **One numerical contract.** Inputs are finite float64 data of ordinary
   scale (standardize first; the exact bounds are in `docs/algorithms.md`
   and enforced by `as_matrix`/`matrix`). Stability comes from the formulation
   (centering, log-domain softmax, scaled radii), not from rescue paths.
   Anything outside the contract raises a clear `ValueError`/`RangeError`.
4. **Same API, same math, not the same bits.** Python and JavaScript share
   names, parameters and equations (`docs/algorithms.md`). Each uses its own
   random generator; results agree to ~1e-9 when the initialization is given.
   `fixtures/` holds Python results that the JavaScript tests reproduce.
5. **Minimal dependencies.** Python: NumPy and SciPy (Numba optional).
   JavaScript: none.
6. **Correctness from mathematics and references.** Tests check invariants
   (row-stochastic memberships, non-increasing objectives, fixed points),
   transcribed reference loops, and scikit-learn where it defines the same
   quantity.
7. **Speed from algorithms.** BLAS products, sparse matrices, separable
   kernels and partial sorts come before compiled kernels. The target is the
   same order as scikit-learn, measured with `bench/run.py`, not guessed.

## Changing an algorithm

- **Math, defaults or API:** change `docs/algorithms.md`, Python and
  JavaScript in one pull request and regenerate the fixtures. The fixture
  diff (one case per line) shows which results changed; list them under
  "Changed results" in `CHANGELOG.md`.
- **Speed only:** one language at a time is fine. The fixtures must not
  change; if they do, the math changed.
- Either way, paste the `python bench/run.py` table from before and after.

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
bench/              run.py + run.mjs: time and quality of both languages
docs/algorithms.md  the equations; code mirrors them
```

## Workflow

- Python: `cd python && pip install -e . --group dev && ruff check . ../fixtures ../bench && ruff format --check . ../fixtures ../bench && pytest`
- JavaScript: `cd js && npm ci --ignore-scripts && npm test && npm run check`
- After changing Python numerics: `python fixtures/generate.py`, then run both test suites.
- Benchmark: `python bench/run.py` from the repository root, with Node.js on PATH.
- Release: bump `python/src/ubukit/__init__.py` and `js/package.json` together,
  date the version in `CHANGELOG.md`, then push a `v*` tag;
  `.github/workflows/release.yml` publishes both.
