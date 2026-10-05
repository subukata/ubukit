# Design

What UbuKit is, and why it is built this way. How to change it is in
`CONTRIBUTING.md`; the reasoning behind individual decisions is in
`docs/decisions.md`. A change that breaks a principle needs a reason in its
pull request and an entry in `docs/decisions.md`.

## Scope

UbuKit is a small library of prototype-based methods, in Python and
JavaScript:

- partitional clustering on the shared engine (k-means, fuzzy and rough
  c-means and their variants);
- self-organizing maps;
- evaluation of clusterings and embeddings (ARI, AMI, trustworthiness,
  continuity);
- hyperparameter search for them (TPE).

Out of scope: density-based, hierarchical or spectral clustering, neural
networks, data loading and preprocessing, plotting, and GPUs.

## Principles

Each principle states a rule and the failure it prevents.

1. **One loop.** Every iterative method runs in the shared loop, which owns
   the control: the stopping rule, the iteration limit, the objective history
   and, in JavaScript, progress and cancellation. A method supplies only its
   step, usually the standard alternating step (distances, then memberships,
   then weighted means), so a new method is usually just a membership rule.
   *Why:* stopping rules, histories and cancellation written per method drift
   apart and each needs its own tests; one loop gets them right once.
2. **One definition per algorithm, at most one accelerated kernel.**
   Vectorized NumPy/SciPy is the reference in Python, plain loops in
   JavaScript. An optional Numba kernel may replace a hot step when the
   benchmark shows a large gain; it must reproduce the reference iterates,
   pass the same tests and be chosen explicitly per call. Nothing is selected
   automatically by library versions, CPU models, data ranges or installed
   packages, and there are no other backends (no WASM, no GPU).
   *Why:* every extra path is a second implementation that can silently
   disagree with the first; the code this library replaced had several, and
   no test could tell which one was right.
3. **One numerical contract.** Inputs are finite float64 data of ordinary
   scale (standardize first; the bounds are in `docs/algorithms.md`).
   Stability comes from the formulation (centering, log-domain softmax,
   scaled radii), not from rescue paths, and anything outside the contract
   raises a clear `ValueError`/`RangeError`.
   *Why:* rescue paths for extreme inputs multiply code and hide wrong
   answers; a stated contract with a clear error is honest and testable.
4. **Same API, same math, not the same bits.** Python and JavaScript share
   names, parameters and equations (`docs/algorithms.md`); results agree to
   ~1e-9 when the initialization is given. Each language may add API only for
   its own concerns (JavaScript: step generators and asynchronous running;
   Python: compiled kernels).
   *Why:* users move between the languages and expect the same results, but
   random generators, BLAS and platforms make bit-for-bit agreement
   unattainable, so agreement is defined by equations and checked by fixtures.
5. **Minimal dependencies.** Python: NumPy and SciPy (Numba optional).
   JavaScript: none.
   *Why:* every dependency is installation friction and supply-chain risk;
   NumPy and SciPy are already in every scientific Python environment.
6. **Correctness from mathematics and references.** Tests check invariants
   (row-stochastic memberships, non-increasing objectives, fixed points),
   transcribed reference loops, and scikit-learn where it defines the same
   quantity. Every bug fix comes with a test that fails without it, and the
   fixtures change only when the math is meant to change.
   *Why:* a test that only repeats the implementation's own output certifies
   its bugs; a test seen failing before the fix is known to test the bug.
7. **Speed from algorithms, measured.** Algorithmic gains (BLAS products,
   sparse matrices, separable kernels, partial sorts, pruned sums) come
   before compiled kernels. Targets: Python within the same order as
   scikit-learn, JavaScript within an order of magnitude of Python, both
   measured with `bench/run.py`. A speedup that adds code needs a clear
   measured gain (as a rule of thumb, 1.5x on the affected case); one that
   adds no code need only not be slower.
   *Why:* an algorithmic gain helps every implementation at once, and an
   unmeasured optimization is usually complexity without benefit.
8. **A short, closed supply chain.** CI actions are pinned to commit SHAs
   and get the least permissions; dependency install scripts do not run;
   tests and builds run without credentials, and the publishing jobs only
   upload the built files through trusted publishing. Only the maintainer
   pushes release tags.
   *Why:* a package is only as trustworthy as the path from source to
   registry, and no third-party code may run where a publishing token can be
   minted.

## Current architecture

How the principles are realized today. This section changes with the code.

- **Engine:** `iterate(X, V, step, max_iter, tol)` in
  `python/src/ubukit/_core.py` and `js/src/core.js`; a step maps
  `(V, state, t)` to `(V, state, objective)`. The standard step is
  `lloyd(assign, update, objective)`; the online SOM's step is one epoch.
- **Input contract:** `as_matrix` (Python) and `matrix` (JavaScript) reject
  non-finite input and data outside the scale bounds.
- **Implementations:** Python is vectorized NumPy/SciPy; JavaScript is plain
  loops on `Float64Array`, with generators for progress and cancellation.
  There is no Numba kernel yet.
- **Cross-language reference:** `fixtures/generate.py` writes Python results
  to `fixtures/fixtures.json` (one case per line); the JavaScript tests
  reproduce them.
- **Benchmark:** `bench/cases.py` defines each case once; `bench/run.py`
  writes the data as binary files and runs it in a process per implementation
  (`worker.py` for Python and scikit-learn, `worker.mjs` for JavaScript),
  each importing the source tree it is given, so `--compare` times another
  commit with the same harness.

```
python/src/ubukit/  _core.py cluster.py som.py metrics.py tpe.py
js/src/             core.js  cluster.js som.js metrics.js tpe.js index.js
fixtures/           generate.py -> fixtures.json
bench/              cases.py run.py worker.py worker.mjs
docs/               algorithms.md (the equations), decisions.md (the reasons)
```

## Non-goals

- Bit-for-bit agreement across languages, platforms or library versions.
- Exact or extended-range arithmetic for subnormal or 1e300-scale data.
- Stateful sessions, workers, schedulers or other application frameworks.
- Custom release machinery such as hand-made provenance manifests or archive
  checks; the registries' own trusted publishing and provenance suffice.
