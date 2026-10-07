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
   the control: the stopping rule, the iteration limit, the objective history,
   progress and cancellation (the loop is a generator, so the caller sees
   every iteration and may stop at any). A method supplies only its
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
   names, parameters and equations (`docs/algorithms.md`), each in its
   language's spelling (`max_iter`, `maxIter`), and both reject a parameter
   they do not take; results agree to ~1e-9 when the initialization is
   given. Each language may add API only for its own concerns (JavaScript:
   running the generators, synchronously or not, and its `{ data, rows, cols }`
   matrices; Python: compiled kernels).
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
   fixtures hold exactly what the code produces, changing beyond rounding
   only when the math is meant to change.
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
   tests and builds run without credentials and in separate jobs, so the
   published files are made with the build tools alone, and the publishing
   jobs only upload them through trusted publishing. Only the maintainer
   pushes release tags, only on commits of `main`, and never moves one.
   *Why:* a package is only as trustworthy as the path from source to
   registry: no third-party code may run where a publishing token can be
   minted, and only the build tools where the published files are made.
9. **Designs follow the model, not the shortest path.** A change is designed
   from what the library should be in the long run. When a feature does not
   fit, the abstraction that keeps it out is changed, even if a workaround
   would be smaller today. Signs of a workaround: an existing concept made to
   carry something else (an output passed back as state), a special case for
   one method, or code that rebuilds what an abstraction should have kept.
   Every design proposal names the alternatives and why they lost.
   *Why:* a workaround is cheap once and costs on every later change, since
   each one constrains the next and adds cases to test, while a fitting
   abstraction makes later features small (one engine made per-iteration
   results and Numba kernels a few lines each).

## Compatibility

What UbuKit promises is written down and kept small; everything else may
change in any release.

- **Promised:** the public names, which are `__all__` of `ubukit` and of
  `ubukit.steps` in Python and the exports of the package entry
  (`index.js`, with its types) in JavaScript, each with its parameters,
  defaults and Result fields; and the equations of `docs/algorithms.md`,
  which the fixtures check. Tests compare the names each package exposes
  with `__all__`, and the two languages with each other, so a change to
  them is a visible diff.
- **Not promised:** where a name is defined (module files and the import
  paths into them, such as `ubukit.cluster`), names with an underscore,
  pickles across versions (a pickle loads in the version that wrote it;
  keep results across versions as arrays), and identical bits (see
  "Non-goals").
- **Breaking changes:** while UbuKit is 0.x, a change to a promise replaces
  the old form at once, in a minor version, listed under "Changed API" or
  "Changed results" in `CHANGELOG.md`; no alias, shim or old code path is
  kept. From 1.0, an old name stays for one minor version as deprecated,
  with the version that removes it written beside it.

*Why:* anything that works without being stated becomes a promise once
someone relies on it, and each such promise needs a compatibility layer when
the code changes; such layers made the code this library replaced hard to
change. A small, written promise keeps everything else free to improve.

## Current architecture

How the principles are realized today. This section changes with the code.

- **Engine:** `iterate(data, V, step, max_iter, tol, view, prepare, keep)`
  in `python/src/ubukit/_core.py` and `js/src/core.js`, a generator in both;
  a step maps `(data, V, state, t)` to `(V, state, objective)`. The data are
  an input of every step, never kept by it: rows sent into the generator
  (`send` in Python, `next(X)` in JavaScript) are prepared by `prepare`
  (centering, and for `rmcm` its neighborhood graph), the prototypes carry
  over, and the state carries over only through the method's `keep`
  (SOM-OLP's memberships); other state is a cache of the old data. The
  standard step, `lloyd(assign, update)`, serves the fuzzy and rough
  c-means, whose `assign` gives the memberships with the objective at them
  (the history value, from the quantities that normalize the memberships);
  k-means (an exact accelerated step with Hamerly's bounds),
  the batch SOM (separable smoothing), the online SOM (one epoch) and SOM-OLP
  (memberships that depend on the previous ones) have their own. With
  `tol=None` the loop runs a fixed schedule, which converges when complete.
  Each method passes the loop its `view`, which adds the method's labels,
  memberships and embedding to what every Result has (`fitted`), so every
  yielded iteration can build, on demand, the Result the run would return
  had it stopped there; steps therefore never modify a state they have
  returned, and views copy the state they put in a Result, so changing a
  Result leaves the run unchanged. Each fitting function is written once as
  such a generator, in `cluster` or `som`: `steps` exposes the generators,
  and the package entry point makes the plain functions that run them to
  the end (`stepwise` in Python's `__init__.py`, `index.js` in JavaScript).
- **Input contract:** `as_matrix` (Python) and `matrix` (JavaScript) reject
  non-finite input and data outside the scale bounds.
- **Implementations:** Python is vectorized NumPy/SciPy; JavaScript is plain
  loops on `Float64Array`, with `runAsync` driving a generator without
  blocking the event loop.
- **Numba kernels:** `_numba.py` holds the kernels behind `engine="numba"`:
  the online SOM epoch, the trustworthiness penalties and the expected mutual
  information, the loops that NumPy cannot vectorize. Each transcribes its
  NumPy reference loop for loop and `tests/test_numba.py` requires the same
  results. The module is imported on first use and caches nothing on disk;
  the parallel kernels split the work by independent rows and leave sums to
  NumPy, so results do not depend on the thread count.
- **Cross-language reference:** `fixtures/generate.py` writes Python's public
  names and results to `fixtures/fixtures.json` (one case per line); the
  JavaScript tests check the names and reproduce the results.
- **Benchmark:** `bench/cases.py` defines each case once; `bench/run.py`
  writes the data as binary files and runs it in a process per implementation
  (`worker.py` for Python and scikit-learn, `worker.mjs` for JavaScript),
  each importing the source tree it is given, so `--compare` times another
  commit with the same harness. `baseline.py` holds the textbook form of
  each method, which `--baseline` times against UbuKit.

```
python/src/ubukit/  _core.py cluster.py som.py steps.py metrics.py tpe.py _numba.py
js/src/             core.js  cluster.js som.js metrics.js tpe.js index.js
fixtures/           generate.py -> fixtures.json
bench/              cases.py run.py worker.py worker.mjs baseline.py
docs/               algorithms.md (the equations), decisions.md (the reasons)
```

## Non-goals

- Bit-for-bit agreement across languages, platforms or library versions.
- Exact or extended-range arithmetic for subnormal or 1e300-scale data.
- Stateful sessions, workers, schedulers or other application frameworks.
- Custom release machinery such as hand-made provenance manifests or archive
  checks; the registries' own trusted publishing and provenance suffice.
