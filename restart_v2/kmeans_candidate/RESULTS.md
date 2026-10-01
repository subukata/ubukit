# Restart-v2 assignment-only finalizer: matched public-fit evidence

This is new optimization work, not restored historical code. The recovered
Lloyd core is unchanged. The public API retains `finalizer='sklearn'` as default;
`finalizer='numba'` explicitly selects the assignment-only direct finalizer.
Both return finalized labels and inertia inside the public fit call.

## Matched end-to-end measurements

One fresh process per fixture, 9 threads, passive OpenMP, same fixed float64
input/initial centers, max_iter=20, and seven interleaved warm repetitions.
Actual sklearn used Lloyd, n_init=1, tol=0, copy_x=True and the same max_iter.
Every repetition passed exact final labels, nonempty core/final clusters,
20 iterations, center/inertia tolerance 1e-10, and independent direct inertia.
Inputs and public package source hashes were unchanged.

| Case and method | Median ms | MAD ms | Observed range ms |
|---|---:|---:|---:|
| Low-D: public register8 + new direct finalizer |6.883|0.886|5.997–16.328|
| Low-D: public register8 + sklearn finalizer |9.153|0.547|8.595–12.338|
| Low-D: actual sklearn |14.620|2.573|12.046–35.541|
| High-D: public vector BLAS + new direct finalizer |28.586|1.424|27.080–82.204|
| High-D: public vector BLAS + sklearn finalizer |29.870|1.519|27.512–31.912|
| High-D: actual sklearn |34.588|1.100|33.488–39.980|

Low-D is N=20,000, D=8, K=16; high-D is N=10,000, D=128, K=64. Seed=20261001.
Measured median speedups versus actual sklearn were 2.12× and 1.21×. Relative to
the old public finalizer, they were 1.33× and 1.045×. The high-D incremental gain
is small relative to variability and is not a robust universal speed claim.
Large scheduling/latency outliers are retained, including the 82.20 ms high-D
sample. This single-seed evidence does not establish broad or tail-latency
superiority. First calls/setup are recorded separately; they are not claimed as
clean-install JIT compilation measurements.

## Correctness and numerical boundaries

24 tiny direct-reference/thread-count checks were bitwise exact for labels,
squared distances and inertia. Immutable/strided inputs, thread restoration,
invalid inputs and arithmetic-overflow rejection also passed. Normal nonempty
actual sklearn fits matched final labels and inertia tolerance.

The new finalizer uses strict increasing-feature-order squared differences,
first-index computed ties, no centroid update and no sqrt/square. It therefore
does not promise sklearn's norm-expansion arithmetic at cancellation-prone
coordinates. Tests retain these observed differences:

- At 1e9 offset, sklearn argmin_min disagreed on 1 of 3 points in a tied case
- On a 75-point offset dataset, direct labels matched sklearn's recentered
  training labels, but sklearn.predict differed on 54 points. Inertias from the
  fitted model and direct calculation differed by only 7.0e-13

These are explicitly reported, not hidden or silently repaired. They support
keeping the direct arithmetic contract an explicit choice.

## Evidence and reproduction

- `finalizer.py`: new assignment-only implementation
- `test_results.json`: all tiny checks and sensitive comparisons
- `results/matched_v2/{low_d,high_d,summary}.json`: every timing/validation record
- `results_summary.json`: compact matched comparison
- `source_manifest.json`: source/dependency hashes and checkpoint status

All work used the shared restart_v2/.venv; no independent dependency install
was performed. No additional timing is running.
