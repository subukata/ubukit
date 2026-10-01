# Restart-v2 assignment-only k-means finalizer

This is new v2 work. It is not represented as recovered historical source.
The recovered Lloyd core remains unchanged.

`finalize(X, centers, threads=1)` returns final nearest-center labels, inertia,
and per-row squared distances. It reuses the recovered target-neutral register8
intrinsic but does not accumulate means, update centers, or compute square
roots. Float64 feature components are accumulated in increasing feature order;
lowest center index wins equal computed squared distances. Inertia uses a fixed
NumPy sum of the distance vector, independent of Numba worker count.

All public-call work belongs inside a fit timer: validation/layout conversion,
center transpose, output allocation, assignment, finite checks and inertia sum.
The optional low-level `_assignment_register8` requires prevalidated finite,
nonempty C-contiguous float64 arrays with matching feature dimensions.

This arithmetic need not reproduce sklearn's norm-expansion comparisons at
near ties or its training-data recentering. Tiny tests assert direct-reference
bitwise agreement, separately compare actual sklearn, and explicitly record
cancellation-prone/recentering-sensitive cases. Inputs and centers are not
mutated; overflows are rejected rather than returning misleading assignments.

## Verification and matched-fit evidence

- `test_finalizer.py`: direct NumPy, exact ties, duplicates/empty centers, odd K,
  K>N, immutable/strided inputs, actual nonempty sklearn fits and thread restore
- `benchmark_matched.py`: actual public portable fit with either finalizer versus
  actual sklearn; first calls separate; all finalization inside each timer
- The benchmark uses fresh processes per fixture, passive OpenMP, interleaved
  repetitions and fixed supplied initialization. It aborts on any output or
  nonempty-cluster mismatch instead of quietly ranking a different result

Tiny correctness and both seven-repeat matched-fit cases passed. See
`RESULTS.md` for timings, retained outliers, numerical differences and limits.
Only the current Linux x86-64 environment was measured.

Run only using the shared restart_v2/.venv, after the parent grants the relevant
exclusive CPU slot. No independent dependency environment or installation is
needed. Public package integration is owned by the common-core worker.
