# Restart-v2 strict joint neighborhood metrics

This is a new, independently checkpointed restart candidate, not the lost final historical package. Current checks and any later timings belong to v2 only.

## API

```python
from restart_v2.metrics_candidate import joint_quality

scores = joint_quality(X, Y, [5, 15, 50], backend='numpy', threads=1,
                       block_rows=256, max_scratch_bytes=32 * 2**20,
                       max_distance_bytes=2 * 2**30)
```

The result is always a list of immutable `Quality` records with `k`, `trustworthiness`, `continuity`, `trustworthiness_penalty`, and `continuity_penalty`. Repeated k values are deduplicated in request order. Both scores use sklearn's exact normalization order. Every k must satisfy 1 <= k < n/2.

Backends:

- `numpy`: recovered NumPy-only strict evaluator; default rank method `sortsearch`. Other methods: `full`, `broadcast`, `searchsorted`
- `sqrt_numpy`: new v2 NumPy-only adaptation of recovered rounded-sqrt interval logic; default `sortsearch`, also `broadcast`
- `numba`: exact recovered core, loaded only on request; `scan` default, also `histogram` and `full`
- `sqrt_numba`: recovered sqrt-elision candidate with its import adapted to a relative NumPy-only dependency; `scan` default, also `histogram`

The default import and both NumPy backends do not import Numba or llvmlite. Optional Numba/llvmlite were installed by the environment owner from official PyPI after the first durable checkpoint. Both optional strict routes have now passed fresh v2 correctness tests. Public block_rows applies to rank/endpoint work for these routes too, after a single full distance call per space.

## Numerical contract

All public routes use the installed sklearn full Euclidean distance call shape and preserve its resulting float32/float64 dtype. They call sklearn's own nearest-neighbor selector separately for every k and preserve NumPy's actual unstable argsort permutation for queried tied distances. There is no max-k prefix assumption, centered-norm substitution, approximate neighbor search, or hidden index-tie policy.

`block_rows` partitions only rank work. It does not turn the distance calculation into rectangular blocks. Distance storage remains quadratic. `max_distance_bytes` refuses a request whose largest one-space distance matrix exceeds that many bytes; it is not a total RSS cap. `max_scratch_bytes` caps the principal NumPy broadcast or sort rank buffer, not all query arrays, inputs, outputs, allocator retention, or numerical-library workspace. NumPy sortsearch refuses a cap smaller than one sorted distance row.

The sqrt backends retain sklearn's squared-distance computation, then identify the exact intervals of representable squared inputs mapping to each query's rounded NumPy sqrt. Tie fallback roots and sorts the original-dtype row. Simply sorting squared distances without this repair is not the implemented contract.

## Current verification

Fresh v2 tests use Python 3.12 / NumPy 2.3.5 / SciPy 1.17.0 / sklearn 1.8.0 / Numba 0.67.0. The four initial NumPy groups passed in 1.285 s; the latest five-group aggregate, including both Numba routes, passed. See TEST_ALL_V2.log for its exact duration:

- Exact scores and integer penalties against independent full-rank calculations and two actual sklearn calls, for random float32/float64, duplicate, all-zero, and grid fixtures, multiple k, six NumPy-method combinations
- A fresh subprocess import trap proves NumPy paths do not import Numba/llvmlite
- Sqrt rounding neighborhoods, subnormal/zero/infinite values, repeated queries, and ties
- API bounds, ordering, immutability, and distance-memory refusal

`TEST_NUMPY_V2.log` and `TEST_ALL_V2.log` are fresh results. The aggregate checks NumPy methods and both Numba scan/histogram routes with rank blocks of seven rows. Two fresh timing slots have completed; see results/FIRST_SLOT_SCOPE.md and results/SECOND_SLOT_SCOPE.md. Computational method sources stayed unchanged during timing. No cross-OS performance statement is made. A platform/core-count warning was emitted by joblib; tests passed.

Run the small suite from the restart parent directory with:

```bash
restart_v2/.venv/bin/python -m unittest restart_v2.metrics_candidate.test_candidates -v
```

## Provenance

`_numba_core.py` is byte-identical to the recovered post-memory-cleanup core (SHA256 5510a88580276154f257c4d403d4376d3a11a45798264236e6a5af3c8233053b). `_numpy.py` is an unmodified copy of the independently restored NumPy candidate; its prior-loss hash was not available, so only its recovery-source hash is asserted. `_sqrt_numba.py` changes the original absolute core import to a relative NumPy-only helper import. `_sqrt_intervals.py` extracts the exact recovered interval functions into a NumPy-only module. The public wrapper, NumPy interval ranker, and tests are new v2 code. `HASH_MANIFEST.json` records current and recovery-source hashes.


## Isolated timing driver (requires a serialized timing slot)

```bash
restart_v2/.venv/bin/python -m restart_v2.metrics_candidate.benchmark \
  --n 2000 --d 64 --k 15 --threads 1 --repeats 3 \
  --methods sklearn numpy numba sqrt_numba sqrt_numpy \
  --output restart_v2/metrics_candidate/results/screen_2k_t1.jsonl
```

Run again with --threads 9 after the one-thread case. Fresh child processes inherit the fixed CPU affinity and numerical environment before imports. Data-generation arithmetic is pinned to one thread for identical inputs across resource budgets. Timings exclude warmup and generation, retain every trial plus median/MAD and process RSS, and compare exact scores, integer penalties, input hashes, and source hashes. Historical results are never merged into this v2 output.


## Measured v2 checkpoint

The n=10,000, d=64 to 2, float64, k=15, nine-thread warmed comparison measured 9.537960 s for two actual sklearn calls, 3.337684 s for NumPy broadcast, 1.231928 s for Numba scan, and 1.128320 s for the rounded-sqrt Numba route. Exact scores, integer penalties, and input hashes matched. The observed speedups were 2.86x, 7.74x, and 8.45x; these are workload/environment-specific, not generic operating-system claims. Public defaults have not been changed based on a single winning screen.

A bundled-digits tie-heavy control is prepared in run_digits_control.sh but has not been run. The benchmark driver now accepts --dataset digits; this data-loader extension was made after the recorded slots and changes no computational method. Future runs record explicit UTC completion timestamps.
