> Historical investigation: results, package identities and command blocks below describe their original recorded snapshot. Commands are archival instructions, not the current checkout workflow. Use [current build instructions](../getting-started.md) and [benchmark layout](../../benchmarks/README.md) for this product tree. No new performance measurement is claimed.

# Efficiency and maintainability: original Python dev5 / JavaScript dev6 evidence

This private preview integrates the reviewed candidates without changing their
numerical source. Python remains opt-in. Original algorithms, explicit fallback
routes, frozen test references and historical measurements are retained. No
registry release, CI enablement or project-wide license decision is implied.

## Python: explicit localized probability-tail experiment

In current Python dev6, use the private functions in `ubukit._impl.portable_accel.som_olp_localized` directly; the 38 `ubukit`
facade exports and existing defaults are unchanged. The original dev5 experiment added one module. Current namespace migration and verification are described in [NAMESPACE_MIGRATION.md](https://github.com/subukata/ubukit/blob/5a197111b605c495f3f2a7ba69a6e09bb44e8caf/preview/NAMESPACE_MIGRATION.md).
Of the 61 inherited runtime modules, 60 are byte-identical and the facade changes
only its private version string. See the [usage and numerical contract](../../python/EXPERIMENTAL_LOCALIZED_SOM.md).

The candidate accelerates ordinary-coordinate cases with tiny initial
probability tails. Unsafe rows, extreme coordinates, explicit optional backends,
and insufficient scratch budgets retain the original route. The direct float64
reductions and cancellation thresholds do not establish all-input, all-dimension
or trajectory equivalence. The original precise implementation remains available.

The recorded fixed-parameter replay completed all 53 historically timed-out
SOM HPO configurations within the original two-second fit timer; maximum observed
fit time was 105.391 ms. This is not a new HPO search or a proof of equivalent
quality across those 53 cases. Historical failures used dev3; the separate paired
matrix compares the unchanged current dev4 numerical implementation.

Three alternating warm samples on the recorded one-thread Linux/CPython stack
showed 5.44x–131.61x gains for four completed-baseline synthetic tail cases.
Timeout ratios for other cases are censored lower bounds, not completed-baseline
speedups. Ordinary no-tail cases were 0.76x and 0.98x in that final sample,
including a noisy approximately 0.10 ms small-case penalty. Existing defaults
never call the new wrapper. Peak RSS was about 59 MiB in four single observations;
no memory reduction is established. Warm fit ratios are not startup ratios.

Frozen measurements are in `python/verification/efficiency/`: the paired matrix,
HPO replay and cold-memory records. They retain original package-version labels,
paths, values and timing conditions. `version_rebind.json` is original delivery
metadata linking numerical bytes to dev5; its packet-local paths describe that
original packet, not additional repository files. Current repository bindings and
fresh local replay results are in [EFFICIENCY_VERIFICATION.json](https://github.com/subukata/ubukit/blob/5a197111b605c495f3f2a7ba69a6e09bb44e8caf/preview/EFFICIENCY_VERIFICATION.json).

## JavaScript: conservative routes and preserved elites

- Traditional SOM auto-selection uses grouped BMUs only for K >= 16 and D <= 8
- `bmuBackend: "scalar"` selects the complete byte-original dev5 SOM module,
  retained as `src/som-elite.js`; its direct dependency closure is unchanged
- `bmuBackend: "grouped"` explicitly enables the specialized route for other shapes
- k-means can reuse WASM center transposes within a safe iteration. Checkpoint
  exposure disables this cache permanently, including late handler installation
  and subsequent removal. Sessions retain per-block refresh
- `wasmCenterCache: false` keeps the original per-block behavior
- Existing k-means JavaScript hot loops and all five embedded WASM binaries are unchanged

There is no extra typed-array scratch allocation in these paths. Retaining the
original SOM module adds 19,104 source bytes and one module to parse; warmed fit
measurements exclude that startup tradeoff. The frozen comparison package under
`javascript/validation/efficiency_baseline_package` is test-only and never packed.

Recorded whole-call medians showed 1.03x–1.41x k-means gains and up to 2.31x for
selected low-D SOM cases. Small online SOM regressed from 0.153 to 0.169 ms;
high-D auto routes could be a few percent slower. Explicit grouped can win on
dense high-D data yet lose when early cutoffs dominate. A large co-resident
D784 gap did not reproduce in independent fresh processes. JIT/warmup/allocation
interactions are plausible, not an established single cause. Keep the original
package wherever it remains the measured champion; no universal winner is claimed.

`javascript/reports/efficiency-{paired,cutoffs,isolated}.json` retain the original
measurements. The version-transition record binds dev6's unchanged 27 runtime
files to the earlier reviewed candidate. Repository-only documentation can alter
archive hashes without changing runtime bytes. Original-packet artifact hashes
are not hashes of a fresh repository rebuild.

## Maintainability without numerical refactoring

The deliberate Python duplicate metric modules keep independent classes,
globals and Numba dispatchers. Replacing one with re-exports broke class identity,
pickle/import behavior and monkey-patching contracts during assessment, so both
remain unchanged. `python/tools/check_duplicate_backend.py` rejects divergence;
`python/tests/test_backend_contracts.py` covers source equality, deliberate-drift
detection, optional-dependency absence, direct imports, identity and dispatcher
contracts. Prior validation passed 13 tests with Numba and six without it.

The [WASM rebuild supplement](../../javascript/wasm/README.md) lives outside
installed runtime files. Four current kernels rebuild from recovered original
WAT; the metric kernel rebuilds from clearly labeled reconstructed WAT. Byte
identity verifies reproducibility. The maintainer confirmed project development;
[the source record](../../javascript/wasm/DEVELOPMENT.md) distinguishes preserved
original WAT from the metric and historical RMCM reconstructions. Pinned WABT metadata is retained;
no node_modules, compiler archive or compiled development payload is vendored.

## Reproduce checks and measurements

Follow [REPRODUCE.md](https://github.com/subukata/ubukit/blob/5a197111b605c495f3f2a7ba69a6e09bb44e8caf/preview/REPRODUCE.md) for clean builds and installed-package checks.
Additional differential checks, run from the repository root:

```sh
node preview/javascript/validation/efficiency/differential.mjs
node preview/javascript/validation/efficiency/workers.mjs
```

The first checks 15,573 cases; the second checks 18 real Node-worker outcomes.
They print JSON without rewriting committed evidence. Browser-like VM results
are not real-browser qualification.

Benchmarks must run serially, with fixed thread budgets and a quiet host. Use a
throwaway checkout for JS timing, because its scripts overwrite report JSON:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I -B preview/python/benchmarks/compare_localized_tails.py --out /tmp/python-paired-local.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I -B preview/python/benchmarks/replay_failed_hpo.py --source preview/python/benchmarks/failed_hpo_trials.jsonl --out /tmp/python-replay-local.json
(cd preview/javascript && node bench/efficiency-paired.mjs && node bench/efficiency-cutoffs.mjs && node bench/efficiency-isolated.mjs)
```

Do not replace frozen evidence merely because a replay differs. Record software,
hardware, cold/warm scope, completed versus timed-out baselines and raw samples.
Other platforms, actual browsers, general numerical equivalence and universal
speed rankings remain unverified.
