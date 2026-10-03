> The harness imports implementation modules only under `ubukit._impl`; ownership guards require the installed `ubukit` tree. Frozen numerical fixtures, tolerances and oracle bytes remain unchanged. Current runtime hashes are in `python/SOURCE_MANIFEST.json` from the repository root. Historical results below do not qualify a newly built alpha artifact.

# Installed-wheel runtime regressions

This harness tests the installed `ubukit` distribution, version `0.1.0a3`. It must not be run against a source tree or editable installation. Every canonical runtime import is checked against the selected virtualenv's site-packages and the distribution's wheel RECORD, including recorded file hashes. Frozen baseline/reference modules live under `_oracles/` and load only under `_ubukit_oracle_*` aliases. They are never added to `sys.path`, bundled into the wheel, or used as candidate implementations.

## Run

Run from the repository root using a fresh, non-editable installed environment:

```sh
python3.12 python/verification/runtime_tests/run_installed.py \
  --python "$PWD/.venv/bin/python" --label repository --include-large-sparse
```

For a separately qualified Numba-enabled environment also pass `--with-numba`.
See `docs/getting-started.md` for the base installation and
`docs/contributing.md` for repository verification guidance. These paths are
relative to the repository root.

The driver invokes `python -I -B`, clears `PYTHONPATH`, disables third-party pytest plugin autoload, fixes BLAS/OpenMP/Numba thread counts to one, uses a dedicated Numba cache, and runs from an empty directory. Nothing is installed by this harness. Results, complete logs, JUnit and loaded-module provenance are written under `results/<label>/`.

## Coverage

- Legacy unified-review component suites: FCM/log memberships, rough/extended rough c-means, RMCM, kmeans, SOM, prepared snapshots, thread policies, portable metrics and optional Numba backends
- Unified-review facade API script, lazy imports, export identity, and sample-first membership axis/ownership/pickling/signature tests
- Fuzzy extension: near-one fuzzifiers, extreme exponents, subnormal objectives, Decimal membership/distance oracles, large-m update bits, wide Numba kernels, rough numerical boundaries, certified BLAS and explicit unsupported-provider fallback
- RMCM extension: exact tree/blocked/strict graph parity, boundary/scale/underflow and edge limits, Gram and float32 screening, cache reuse/churn/stable point IDs, complete graphs, tree guards and unsupported-provider fallback
- SOM/kmeans extension: strict numerical/reference parity, layouts/dtypes/readonly inputs, scratch budgets, overflow guards, portable SciPy finalization without sklearn/Numba imports, certified BLAS and mocked unsupported-build fallback
- Public metrics: original 80 fixtures with all four metric backends when Numba is available (40 fixtures without it), plus original 360 nearest-neighbor fixtures and nine translated self-position/tie counterexamples against public sklearn
- Added isolated Numba-absence coverage: facade lazy import, `auto` kmeans fallback parity, explicit base FCM/ExRCM/RMCM execution, and five explicit optional-backend/finalizer requests raising ImportError

## Exact exclusions and limits

1. `work/fuzzy/tests/test_numpy_reference.py`: all 13 cases are excluded because they validate only independent/reference implementations and do not execute the installed candidate. Existing candidate-versus-frozen and Decimal tests remain included.
2. `rmcm_extension/test_graph_parity.py::test_memory_limit_and_sparse_large_n`: both graph-tree and graph-blocked parameter cases are skipped by default to avoid the original 8,000-point all-pairs sparse exercise. Their original code is retained; `--include-large-sparse` restores both. An added 257-point test covers both sparse graph paths and retains the exact original 100-point memory-limit fixture.
3. Numba-dependent upstream cases retain their existing availability skips. The base environment does not install Numba. The optional environment executes them normally. Public-metric fixture loops omit only Numba/sqrt-Numba when Numba is absent.
4. The optional scikit-fuzzy comparator retains the original availability skip; this harness does not install it.
5. Three active rough BLAS certificate tests retain their upstream provider/build eligibility skips when the installed BLAS build is not certified. The unsupported-provider fallback assertions still run.
6. JavaScript/WASM suites are excluded because this artifact is a Python wheel. No benchmarks, timing comparisons, performance claims, external datasets, experiments, or training workloads are run.
7. ARI/AMI is included in the package and tested separately under `verification/external_tests`; this original-seven-family harness does not duplicate that full numerical panel.
8. Obsolete metrics `test_neighbor_query.py` and `test_neighbor_reuse.py` target superseded work candidates. The final validated `test_neighbor_brute.py` is included instead.

This harness originated with dev3 and was adapted for the dev6 namespace and
38-export contract; current identity gates target the alpha distribution.
Traditional SOM tests are in `../../tests/`. Extreme FCM/SOM expectations were
deliberately updated to the reviewed numerical contract; ordinary tests remain
in place. `tools/provenance/VERIFICATION_SNAPSHOT.json` at the repository root
binds the current verification bytes. Historical copy manifests are not current hash attestations
and are omitted.

Passing this suite establishes the exercised installed behavior, not universal numerical correctness or a new performance certification.

The historical integration verification runs included both original 8,000-point sparse graph cases via `--include-large-sparse`; the default bounded mode remains available for later smoke runs. A fresh alpha run must record its own coverage.
