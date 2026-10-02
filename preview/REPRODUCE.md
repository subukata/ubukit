# Reproduce Python dev6 / JavaScript dev6 from repository source

Run from the repository root. Python examples below use POSIX virtualenv paths;
Windows requires the corresponding `Scripts/python.exe` path and is unverified.
Generated artifacts, dependency caches and test output are deliberately ignored.
No command below publishes a package or deploys a site.

## Python build and install

Use fresh, separate build and runtime environments. Keep old aggregate installations separate. Only `ubukit` is installed at the top level; standalone legacy names alone are no longer a namespace collision. Use the read-only install-environment guard for actual `ubukit` ownership conflicts.

```sh
python3.12 -m venv .venv-build
.venv-build/bin/python -m pip install build==1.4.0 setuptools==82.0.1 wheel==0.46.3
mkdir -p preview/python/artifacts
.venv-build/bin/python -m build --no-isolation --outdir preview/python/artifacts preview/python/staging
python3.12 -m venv .venv
.venv/bin/python -m pip install -c preview/python/constraints-namespace-verified.txt preview/python/artifacts/ubukit_bundled_local_preview-0.0.0.dev6-py3-none-any.whl
.venv/bin/python -m pip install pytest==8.4.2
```

For an optional accelerated environment, create a different fresh venv and
install the same wheel with `[numba]` using `constraints-final-stack.txt`. This
SOM integration replay uses the base stack without Numba; the earlier dev3
Numba matrix remains historical evidence. For source-distribution verification use the generated
`.tar.gz` instead of the wheel. `constraints-compat.txt` records the alternative
stack. Constraints set versions; the `requirements-*-verified.txt` files also
record the specific Linux/CPython 3.12 wheel hashes used in prior validation.
They are not portable locks for other operating systems or Python versions.

## Python installed-package checks

Use absolute paths to the selected runtime interpreter. These harnesses reject
source-tree / editable imports and check installed distribution ownership.
The complete checks can take several minutes, especially with cold Numba caches.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -I -B -m pytest --import-mode=importlib -p no:cacheprovider -q preview/python/tests preview/python/verification/test_membership_axis.py
.venv/bin/python -I -B preview/python/examples/som_variants.py
python3.12 preview/python/verification/runtime_tests/run_installed.py --python "$PWD/.venv/bin/python" --label repository
python3.12 preview/python/verification/external_tests/run_installed.py --python "$PWD/.venv/bin/python" --label repository
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -I -B -m pytest --import-mode=importlib -p no:cacheprovider -q preview/python/verification/new_tests
.venv/bin/python -I -B preview/python/tools/check_installed_ownership.py base
.venv/bin/python -I -B preview/python/tools/check_artifact_origins.py base-wheel
.venv/bin/python -I -B preview/python/verification/python_api.py
.venv/bin/python -I -B preview/python/examples/all_methods.py
.venv/bin/python -I -B -m pytest -p no:cacheprovider -q preview/python/verification/test_membership_axis.py preview/python/verification/test_preflight.py
.venv/bin/python -I -B preview/python/tools/check_installed_ownership.py base
```

For a Numba environment pass `--with-numba` to both `run_installed.py` commands,
`python_api.py` and `all_methods.py`; use `numba` instead of `base` for
`check_installed_ownership.py`. The new-test suite detects installed Numba.
Frozen baseline files under `_oracles/` are test references, not runtime imports.
Runtime harness exclusions/skips are documented in VERIFICATION.md. Add
`--include-large-sparse` for the two inherited 8,000-sample sparse cases; they
were not rerun for this additive SOM integration. Repeat the focused
gate, examples, ownership and artifact-origin checks in a fresh sdist-installed
venv, using `base-sdist` for `check_artifact_origins.py`. The sdist build requires the build dependencies from the build step.

## JavaScript build, install and all tests

Node >=20 is declared; the tested runtime is Node v24.19.0 on Linux x86-64.
The package has no runtime npm dependencies or install scripts.

```sh
mkdir -p preview/javascript/artifacts
cd preview/javascript/package
npm pack --ignore-scripts --pack-destination ../artifacts
cd ../consumer
npm install --offline --ignore-scripts --no-audit --no-fund
cd ..
node --test --test-concurrency=1 tests/*.test.mjs consumer/tests/*.test.mjs validation/tests/*.js validation/tests/*.mjs
UBUKIT_AUDIT_SOURCE=../consumer/node_modules/ubukit-js/src/external-metrics.js node audit/audit-numerics.mjs
node tools/check-package-docs.mjs
node --experimental-vm-modules tests/runtime-compatibility.mjs
node --experimental-vm-modules tests/som-browser-compatibility.mjs
cd ../..
```

Tests use the actual installed tarball in `consumer/node_modules/ubukit-js`,
including real Node Workers, cancellation/restart, source hashes and entrypoints.
The consumer dependency points to `ubukit-js-0.1.0-dev.6.tgz`. Use a clean
consumer without an old node_modules or package-lock.json for a fresh replay.
The exact glob above runs 1,041 tests; it does not treat helper scripts as tests.
The oracle audit and runtime smoke are separate checks. The test/audit commands
may regenerate report JSON files; preserve committed measurements when comparing
a replay, and review those differences before committing. See
[javascript/REPRODUCE.md](javascript/REPRODUCE.md) for the numerical references
and matched timing commands.

The `validation/references/` and `validation/baseline_package/` trees are frozen
comparison inputs. They are not the current candidate and are not packed.

## SOM reference fixtures and bounded timing

The shared ten-case scalar oracle is committed for both languages. To regenerate
it in a throwaway checkout, run `python3.12 preview/javascript/tools/generate-som-reference.py`
and compare its result with both committed `som-shared-reference.json` files.
Do not replace reference fixtures simply to make a failing implementation pass.

The committed traditional-SOM measurements are historical final-candidate runs,
not new source-integration timings. To produce a separate local measurement:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I -B preview/python/benchmarks/measure_som.py > som-python-local.json
(cd preview/javascript && node bench/som-bounded.mjs)
```

The JS command overwrites `reports/som-bounded-benchmark.json`; preserve the
committed record when reviewing the replay. These online/batch budgets differ
and do not establish equal-quality algorithm superiority.

## Efficiency and backend compatibility checks

The focused Python suite includes the namespace/no-alias and collision tests, plus the opt-in localized-tail tests and the
intentional duplicate-backend contracts. With Numba absent the compiled-only
class is skipped; the source guard still runs. Additional exact-arithmetic tests:

```sh
.venv/bin/python -I -B preview/python/tools/check_package.py
.venv/bin/python -I -B preview/python/tools/check_high_precision.py
python3.12 preview/python/tools/check_duplicate_backend.py
```

`check_package.py` expects both freshly built archive forms under
`preview/python/artifacts` and verifies all 63 source/archive/installed modules.
`check_high_precision.py` runs 113 oracle and preserved-baseline cases. To check
an installed duplicate-backend pair instead of repository sources, set
`UBUKIT_SOURCE_ROOT` to that environment's site-packages before running its
`test_backend_contracts.py`. The default intentionally audits repository sources.

WASM byte-exact rebuild and bounded checks are a separate development-only step:

```sh
(cd preview/javascript/wasm-rebuild && npm ci --ignore-scripts --no-audit --no-fund && npm run verify)
```

WABT is a pinned development tool, not an installed UbuKit runtime dependency.
See [EFFICIENCY.md](EFFICIENCY.md) for bounded serial benchmark commands and
[wasm-rebuild/README.md](javascript/wasm-rebuild/README.md) for provenance limits.

## Source integrity

```sh
python3.12 preview/tools/verify_snapshot.py
```

This checks all 90 runtime files (63 Python dev6 and 27 JavaScript dev6) against SOURCE_SNAPSHOT.json, plus the test/reference/evidence bytes in
VERIFICATION_SNAPSHOT.json. Run it on a clean checkout before replay commands
regenerate report JSON, or restore the committed evidence afterward. Package
documentation was updated for GitHub; archive-byte equality with earlier
standalone deliveries is not claimed. No registry release or public-license decision follows
from a passing local test. Real browsers and other platform combinations need
separate validation.

## Namespace migration audit

Run `python3.12 preview/python/tools/check_namespace_migration.py --baseline-root /path/to/preserved-dev5/staging/src` to reproduce the read-only 62-file numerical-AST and exact hash comparison. Use a preserved dev5 checkout or artifact; do not mix old and new runtime files in one environment. The full migration matrix and pickle/cache cautions are in [NAMESPACE_MIGRATION.md](NAMESPACE_MIGRATION.md). The manual CI drivers are adapted to dev6, but their explicit approved-SHA gate remains fail-closed until a separately approved candidate SHA is pinned. No Actions dispatch is part of this source integration.
