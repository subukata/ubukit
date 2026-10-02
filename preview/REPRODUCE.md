# Reproduce dev3 from repository source

Run from the repository root. Python examples below use POSIX virtualenv paths;
Windows requires the corresponding `Scripts/python.exe` path and is unverified.
Generated artifacts, dependency caches and test output are deliberately ignored.
No command below publishes a package or deploys a site.

## Python build and install

Use fresh, separate build and runtime environments. Do not combine this package
with the former independent distributions, which own overlapping import paths.

```sh
python3.12 -m venv .venv-build
.venv-build/bin/python -m pip install build==1.4.0 setuptools==82.0.1 wheel==0.46.3
mkdir -p preview/python/artifacts
.venv-build/bin/python -m build --no-isolation --outdir preview/python/artifacts preview/python/staging
python3.12 -m venv .venv
.venv/bin/python -m pip install -c preview/python/constraints-final-stack.txt preview/python/artifacts/ubukit_bundled_local_preview-0.0.0.dev3-py3-none-any.whl
.venv/bin/python -m pip install pytest==8.4.2
```

For an accelerated environment, create a different fresh venv and install the
same wheel with `[numba]`. For source-distribution verification use the generated
`.tar.gz` instead of the wheel. `constraints-compat.txt` records the alternative
stack. Constraints set versions; the `requirements-*-verified.txt` files also
record the specific Linux/CPython 3.12 wheel hashes used in prior validation.
They are not portable locks for other operating systems or Python versions.

## Python installed-package checks

Use absolute paths to the selected runtime interpreter. These harnesses reject
source-tree / editable imports and check installed distribution ownership.
The complete checks can take several minutes, especially with cold Numba caches.

```sh
python3.12 preview/python/verification/runtime_tests/run_installed.py --python "$PWD/.venv/bin/python" --label repository --include-large-sparse
python3.12 preview/python/verification/external_tests/run_installed.py --python "$PWD/.venv/bin/python" --label repository
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -I -B -m pytest --import-mode=importlib -p no:cacheprovider -q preview/python/verification/new_tests
.venv/bin/python -I -B preview/python/tools/check_installed_ownership.py base
.venv/bin/python -I -B preview/python/verification/python_api.py
.venv/bin/python -I -B preview/python/examples/all_methods.py
.venv/bin/python -I -B -m pytest -p no:cacheprovider -q preview/python/verification/test_membership_axis.py preview/python/verification/test_preflight.py
.venv/bin/python -I -B preview/python/tools/check_installed_ownership.py base
```

For a Numba environment pass `--with-numba` to both `run_installed.py` commands,
`python_api.py` and `all_methods.py`; use `numba` instead of `base` for
`check_installed_ownership.py`. The new-test suite detects installed Numba.
Frozen baseline files under `_oracles/` are test references, not runtime imports.
Runtime harness exclusions/skips are documented in VERIFICATION.md.

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
node --test --test-concurrency=1 validation/tests/* consumer/tests/*.mjs tests/*.mjs
cd ../..
```

Tests use the actual installed tarball in `consumer/node_modules/ubukit-js`,
including real Node Workers, cancellation/restart, source hashes and entrypoints.
The `validation/references/` and `validation/baseline_package/` trees are frozen
comparison inputs. They are not the current candidate and are not packed.

## Source integrity

```sh
python3.12 preview/tools/verify_snapshot.py
```

This checks all 84 runtime files against SOURCE_SNAPSHOT.json. Package documentation
was updated for GitHub; archive-byte equality with the older delivered dev3
archives is not claimed. No registry release or public-license decision follows
from a passing local test. Real browsers and other platform combinations need
separate validation.
