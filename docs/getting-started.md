# Getting started

UbuKit is alpha software. The candidates are `ubukit==0.1.0a3` and `ubukit-js@0.1.0-alpha.3`. At source preparation on 2026-10-03, new exact-artifact qualification and initial registry publication were pending. This is a dated preparation record; check the maintainer’s release record for subsequent results and exact archive hashes. Build from a repository checkout or install a supplied local archive using the commands below.

No UbuKit release described here had been published to PyPI or npm at source preparation. Bare registry commands such as `pip install ubukit`, `pip install 'ubukit[numba]'`, or `npm install ubukit-js` require a subsequently announced and verified registry release; the supplied local artifacts do not. The package/import names used in code examples become available after installing the supplied local artifacts.

Run repository commands from the repository root. Python declares Python 3.10 or later and JavaScript declares Node.js 20 or later; these floors are unchanged and are not a tested version matrix. The most recent exact-artifact evidence is for alpha.2 on Linux x64, Windows x64 and macOS ARM64 with CPython 3.12 / Node 24. The full Python wheel regression ran only on Linux; optional Numba was not tested. See the [coverage boundary](numerics.md#coverage-boundary) for the exact scope and run identities. Those results do not qualify newly built alpha.3 archives.

## Python

Build a wheel, then install it in a fresh environment with the hash-pinned CPython 3.12 base verification profile. These pins do not qualify every consumer environment, Python version or platform:

```sh
python3.12 -m venv .venv-build
.venv-build/bin/python -m pip install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: --no-deps -r tools/ci/constraints/installer.txt
.venv-build/bin/python -m pip install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: -r tools/ci/constraints/build-py312-locked.txt
.venv-build/bin/python -m build --no-isolation --outdir python/artifacts python
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: --no-deps -r tools/ci/constraints/installer.txt
.venv/bin/python -I -B python/tools/check_install_environment.py
.venv/bin/python -m pip install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: -r tools/ci/constraints/base-py312-locked.txt
.venv/bin/python -m pip install --no-deps python/artifacts/ubukit-0.1.0a3-py3-none-any.whl
.venv/bin/python -c "import ubukit; print(ubukit.__version__)"
```

If you already have the wheel, use the self-contained local installation commands in the [Python README](../python/README.md#installation-boundary). Use a fresh environment when moving from earlier previews: overlapping installations can have conflicting file ownership. Stop if the repository's environment guard reports a conflict. The commands above use POSIX shell paths. On Windows, a virtual environment uses `Scripts/python.exe` instead of `bin/python`. The recorded Windows qualification consumed the same frozen Linux-built archives with checkout line-ending conversion disabled; it does not qualify every Windows source checkout or build configuration.

Numba is optional and enabled through the wheel's `[numba]` extra. The [historical optional-backend constraints](../python/constraints/constraints-final-stack.txt) record an earlier checked stack; they do not qualify Numba for this alpha. Use the base installation above unless you separately verify the optional backend in your environment. If enabling Numba, use a private, trusted `NUMBA_CACHE_DIR` that other users or request submitters cannot write; never accept compiler caches from untrusted inputs.

## JavaScript

```sh
mkdir -p javascript/artifacts
cd javascript
npm pack --ignore-scripts --pack-destination artifacts
cd ..
npm install --offline --ignore-scripts --no-audit --no-fund ./javascript/artifacts/ubukit-js-0.1.0-alpha.3.tgz
```

If you already have the tarball, install it using its local path and skip `npm pack`. The package has no runtime npm dependencies.

For browser use, serve the ES modules and Worker modules over HTTP(S), or use a bundler that preserves module Worker URLs. The [browser example](../examples/README.md) imports `javascript/src/index.js` directly from the checkout. The alpha.2 artifacts passed bounded Playwright Chromium, Firefox and WebKit scenarios, including entropy-regularized FCM and external metrics, plus repository demo-server smoke tests. WebKit is not native Safari, and these fixtures do not establish every-algorithm or every-browser support. Alpha.3 requires fresh qualification.

## Untrusted requests

Library validation and scratch-memory options do not impose a complete CPU, memory or wall-time limit. Before allocating arrays, applications accepting untrusted requests must bound request bytes, sample/feature/cluster or map dimensions, iteration counts and requested outputs. Account for array copies and backend temporaries, and run admitted work in an isolated worker process with externally enforced resource limits. See [resource boundaries](numerics.md#resource-boundaries-for-untrusted-requests) and the [security-policy draft](../SECURITY.md).

## Next steps

- [Python API and examples](../python/README.md)
- [JavaScript API and examples](../javascript/README.md)
- [Choosing an API](api.md)
- [Repository overview](../README.md)
