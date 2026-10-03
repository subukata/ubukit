# Getting started

UbuKit is alpha software. Build it from a repository checkout or install a supplied local archive using the commands below. These instructions do not require a registry release.

Run repository commands from the repository root. Python declares Python 3.10 or later and JavaScript declares Node.js 20 or later. Existing local evidence covers Linux / CPython 3.12 and Linux / Node 24; minimum-version declarations are not a tested platform matrix.

## Python

Build a wheel, then install it in a fresh environment with the checked base dependency stack:

```sh
python3.12 -m venv .venv-build
.venv-build/bin/python -m pip install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: --no-deps -r tools/ci/constraints/installer.txt
.venv-build/bin/python -m pip install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: -r tools/ci/constraints/build-py312-locked.txt
.venv-build/bin/python -m build --no-isolation --outdir python/artifacts python
python3.12 -m venv .venv
.venv/bin/python -I -B python/tools/check_install_environment.py
.venv/bin/python -m pip install -c python/constraints/constraints-namespace-verified.txt python/artifacts/ubukit-0.1.0a2-py3-none-any.whl
.venv/bin/python -c "import ubukit; print(ubukit.__version__)"
```

If you already have the wheel, use the self-contained local installation commands in the [Python README](../python/README.md#installation-boundary). Use a fresh environment when moving from earlier previews: overlapping installations can have conflicting file ownership. Stop if the repository's environment guard reports a conflict. On Windows, a virtual environment uses `Scripts/python.exe` instead of `bin/python`; this path convention is not a Windows qualification claim.

Numba is optional and enabled through the wheel's `[numba]` extra. The [historical optional-backend constraints](../python/constraints/constraints-final-stack.txt) record an earlier checked stack; they do not qualify Numba for this alpha. Use the base installation above unless you separately verify the optional backend in your environment.

## JavaScript

```sh
mkdir -p javascript/artifacts
cd javascript
npm pack --ignore-scripts --pack-destination artifacts
cd ..
npm install --offline --ignore-scripts --no-audit --no-fund ./javascript/artifacts/ubukit-js-0.1.0-alpha.2.tgz
```

If you already have the tarball, install it using its local path and skip `npm pack`. The package has no runtime npm dependencies.

For browser use, serve the ES modules and Worker modules over HTTP(S), or use a bundler that preserves module Worker URLs. The [browser example](../examples/README.md) imports `javascript/src/index.js` directly from the checkout. Actual browser execution, including EFCM, remains unverified by the current checks.

## Next steps

- [Python API and examples](../python/README.md)
- [JavaScript API and examples](../javascript/README.md)
- [Choosing an API](api.md)
- [Repository overview](../README.md)
