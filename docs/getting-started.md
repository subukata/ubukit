# Getting started

UbuKit is a pre-release preview. Packages are not yet available from PyPI or npm; build them from a repository checkout or use a supplied local artifact.

Run the commands below from the repository root. The Python package declares Python 3.10 or later; the installation steps use the locally checked Linux / Python 3.12 stack. JavaScript requires Node.js 20 or later. Declared requirements do not imply that every version or platform has been tested.

## Python

```sh
python3.12 -m venv .venv-build
.venv-build/bin/python -m pip install build==1.4.0 setuptools==82.0.1 wheel==0.46.3
.venv-build/bin/python -m build --no-isolation --outdir python/artifacts python
python3.12 -m venv .venv
.venv/bin/python -I -B python/tools/check_install_environment.py
.venv/bin/python -m pip install -c python/constraints/constraints-namespace-verified.txt python/artifacts/*.whl
```

On Windows, replace a virtual environment's `bin/python` with `Scripts/python.exe`. If you already have a built wheel, skip the build steps. Use a fresh virtual environment rather than overwriting an older preview, and stop if the environment check reports a conflict.

Numba is optional. To enable it, install the same wheel with its `[numba]` extra and the [tested dependency constraints](../python/constraints/constraints-final-stack.txt).

## JavaScript

```sh
mkdir -p javascript/artifacts
cd javascript
npm pack --ignore-scripts --pack-destination artifacts
cd ..
npm install --offline --ignore-scripts ./javascript/artifacts/ubukit-js-0.1.0-dev.6.tgz
```

For browser use, serve `javascript/src/index.js` over HTTP and import it as an ES module. The [browser example](../examples/README.md) imports the bundled source directly. Serve Worker modules over HTTP(S).

## Next steps

- [Python API and examples](../python/README.md)
- [JavaScript API and examples](../javascript/README.md)
- [Choosing an API](api.md)
- [Repository overview](../README.md)
