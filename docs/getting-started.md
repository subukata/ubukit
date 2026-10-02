# 導入

private previewのため、PyPI / npmからの一般公開パッケージ導入はまだできません。以下はリポジトリのルートで実行します。Pythonは3.10以上を宣言し、ローカル検証はLinux / Python 3.12です。Node.jsは20以上です。

## Python

```sh
python3.12 -m venv .venv-build
.venv-build/bin/python -m pip install build==1.4.0 setuptools==82.0.1 wheel==0.46.3
.venv-build/bin/python -m build --no-isolation --outdir python/artifacts python
python3.12 -m venv .venv
.venv/bin/python -I -B python/tools/check_install_environment.py
.venv/bin/python -m pip install -c python/constraints/constraints-namespace-verified.txt python/artifacts/*.whl
```

Windowsでは仮想環境の `bin/python` を `Scripts/python.exe` に読み替えてください。配布済みwheelがあればビルドは不要です。古いpreviewが入った環境へ上書きせず、新しい仮想環境を使ってください。
Numbaは任意です。同じwheelの `[numba]` extraと [検証済み制約](../python/constraints/constraints-final-stack.txt)を指定してください。

## JavaScript

```sh
mkdir -p javascript/artifacts
cd javascript
npm pack --ignore-scripts --pack-destination artifacts
cd ..
npm install --offline --ignore-scripts ./javascript/artifacts/ubukit-js-0.1.0-dev.6.tgz
```

ブラウザでは `javascript/src/index.js` をHTTP経由でimportできます。[実在するブラウザ例](../examples/README.md)は同梱ソースを直接参照します。WorkerはHTTP(S)で提供してください。

[Python API](../python/README.md) / [JavaScript API](../javascript/README.md) / [README](../README.md)
