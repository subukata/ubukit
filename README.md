<p align="center">
  <img src="docs/assets/ubukit-logo-b.png" alt="UbuKit" width="520">
</p>

# UbuKit

Python / JavaScriptで使える、クラスタリング・SOM・評価指標・軽量パラメータ探索のライブラリです。

- クラスタリング: k-means、FCM、rough clustering
- 次元削減・可視化: オンラインSOM、BatchSOM、SOM-OLP
- 評価: Trustworthiness / Continuity、ARI / AMI
- パラメータ探索: TPE / random search

現在はprivate previewです。PyPI / npmには未公開のため、ローカル配布物から導入してください。

## Python

Python 3.10以上を宣言しています。以下は検証済みのLinux / Python 3.12向けです。リポジトリのルートで、新しい仮想環境を作ってビルド・導入します。

```sh
python3.12 -m venv .venv-build
.venv-build/bin/python -m pip install build==1.4.0 setuptools==82.0.1 wheel==0.46.3
.venv-build/bin/python -m build --no-isolation --outdir preview/python/artifacts preview/python/staging
python3.12 -m venv .venv
.venv/bin/python -I -B preview/python/tools/check_install_environment.py && \
  .venv/bin/python -m pip install -c preview/python/constraints-namespace-verified.txt preview/python/artifacts/ubukit_bundled_local_preview-0.0.0.dev6-py3-none-any.whl
```

配布済みのwheelがあればビルドを省き、最後のコマンドにそのパスを指定できます。以下の例は `.venv/bin/python` で実行してください。

```python
import numpy as np
import ubukit

X = np.array([[0.0], [0.1], [4.0], [4.1]], dtype=np.float64)
result = ubukit.fit_fcm(X, 2, random_state=1, max_iter=20, backend="numpy")
print(result["labels"])

som = ubukit.som_batch(X, grid_shape=(4, 3), epochs=2, random_state=1)
print(som["embedding"])
```

公開APIは `import ubukit` にまとまっています。Numbaを使う場合は同じwheelの `[numba]` extraと `preview/python/constraints-final-stack.txt` を指定してください。

## JavaScript

Node.js 20以上。配布されたtgzを指定します。未所持の場合は[ソースから作成](preview/REPRODUCE.md#javascript-build-install-and-all-tests)してください。

```sh
npm install --offline --ignore-scripts /path/to/ubukit-js-0.1.0-dev.6.tgz
```

例を `example.mjs` として保存し、`node example.mjs` で実行します。

```js
import { run } from 'ubukit-js';

const input = {
  data: Float64Array.of(0, 0.1, 4, 4.1),
  nSamples: 4,
  nFeatures: 1
};
const result = run('kmeans', input, { nClusters: 2, seed: 1 });
console.log(result.labels);
```

JavaScriptはランタイム依存なしのES Modulesです。Session、Worker、非同期実行にも対応しています。ブラウザで使う場合はmodule Workerを配信できるサーバーまたはbundlerを用意してください。

## ドキュメント

- [Python API](preview/python/staging/README.md) / [JavaScript API](preview/javascript/package/README.md)
- SOM: [Python](preview/python/staging/SOM.md) / [JavaScript](preview/javascript/package/SOM.md)
- パラメータ探索: [Python](preview/python/staging/OPTIMIZATION.md) / [JavaScript](preview/javascript/package/OPTIMIZATION.md)
- [ソースからのビルド・テスト](preview/REPRODUCE.md)
- [数値上の制限](preview/NUMERICAL_LIMITS.md) / [性能とbackendの選択](preview/EFFICIENCY.md)
- [Python旧版からの移行](preview/NAMESPACE_MIGRATION.md)

最新ソースは [`preview/`](preview/) にあります。トップレベルの旧実装は過去の研究版です。プロジェクト全体の公開ライセンスは未決定です。第三者コードの表示は各パッケージのNOTICE / LICENSEを参照してください。
