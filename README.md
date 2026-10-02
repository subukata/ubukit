<p align="center">
  <img src="docs/assets/ubukit-logo-b.png" alt="UbuKit" width="520">
</p>

# UbuKit

Python / JavaScriptで使える、クラスタリング・SOM・評価指標・軽量パラメータ探索のライブラリです。

- クラスタリング: k-means、FCM、RCM / ExRCM、RMCM
- SOM: オンラインSOM、BatchSOM、SOM-OLP
- 評価: Trustworthiness / Continuity、ARI / AMI
- パラメータ探索: TPE / random search

現在はprivate previewです。PyPI / npmには未公開です。[導入手順](docs/getting-started.md)に従い、ローカル配布物から利用してください。

## Python

```python
import numpy as np
import ubukit

X = np.array([[0.0], [0.1], [4.0], [4.1]], dtype=np.float64)
result = ubukit.fit_fcm(X, 2, random_state=1, max_iter=20, backend="numpy")
print(result["labels"])

som = ubukit.som_batch(X, grid_shape=(4, 3), epochs=2, random_state=1)
print(som["embedding"])
```

公開APIは `import ubukit` にまとまっています。[Python API](python/README.md)

## JavaScript

```javascript
import { run } from 'ubukit-js';

const X = { data: new Float64Array([0, 0.1, 4, 4.1]), nSamples: 4, nFeatures: 1 };
const result = run('kmeans', X, { nClusters: 2, seed: 1, maxIterations: 20 });
console.log(result.labels);
```

Node.jsとブラウザのES Modulesに対応。ランタイム依存はありません。[JavaScript API](javascript/README.md)

## ガイド

- [導入・ビルド](docs/getting-started.md)
- [API一覧と選び方](docs/api.md)
- [数値計算の制約](docs/numerics.md)
- [性能の読み方](docs/performance.md)
- [開発・検証](docs/contributing.md)
- [サンプル](examples/README.md)

Pythonの実装は `python/src/ubukit/`、JavaScriptの実装は `javascript/src/` にあります。
旧研究コード・過去の計測結果は、コミットに紐づく完全保存版から復元できます。[構成変更と保存方針](docs/repository-layout.md)

プロジェクト全体のライセンスは未選定です。各パッケージの第三者ライセンス・NOTICEは保持しています。
