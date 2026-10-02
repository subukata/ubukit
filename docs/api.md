# API

公開APIと戻り値の詳しい契約は [Python](../python/README.md) / [JavaScript](../javascript/README.md) を参照してください。

- クラスタリング: k-means、FCM、RCM / ExRCM、RMCM
- SOM: online / batch / SOM-OLP。アルゴリズムごとに反復順序・初期化・所属度の意味が異なります
- 評価: 近傍品質（Trustworthiness / Continuity）と外部指標（ARI / AMI）
- 探索: TPE / random search。PythonとJavaScriptで共有契約を検証しています

Pythonの公開入口は `ubukit` です。`ubukit._impl` は内部実装で、安定した公開APIではありません。JavaScriptの公開入口は `ubukit-js` と package.json で指定するサブパスです。

[数値の制約](numerics.md) / [実行例](../examples/README.md)
