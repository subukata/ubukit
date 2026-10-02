# UbuKit

クラスタリング・SOM-OLP・近傍評価・軽量ハイパーパラメータ探索の Python / JavaScript 研究用実装です。

## 最新実装: dev3 private preview

**最新の統合ソースは [`preview/`](preview/) です。** Python は `0.0.0.dev3`、JavaScript は `0.1.0-dev.3`。
Python の `import ubukit` に31の公開エクスポートをまとめ、ARI / AMI と TPE / random 探索を追加しました。
FCM / SOM の極端な数値範囲への対処、入力検証・所有権・Worker / Session の修正を含みます。
PyPI / npm には公開していません。リポジトリからローカルにビルドして使用します。

このページは**表示しているブランチの実装**を説明します。draft PR の間は main への統合済みを意味しません。
既存の `restart_v2/`、`fcm_challenge/`、`rcm_challenge/`、`rmcm_challenge/`、`javascript/` は過去の研究チェックポイントとして保持しています。
古い4パッケージ構成の利用ガイドは [LEGACY_README.md](LEGACY_README.md) にあります。

[導入](#インストール) · [Python API](preview/python/staging/README.md) · [JavaScript API](preview/javascript/package/README.md) · [数値上の制限](preview/NUMERICAL_LIMITS.md) · [再現手順](preview/REPRODUCE.md) · [検証範囲](preview/VERIFICATION.md)

## できること

| 目的 | Python (`ubukit`) | JavaScript |
|---|---|---|
| 各点を1クラスタに分ける | `fit_kmeans` | `run('kmeans', ...)` |
| 連続値の所属度 | `fit_fcm` | `run('fcm', ...)` |
| rough clustering | `fit_rcm`, `fit_exrcm` | `run('rcm', ...)`, `run('exrcm', ...)` |
| δ近傍に基づく rough clustering | `fit_rmcm`, `PreparedRMCM` | `run('rmcm', ...)`, `PreparedRMCM` |
| 格子への写像 | `fit_som_olp`, `PreparedSOM` | `run('som-olp', ...)` |
| Trustworthiness / Continuity | `joint_quality` | `run('neighborhood', ...)` |
| 外部クラスタ評価 | `adjusted_rand_score`, `adjusted_mutual_info_score`, `adjusted_scores` | 未収録 |
| 軽量 TPE / random 探索 | `optimize`, `TPEOptimizer` | `optimize`, `optimizeAsync`, `TPEOptimizer` |

JavaScript の ARI / AMI は別途追加作業中で、この dev3 スナップショットには含まれません。

Python の統一 API ではクラスタ所属度を `(N, K)` に揃えています。ただし返り値の型・フィールド名は手法ごとに異なります。
従来の `rough_cmeans` 直接 import は `(K, N)` のままです。JavaScript の配列は行優先の一次元 TypedArray です。

## インストール

### Python

Python >=3.10 を宣言していますが、以下の導入例は検証済み依存版を固定した CPython 3.12 向けです。実行検証は Linux x86-64 に限ります。
非公開リポジトリの取得にはアクセス権が必要です。表示中のブランチを取得し、ルートから実行してください。

**新しい仮想環境を使用してください。** 旧4パッケージは同じ import パスを所有するため、統合版との併存はできません。
`pip check` ではこの所有権の衝突を検出できません。

```sh
python3.12 -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell の場合: .venv\Scripts\Activate.ps1
python -m pip install -c preview/python/constraints-final-stack.txt ./preview/python/staging
```

任意の Numba バックエンドが必要な場合だけ、最後の行の代わりに次を使います。

```sh
python -m pip install -c preview/python/constraints-final-stack.txt './preview/python/staging[numba]'
```

NumPy・SciPy・scikit-learn・threadpoolctl は依存パッケージです。HPO は必須依存を追加しません。
宣言上の最小依存版、他OS、他CPUアーキテクチャ、他Python版は検証済みとはしていません。
厳密な検証スタックと wheel / sdist のビルド方法は [再現手順](preview/REPRODUCE.md) を参照してください。

```python
import numpy as np
import ubukit

X = np.array([[0.0], [0.1], [4.0], [4.1]], dtype=np.float64)
result = ubukit.fit_fcm(X, 2, random_state=1, max_iter=20, backend="numpy")
print(result["labels"])
print(result["membership"].shape)  # (4, 2)

search = ubukit.optimize(
    lambda p: (p["x"] - 0.3) ** 2,
    {"x": ubukit.float_range(-1.0, 1.0)},
    n_trials=30,
    seed=42,
)
print(search.best_params, search.best_value)
```

全手法の例は [all_methods.py](preview/python/examples/all_methods.py)、探索の契約は [OPTIMIZATION.md](preview/python/staging/OPTIMIZATION.md) にあります。

### JavaScript / Node

Node >=20 を宣言しています。検証に使用したのは Node 24.19.0 です。
旧 `javascript/` ではなく、最新の `preview/javascript/package/` を使用してください。

```sh
mkdir -p preview/javascript/artifacts
cd preview/javascript/package
npm pack --ignore-scripts --pack-destination ../artifacts
cd ../consumer
npm install --offline --ignore-scripts --no-audit --no-fund
```

`preview/javascript/consumer/` 内で実行できます。

```js
import { run, optimize, floatRange } from 'ubukit-js';

const input = {
  data: Float64Array.of(0, 0.1, 4, 4.1),
  nSamples: 4,
  nFeatures: 1,
};
const result = run('fcm', input, { nClusters: 2, seed: 1, maxIterations: 20 });
console.log(result.labels, result.membership);
const search = optimize(p => (p.x - 0.3) ** 2,
  { x: floatRange(-1, 1) }, { nTrials: 30, seed: 42 });
console.log(search.bestParams, search.bestValue);
```

ランタイム npm 依存はありません。同期実行、段階実行、Session、Worker、非同期探索を備えます。
ブラウザでは ES Modules と module Worker を適切に配信するサーバーまたは bundler が必要です。
時間予算は処理ブロック間で確認する soft limit で、FPSや最大応答時間の保証ではありません。
[JavaScript ガイド](preview/javascript/package/README.md) と [探索 API](preview/javascript/package/OPTIMIZATION.md) を参照してください。

## 数値と性能の注意

- FCM は有限の `m` に対する log-domain fallback を持ちます。復元した中心が丸められる場合、公開中心から再計算した所属度と返却所属度が一致しないことがあります。objective の overflow / underflow は状態を明示します
- 高い `m` の近接した軌道では有限精度の影響が大きく、所属度の精度が一様に改善するわけではありません
- SOM の表現可能な範囲は Python / JavaScript で異なります。表現不能な最終結果を無断で clip しません
- SOM の保守的な安全経路は、通常に標準化したデータでも極小の初期確率により遅くなる場合があります。実データHPO評価では840件の物理SOM評価中53件が2秒/fit以内に未完了でした
- TPE は flat / single-objective の軽量実装です。条件付き空間、pruning、分散ストレージ、汎用的な Optuna 優位は主張しません

[数値契約](preview/NUMERICAL_LIMITS.md)、[探索的な実データ評価](preview/HPO_FINDINGS_JA.md)、[小規模HPO計測](preview/OPTIMIZATION_PERFORMANCE_JA.md) を併せて読んでください。
速度はデータ・初期値・停止条件・thread数・計測範囲を揃えて比較してください。異なる条件の倍率は合算できません。

## 検証とデモ

dev3 の事前検証では Python 4つの新規 installed-artifact 環境と JavaScript の実 tarball を検証しました。
JavaScript は704テスト、Python は Numba 有無ごとの runtime / 新規数値・探索 / 外部指標テストを通過しています。
正確な件数、skip、対象外、今回のソースからの再実行結果は [VERIFICATION.md](preview/VERIFICATION.md) を参照してください。
84のランタイムファイルは [SOURCE_SNAPSHOT.json](preview/SOURCE_SNAPSHOT.json) で固定しています。

- [ブラウザ実データデモ](https://ubukit-browser-lab.ubucat.chatgpt.site): 同梱ライブラリは以前の版です。dev3 の全機能・数値契約を検証するデモではありません
- [進捗ダッシュボード](https://acceleration-progress.ubucat.chatgpt.site): 作業状況と検証記録の概要

上記サイトにはそれぞれのアクセス権が必要な場合があります。ブラウザのデモ動作と dev3 の全ブラウザ検証は区別します。

## ライセンスと公開範囲

このリポジトリは private preview です。プロジェクト全体の公開ライセンス・正式な配布名は未決定です。
既存の SOM や外部指標の表示は各パッケージの NOTICE / LICENSE ファイルに保持しています。
第三者コードのライセンスはプロジェクト全体への新たなライセンス付与を意味しません。
ソースと必要なテスト参照を保存し、仮想環境・node_modules・生成済み配布物・データセット本体は追加していません。
