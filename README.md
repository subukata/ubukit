# UbuKit

クラスタリング・SOM-OLP・次元削減の近傍評価を試すための、Python / JavaScript の研究用実装です。
**k-means、FCM、RCM / ExRCM、RMCM、SOM-OLP、T/C の6系統**を収録しています。
下の例はデータを含み、現行 `main` の API で実行できます。

[インストール](#インストール) · [Python の使い方](#python-の最小例) · [JavaScript](#javascript-版) · [リアルタイム処理](javascript/docs/REALTIME.md) · [速度表の読み方](#速度表の読み方)

## どれを使う？

| やりたいこと | 手法 / Python API | 最小実行ファイル |
|---|---|---|
| 各点を1つのクラスタに分ける | [k-means](#k-means) / `portable_accel.fit_kmeans` | [examples/kmeans.py](examples/kmeans.py) |
| 各点に連続値の所属度を付ける | [FCM](#fcm) / `ubukit_fcm.fit_fcm` | [examples/fcm.py](examples/fcm.py) |
| 距離の閾値で複数クラスタへの所属を許す | [RCM](#rcm) / `rough_cmeans.fit_rcm` | [examples/rcm.py](examples/rcm.py) |
| RCM の距離べき `p` も変える（同じ系統） | [ExRCM](#exrcm) / `rough_cmeans.fit_exrcm` | [examples/exrcm.py](examples/exrcm.py) |
| 固定した δ 近傍内のクラスタ占有率を使う | [RMCM](#rmcm) / `ubukit_rmcm.fit_rmcm` | [Python の使い方](rmcm_challenge/USAGE_JA.md) |
| 高次元データを格子上へ写す | [SOM-OLP](#som-olp) / `portable_accel.fit_som_olp` | [examples/som_olp.py](examples/som_olp.py) |
| 埋め込みが元の近傍を保っているか測る | [Trustworthiness / Continuity](#trustworthiness--continuity) / `portable_accel.joint_quality` | [examples/neighborhood_quality.py](examples/neighborhood_quality.py) |
| ブラウザ / Node.js から使う | [JavaScript 版](#javascript-版) / `run`, `createWorkerClient` | [examples/javascript.mjs](examples/javascript.mjs) |

現在は **4つの独立した Python パッケージと JavaScript 版**を収録しています。
`import ubukit` という統一 API はまだありません。初期収録の `portable_accel` は
`0.2.0a2`、FCM / RCM・ExRCM / RMCM / JavaScript は研究用の追加実装です。
このリポジトリからの導入を案内しており、PyPI / npm 公開済みとはしていません。

## インストール

### Python

Python **3.10 以上**。以下はリポジトリを取得した後、そのルートで実行します。
非公開リポジトリの取得にはアクセス権が必要です。

```sh
git clone https://github.com/subukata/ubukit.git
cd ubukit
python -m venv .venv
```

仮想環境を有効にします。

```sh
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

使う手法のパッケージをインストールしてください。全系統を試すなら4行とも実行します。

```sh
python -m pip install ./restart_v2     # k-means, SOM-OLP, T/C
python -m pip install ./fcm_challenge  # FCM
python -m pip install ./rcm_challenge  # RCM / ExRCM
python -m pip install ./rmcm_challenge # RMCM
```

- `restart_v2`: NumPy・SciPy・scikit-learn・threadpoolctl が自動で入ります
- `fcm_challenge`: NumPy・SciPy・threadpoolctl が自動で入ります
- `rcm_challenge`: NumPy が必須。SciPy / Numba は任意です
- `rmcm_challenge`: NumPy・SciPy・threadpoolctl が自動で入ります
- 下の Python 例は Numba 不要です。UbuKit 独自の C/C++ ビルドも必要ありません。依存先の wheel がない環境では依存パッケージのビルドが必要になる場合があります
- Python の対応宣言と実機検証範囲は別です。現時点の実機検証は Linux x86-64 / Python 3.12。Windows・macOS・ARM は未検証です

任意の Numba バックエンドを使うときだけ、対応する行を追加実行します。

```sh
python -m pip install './restart_v2[numba]'
python -m pip install './fcm_challenge[numba]'
python -m pip install './rcm_challenge[scipy,numba]'
python -m pip install './rmcm_challenge[numba]'
```

保存済み `portable_accel` wheel から導入する場合は、ソース導入の代わりに次を使えます。
これは k-means / SOM-OLP / T/C のみを含む `0.2.0a2` です。

```sh
python -m pip install ./restart_v2/evidence/distribution_a2/dist/portable_accel_restart-0.2.0a2-py3-none-any.whl
```

インストール後の動作確認は、ルートから `python examples/kmeans.py` または `python examples/fcm.py` で始められます。
他の手法は冒頭の表と以下のコピー可能な例を参照してください。

## Python の最小例

共通して入力 `X` は **行がサンプル、列が特徴量**の有限な数値配列 `(N, D)` です。
NaN / inf / 文字列は使えません。CSV なら、数値列だけを取り出してこの形にします。
距離を使うので、単位や桁の異なる特徴をどう正規化するかは呼出側で決めてください。
`N` は点数、`D` は特徴数、`K` はクラスタ数、`M` は SOM ノード数、`Q` は埋め込み次元数です。

| 出力 | Python | JavaScript（行優先の一次元 TypedArray） |
|---|---|---|
| クラスタ中心 | `centers`: `(K, D)` | `centers`: 長さ `K * D` |
| FCM 所属度 | `result["membership"]`: `(N, K)` | `membership`: 長さ `N * K` |
| RCM / ExRCM 所属度 | `result.memberships`: **`(K, N)`** | `membership`: 長さ `N * K` |
| RMCM 所属度 | `result.memberships`: `(N, K)` | `membership`: 長さ `N * K` |
| SOM 埋め込み | `result["V"]`: `(N, Q)` | `embedding`: 長さ `N * Q` |

Python の dict と属性アクセス、`membership` と `memberships`、配列の向きを混同しないでください。
JS の所属度の `(i, c)` は `membership[i * K + c]` です。

### k-means

クラスタ数は、渡した初期中心 `init` の行数で決まります。
自動で k-means++ 初期化や複数回リスタートを行う API ではありません。

```python
import numpy as np
from portable_accel import ExecutionPolicy, fit_kmeans

X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
result = fit_kmeans(
    X, init=X[[0, 3]], max_iter=30, backend="numpy",
    policy=ExecutionPolicy(threads=1),
)
print(result["centers"])  # (2, 2): 各クラスタの中心
print(result["labels"])   # [0 0 0 1 1 1]: 各点のクラスタ番号
print(result["inertia"])  # 約 2.6667: 最終中心への二乗距離和
```

`labels` は最終中心に対する割当です。`n_iter` は反復数。
詳しい [引数・戻り値・終了条件](docs/API_JA.md#k-means) と [実行ファイル](examples/kmeans.py)。

### FCM

`m > 1` は曖昧さの指数です。まずは `m=2`。
所属度は各点について合計1になり、最大の所属度のクラスタが `labels` です。

```python
import numpy as np
from ubukit_fcm import fit_fcm

X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
result = fit_fcm(
    X, n_clusters=2, m=2.0, random_state=4,
    backend="scipy", max_iter=100, tol=1e-5, threads=1,
)
print(result["centers"])     # (K, D)
print(result["membership"])  # (N, K): 行ごとに合計1
print(result["labels"])      # (N,)
print(result["objective"], result["converged"])
```

比較実験では `init=U0` に同一の初期所属度 `(N, K)` を渡します。
`tol=0` なら指定した `max_iter` 回を必ず実行します。
詳しい [API](docs/API_JA.md#fcm)、[数値契約](fcm_challenge/README.md)、[実行ファイル](examples/fcm.py)。

### RCM

ここでの RCM は、距離 `d` が `d <= alpha * d_min + beta` を満たすクラスタを選び、
選ばれたクラスタへ所属度を等分する方式です。中心はこの所属度による重み付き平均です。
`alpha >= 1` は相対的な幅、`beta >= 0` は距離と同じ単位の加算幅です。

```python
import numpy as np
from rough_cmeans import fit_rcm

X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
result = fit_rcm(
    X, n_clusters=2, init=X[[0, 3]],
    alpha=1.1, beta=0.5, backend="numpy", max_iter=100,
)
print(result.centers)       # (K, D)
print(result.memberships)   # (K, N): 列ごとに合計1。FCM と向きが逆
print(result.upper_memberships)  # (K, N): 所属するかどうかの bool
print(result.stop_reason, result.converged)
```

戻り値は dict ではなく属性を持つオブジェクトです。
複数所属を1つのラベルに潰すと重なりの情報を失うので、まず mask / 所属度を確認してください。
詳しい [API](docs/API_JA.md#rcm--exrcm)、[実行ファイル](examples/rcm.py)。

### ExRCM

RCM の判定を `d**p <= alpha**p * d_min**p + beta**p` に拡張します。
`p=1` は RCM、`p=2` は二乗距離で判定できます。`beta` はどの `p` でも距離と同じ単位です。

```python
import numpy as np
from rough_cmeans import fit_exrcm

X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
result = fit_exrcm(
    X, n_clusters=2, init=X[[0, 3]],
    p=2.0, alpha=1.1, beta=0.5, backend="numpy", max_iter=100,
)
print(result.centers)      # (K, D)
print(result.memberships)  # (K, N)
print(result.stop_reason, result.converged)
```

RCM / ExRCM は固定点・周期・反復上限を区別します。`converged=False` の結果を収束済みとして扱わないでください。
詳しい [API](docs/API_JA.md#rcm--exrcm)、[数式・数値契約](rcm_challenge/README.md)、[実行ファイル](examples/exrcm.py)。

### RMCM

Rough Membership C-means は、各点の **固定 δ 近傍内のクラスタ占有率**を所属度にする手法です。
近傍はユークリッド距離 `<= delta` で自己点を含みます。`delta >= 0` は入力の距離と同じ単位です。
RCM / ExRCM とは別系統で、RMCM2 は実装していません。

```python
import numpy as np
from ubukit_rmcm import fit_rmcm

X = np.array([[0.0], [1.0], [3.0], [8.0], [9.0]])
result = fit_rmcm(
    X, n_clusters=2, delta=2.0, random_state=42,
    backend="adjoint", max_iter=100, threads=1,
)
print(result.centers)      # (K, D)
print(result.memberships)  # (N, K): 行ごとに合計1
print(result.labels)       # (N,): 返却中心を生成した暫定 hard 割当
print(result.stop_reason, result.converged)
```

返却所属度・labels は**返却中心を生成した状態**です。最終中心への再割当ではなく、所属度最大のラベルとも限りません。
一般的な収束保証はなく、`fixed_point` / `cycle` / `max_iter` を区別します。
`cycle_window` は検出する履歴の長さで、長い周期を見逃す場合があります。
ほぼ同距離の境界では加算順の丸め差が軌道を変えることがあります。

同じ `X, delta` で初期値やクラスタ数を変える場合は `prepare_rmcm(X, delta)` の
`.fit(n_clusters=...)` で近傍と前計算を再利用できます。入力や δ を変えたら準備し直します。
`max_edges`（既定10,000,000）は自己辺を含む有向候補辺数の制限で、総メモリ上限ではありません。

詳しい [使い方・出力契約](rmcm_challenge/USAGE_JA.md)、[数式](rmcm_challenge/THEORY.md)、
[Python 性能](rmcm_challenge/REPORT.md)、[JS API](javascript/docs/RMCM.md)、
[JS 実行例](javascript/examples/rmcm.js)、[JS 性能](javascript/docs/RMCM_BENCHMARK.md)。

### SOM-OLP

`R` は出力側の格子座標 `(M, Q)`。下は4ノードの2次元格子です。
`gamma >= 0` は格子側の項の重み、`lam > 0` はエントロピー正則化の重みです。

```python
import numpy as np
from portable_accel import ExecutionPolicy, fit_som_olp

X = np.random.default_rng(4).normal(size=(24, 3))
R = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
result = fit_som_olp(
    X, R, gamma=0.5, lam=1.0, max_iters=50, tol=1e-4,
    backend="threadpool", initializer="svd_lowrank",
    policy=ExecutionPolicy(threads=1),
)
print(result["V"].shape)  # (24, 2): 可視化・近傍評価に使う埋め込み
print(result["W"].shape)  # (4, 3): 元の特徴空間でのノード位置
print(result["P"].shape)  # (24, 4): 点からノードへの所属度
```

参照計算は `backend="cdist", initializer="original"`。
同じ X で条件を何度も変える場合は [PreparedSOM の例](examples/prepared_som.py) で完全 SVD を共有できます。
元実装の更新順序を保持しており、返却 `V` は返却 `P @ R` と一般には同じではありません。
詳しい [API と返却時点](docs/API_JA.md#som-olp)、[実行ファイル](examples/som_olp.py)。

### Trustworthiness / Continuity

`X` は元のデータ、`Y` は**同じ点を同じ行順で並べた**埋め込みです。
SOM の結果を評価するなら `Y = som_result["V"]` に置き換えます。

```python
import numpy as np
from portable_accel import ExecutionPolicy, joint_quality

X = np.random.default_rng(4).normal(size=(24, 3))
Y = X[:, :2].copy()
qualities = joint_quality(
    X, Y, ks=[3, 5], backend="numpy",
    max_distance_bytes=64 * 2**20,
    policy=ExecutionPolicy(threads=1),
)
for q in qualities:
    print(q.k, q.trustworthiness, q.continuity)
```

T は「埋め込みで近くなった偽の近傍」、C は「埋め込みで失われた元の近傍」を評価します。
どちらも0〜1で、大きいほど近傍を保っています。`1 <= k < N/2` が必要です。
`ks=3` のように1つだけ指定しても、戻り値はリストです。

**Python 版の距離行列は O(N²) メモリです。** `max_distance_bytes` は最大の距離行列1枚に対する事前チェックで、プロセス全体のメモリ上限ではありません。
詳しい [API](docs/API_JA.md#trustworthiness--continuity)、[実行ファイル](examples/neighborhood_quality.py)。

## JavaScript 版

Python とは独立した、外部依存なしの ES Modules / Float64Array CPU 実装です。
Node.js **20 以上**。依存パッケージのインストールなしで、ルートから実行できます。
1本目は k-means / FCM / RCM / ExRCM / SOM-OLP / T/C、2本目は RMCM の例です。

```sh
node examples/javascript.mjs
node javascript/examples/rmcm.js
```

自分のコードから呼ぶ最小例です。ルートに `quickstart.mjs` として保存し、`node quickstart.mjs` で実行します。

```javascript
import { run } from './javascript/src/index.js';

const input = {
  data: Float64Array.of(0, 0, 0, 1, 8, 8, 8, 9),
  nSamples: 4, nFeatures: 2,
};
const result = run('fcm', input, {
  nClusters: 2, m: 2, seed: 4, maxIterations: 100,
});
console.log(result.centers, result.membership, result.converged);
```

`data` は行優先で、長さは必ず `nSamples * nFeatures`。Python の引数名をそのまま渡さず、次を使います。

| `run` の手法名 | 主な options | 主な戻り値 |
|---|---|---|
| `'kmeans'` | `nClusters`, `initCenters`, `maxIterations` | `centers`, `labels`, `inertia` |
| `'fcm'` | `nClusters`, `m`, `tolerance` | `centers`, `membership`, `objective`, `converged` |
| `'rcm'` / `'exrcm'` | `nClusters`, `alpha`, `beta`（ExRCM は `p` も指定） | `centers`, `membership`, `mask`, `stopReason` |
| `'rmcm'` | `nClusters`, `delta`（必須）, `backend: 'adjoint'` | `centers`, `membership`, `labels`, `stopReason` |
| `'som-olp'` | `grid`（入力と同じ形式）, `gamma`, `lambda` | `embedding`, `prototypes`, `memberships` |
| `'neighborhood'` | `embedding`（入力と同じ形式）, `ks` | `qualities`: `k`, `trustworthiness`, `continuity` の配列 |

クラスタリングの初期中心 `initCenters` は長さ `K * D`。FCM の `initMembership` は長さ `N * K` です。
RMCM に FCM の `m` や RCM の `alpha` / `beta` / `p` を渡すことはできません。
Python では SOM の `lam` / `max_iters`、JS では `lambda` / `maxIterations` を使います。
T/C は入力と埋め込みの行順を揃え、`1 <= k < N/2` とします。

ブラウザのデモをローカルで開く場合:

```sh
cd javascript
npm run demo
```

表示された `http://localhost:8765/` をブラウザで開きます。`file://` ではなく HTTP で配信します。
画面を固めないよう、組み込みには `createWorkerClient` を使ってください。

```javascript
import { createWorkerClient } from './javascript/src/index.js';

const client = createWorkerClient();
try {
  const result = await client.run('fcm', {
    data: Float64Array.of(0, 0, 0, 1, 8, 8, 8, 9),
    nSamples: 4, nFeatures: 2,
  }, { nClusters: 2, m: 2, seed: 4 });
  console.log(result.centers, result.membership);
} finally {
  client.dispose();
}
```

これはリポジトリのルートを配信した HTML 内の `<script type="module">` 用です。
自分のサイトでは import 先を配置場所に合わせます。
手法ごとの [JavaScript 入出力・呼出例](docs/JAVASCRIPT_JA.md)、[Worker・キャンセル等の仕様](javascript/README.md)。
Python と JS では seed の乱数系列や同距離順位規則が異なるため、同じ seed だけで結果一致は保証しません。

## JavaScript の状態保持・リアルタイム処理

動くデータには `createSession` を使い、`step` で段階実行、`snapshot` で完了した反復を取得できます。`updateData` は点の移動・発生・消滅、`updateParameters` はパラメータ変更を扱い、互換な中心を保持しながら古い所属度や近傍グラフを明示的に無効化します。

Session は k-means / FCM / RCM / ExRCM / RMCM / SOM-OLP に対応します。
`neighborhood` は Session の手法名ではなく、単発の `run` または `createMetricScheduler` で評価します。
`step` の戻り値は進行状況で、数値結果は `snapshot().result` にあります。
最初の反復が完了する前は結果が `null`、途中では最後に完了した結果を返します。
終了処理にも step が必要なため、`session.status.done` まで進めてください。

- [実行例](javascript/examples/realtime.js): `node javascript/examples/realtime.js`
- [Session / 最新入力優先 Worker / 近傍評価 cache の契約](javascript/docs/REALTIME.md)
- [初回・暖機後・step p50/p95 を分けた計測](javascript/docs/REALTIME_BENCHMARK.md)
- [Python との対応と更新契約](javascript/docs/PYTHON_REALTIME_CONTRACT.md)

時間予算は処理ブロック間で確認する soft limit です。厳密な応答時間や FPS を保証しません。Python は現行の batch / prepared API を維持しており、今回の状態保持 API は JavaScript 向けです。既存の `run` 等の出力は変えていません。

## 困ったとき

- `No module named portable_accel` 等: スクリプトを実行するのと同じ `python` で `python -m pip install ...` したか確認してください。リポジトリのルート自体は Python パッケージではありません
- Numba エラー / 初回だけ遅い: まず例の `numpy` / `scipy` を使ってください。Numba の初回はコンパイル時間が入ります
- 手法を変えたら所属度の shape が合わない: Python FCM / RMCM は `(N, K)`、Python RCM / ExRCM は `(K, N)`、JS のクラスタ所属度は行優先 `(N, K)` です
- 大きいデータでメモリが足りない: まず少数点で確認してください。特に Python T/C の距離行列、FCM `numpy` の `(N, K, D)` 中間配列に注意してください
- `auto` は常に最速という意味ではありません。利用可否による選択です。[バックエンドと引数の一覧](docs/API_JA.md) を参照してください

## 速度表の読み方

[実装別の性能と再計測](docs/PERFORMANCE_JA.md)、
[Python / JavaScript の同一条件比較](docs/PYTHON_VS_JAVASCRIPT_JA.md)、
[RMCM Python](rmcm_challenge/REPORT.md) / [RMCM JS](javascript/docs/RMCM_BENCHMARK.md) に測定値をまとめています。

| 表の項目 | 意味・比較時の注意 |
|---|---|
| ms / 秒 | 小さいほど速い。表ごとの単位を確認 |
| 参照 / UbuKit、速度倍率 | 参照時間 ÷ 対象時間。1 より大きければ対象が速く、0.69× は高速化ではない |
| fit 全体 / end-to-end | その記録が定義する入力検証・初期化・最終出力処理を含む範囲。ファイル読込・Worker 転送まで含むとは限らない |
| prepared / 更新部分 | 近傍構築・SVD 等を除外する場合がある。初回の全処理と直接比較しない |
| cold / warm | 初回の import・JIT 等と、暖機後の実行を分ける |
| p50 / p95 | 測定した遅延の50 / 95パーセンタイル。平均でも最悪時間の保証でもない |

同じデータ形状・初期値・反復数または停止条件・スレッド数・出力契約で比較してください。
RCM / ExRCM と RMCM の結果や、異なる参照実装に対する倍率は合算できません。
既存の JS 比較には **FCM / SOM で高速化しなかった条件**も含まれます。
Node の同期測定はブラウザの FPS や Worker 通信込みの速度を示しません。
研究中の未統合バックエンド・実験結果は、この README の利用可能な機能には含めていません。

## テスト・仕様・性能

- 利用者向け: [Python API](docs/API_JA.md)、[JavaScript API](docs/JAVASCRIPT_JA.md)、[最小例](examples/)
- k-means / SOM / T/C: [API・数値契約](restart_v2/README.md)、[結果索引](restart_v2/RESULT_INDEX.md)、[最終レポート PDF](restart_v2/evidence/report_stable/restart_v2_final_report_ja.pdf)
- FCM: [実装・テストの説明](fcm_challenge/README.md)、[数式・数値安定性](fcm_challenge/THEORY.md)
- RCM / ExRCM: [定義・テスト・再現手順](rcm_challenge/README.md)
- RMCM: [定義・テスト・再現手順](rmcm_challenge/README.md)
- JavaScript: [検証範囲](javascript/docs/VALIDATION.md)
- 性能: [比較条件付きの要約と再計測方法](docs/PERFORMANCE_JA.md)

初期収録部分のテストはルートで次を実行します。

```sh
python -m unittest discover -s restart_v2/tests -v
```

追加部分はライブラリ本体・テスト・実行例・数値仕様・再計測スクリプトを収録しています。
実験中の全ログ・旧ソースのコピー・生成済み配布物を使い方の前提にはしていません。
追加パッケージのテスト手順は各 README にあります。最小例の成功と全テスト・性能測定の成功は区別してください。
性能値は各記録のデータ・スレッド数・初期化・計時範囲に限られ、異なる条件の倍率は合算できません。

## 保存元・履歴・ライセンス

`restart_v2/` の303ファイルは初期登録時のチェックポイントを保持しています。
このディレクトリ内の「GitHub push は未実施」などは、その記録を作った時点の履歴です。
今回の利用ガイドと実行例は外側に追加し、過去の数値コード・測定記録を変更していません。

- 保存元: `restart_v2_checkpoint.zip`（748,902 bytes）
- SHA-256: `d04316dd1bb513416aae72a78a95b6230b319a09d24f6c7259194550d91aa859`
- [チェックポイント内の SHA-256 一覧](restart_v2/CHECKPOINT_MANIFEST.json)（自身を除く302ファイル）
- [a2 配布検証](restart_v2/evidence/distribution_a2/) と [旧 a1 配布検証](restart_v2/evidence/distribution/) は別の記録です
- データセット本体・仮想環境・JIT キャッシュ・ネイティブビルド生成物は収録していません。実験スクリプトには測定当時のパスが残り、再実行に調整が必要なものがあります

プロジェクト全体の公開ライセンスは未選定です。
第三者由来の表示は [NOTICE.txt](restart_v2/NOTICE.txt)、[LICENSE-SOM.txt](restart_v2/LICENSE-SOM.txt)、
[JavaScript の SOM-OLP 表示](javascript/LICENSES/SOM-OLP-MIT.txt) に保持しています。
これらはプロジェクト全体への新たなライセンス付与を意味しません。

