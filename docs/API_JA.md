# Python API: 引数・戻り値・使い分け

[README に戻る](../README.md)。以下は同梱ソースの API です。
`portable_accel`、`ubukit_fcm`、`rough_cmeans` は別パッケージなので、使用するものをそれぞれ導入します。

## 共通の配列と設定

- `N`: サンプル数、`D`: 特徴数、`K`: クラスタ数、`M`: SOM ノード数、`Q`: 埋め込み次元数
- 入力 `X`: `(N, D)` の非空・有限な実数配列。DataFrame は数値列を選んで `.to_numpy()` 等で渡します
- k-means / SOM / FCM / RCM・ExRCM は float64 で計算します。T/C は float32 入力を保持し、それ以外を通常 float64 にします
- 自動スケーリング・欠損補完は行いません。極端な桁では明示エラーになる場合があります
- 各例の小さなデータは呼び出し確認用です。速度やクラスタ品質を比較するデータではありません

### ExecutionPolicy

`portable_accel` の3手法と `PreparedSOM.fit` で使います。FCM / RCM に渡す設定ではありません。

```python
from portable_accel import ExecutionPolicy
policy = ExecutionPolicy(threads=1, block_rows=256, max_scratch_bytes=32 * 2**20)
```

| 引数 | 既定値 | 意味 |
|---|---|---|
| `threads` | `1` | スレッド数。`None` は既存制限を引き継ぐが、一部 SOM 経路は明示値が必須 |
| `block_rows` | `256` | 一部の作業配列を処理する行数の上限 |
| `max_scratch_bytes` | `32 MiB` | 対象となる主要作業配列の予算 |

`max_scratch_bytes` は総メモリ上限ではありません。入力・出力・SVD・ライブラリ内部領域・T/C の距離行列などは別です。
スレッド制限はプロセス全体へ作用するため、独立した異なる設定で同時実行する場合は別プロセスを使います。

## k-means

[最小例](../examples/kmeans.py) / [実装](../restart_v2/portable_accel/kmeans.py)

```python
fit_kmeans(X, init, *, max_iter=20, backend="auto", policy=None,
           finalize=True, finalizer="sklearn")
```

| 引数 | 指定内容 |
|---|---|
| `X` | `(N, D)` |
| `init` | **必須**の初期中心 `(K, D)`。クラスタ数は行数から決まる |
| `max_iter` | 正の整数。ラベルが変化しなくなれば上限より早く終了 |
| `backend` | `numpy` / `numba` / `numba_blas` / `numba_blas_vector` / `auto` |
| `finalize` | 通常は `True` のまま。最終中心への割当と inertia を再計算 |
| `finalizer` | 最終割当の実装。`sklearn` または任意依存を使う `numba` |

`backend="auto"` は Numba が利用可能なら `numba`、そうでなければ `numpy`。
Numba 初回には JIT 時間が入ります。Numba BLAS 系は距離の式変形と近接 tie の修復を使い、
全入力での bit 単位の一致は保証しません。

戻り値は dict:

| キー | 内容 |
|---|---|
| `centers` | `(K, D)` 最終中心 |
| `labels` | `(N,)`、既定では最終中心へのクラスタ番号 |
| `inertia` | その labels / centers による二乗距離和 |
| `n_iter` | 実施した中心更新反復数 |
| `core_labels` | 最終の中心更新より前の割当 |
| `backend`, `finalizer`, `finalized_labels` | 実際の処理経路 |

`finalize=False` は研究比較用で `labels=core_labels`、`inertia=None` になります。
空クラスタは前の中心を保持します。scikit-learn `KMeans` の初期化・停止条件・空クラスタ処理を完全再現する API ではありません。

## FCM

[最小例](../examples/fcm.py) / [詳しい数値契約](../fcm_challenge/README.md)

```python
from ubukit_fcm import fit_fcm, fit_fcm_numpy

fit_fcm(X, n_clusters=None, *, init=None, m=2.0, max_iter=300,
        tol=1e-5, backend="auto", threads=1, random_state=None,
        return_history=False)
```

| 引数 | 指定内容 |
|---|---|
| `n_clusters` | 初期所属度を省略するときに必要な `K` |
| `init` | 初期所属度 `(N, K)`。非負・有限で各行の合計が正。コピー上で行正規化される |
| `m` | 有限かつ `> 1`。所属度を中心計算で `U**m` として使う指数 |
| `max_iter`, `tol` | `||U_new - U_old||_F < tol` で終了。`tol=0` は必ず上限回数まで実行 |
| `random_state` | 初期所属度を作る NumPy 乱数の seed。比較には `init` 推奨 |
| `threads` | BLAS スレッド上限。`numba_parallel` の worker 数にも適用 |
| `return_history` | `True` なら目的関数履歴 `objective_history` を追加 |

| backend | 使い分け |
|---|---|
| `scipy`, `auto` | SciPy `cdist` を使う、JIT 不要の既定経路 |
| `numpy` | `(N,K,D)` の差分配列を作る単純な比較用実装。大規模入力はメモリに注意 |
| `blas` | 中心化した Gram 形式と桁落ち修復。高次元の候補 |
| `numba` | 任意依存。逐次の融合カーネル |
| `numba_parallel` | 任意依存。並列の融合カーネル |

`fit_fcm_numpy(X, ...)` は `backend="numpy"` を使う簡便な関数です。

戻り値 dict の主要キー:

- `centers`: `(K, D)`、`membership`: `(N, K)`、`labels`: `(N,)` の所属度最大のクラスタ番号
- `objective`: 返却された中心と所属度で評価した `sum(U**m * squared_distance)`
- `fpc`: fuzzy partition coefficient、`n_iter`: 反復数、`converged`: 収束判定
- `delta`: 最後の所属度変化の Frobenius norm、`backend`: 実際に使った経路

最終中心は更新前の U から求め、返却 U はその後で更新したものです。返却直前に隠れた中心再計算はしません。
距離ゼロの中心が複数あれば、それらに所属度を等分します。空クラスタは前の中心（初回は全体平均）を保持します。

## RCM / ExRCM

[RCM 最小例](../examples/rcm.py) / [ExRCM 最小例](../examples/exrcm.py) / [定義と制限](../rcm_challenge/README.md)

```python
from rough_cmeans import fit_rcm, fit_exrcm, assign

fit_exrcm(X, n_clusters, *, alpha=1.1, beta=0.0, p=1.0,
          init=None, seed=0, max_iter=300, backend="auto",
          block_size=4096, return_memberships=True, cycle_window=16)
# fit_rcm は同じ引数を受け取り、p=1 に固定します。
```

| 引数 | 指定内容 |
|---|---|
| `n_clusters` | 正の整数 `K <= N` |
| `init` | 初期中心 `(K, D)`。省略時は seed を使いデータの異なる行番号から選択 |
| `alpha`, `beta`, `p` | 有限で `alpha>=1`, `beta>=0`, `p>0` |
| `max_iter` | 正の整数。反復上限 |
| `block_size` | ブロック処理の行数 |
| `return_memberships` | `False` なら最終の密な所属度・mask の返却を省く |
| `cycle_window` | 最近の何状態まで周期を探すか。`0` は周期検出なし |

判定は Euclidean distance `d[c,i]` を使い、
`M[c,i] = (d[c,i]**p <= alpha**p * min_j(d[j,i])**p + beta**p)`。
`U[c,i] = M[c,i] / sum_j(M[j,i])` とし、中心は U の重み付き平均です。
通常の rough-c-means にある別の lower / upper 重みは導入していません。
`beta=0` では実数演算上 p が打ち消されるので、p だけを変えても判定規則は変わりません。

- `backend="auto"`: SciPy があれば `scipy`、なければ `numpy`
- `numpy`: ブロック化した直接差分、`scipy`: ブロック化した `cdist`
- `numba`: 任意の逐次融合カーネル、`naive`: `(N,K,D)` の単純参照実装

戻り値は `RoughCMeansResult` オブジェクト。**dict ではありません。**

| 属性 | 内容 |
|---|---|
| `.centers` | `(K, D)` |
| `.memberships` | `(K, N)`、各列の合計1。省略時は `None` |
| `.upper_memberships` | `(K, N)` の bool mask。省略時は `None` |
| `.n_iter` | 中心更新の反復数 |
| `.converged`, `.fixed_point` | 固定点確認の結果 |
| `.stop_reason` | `fixed_point` / `cycle` / `max_iter` |
| `.cycle_length` | 検出した周期の長さ、なければ `None` |
| `.empty_cluster_updates` | 空クラスタで前の中心を保持した回数 |

返却 U は、返却中心に対して再計算されます。収束しなかったときは、返却中心がその U の重み付き平均になっているとは限りません。
`return_memberships=False` でも固定点・周期判定用の packed mask と履歴のメモリは必要です。

学習せず、新しいデータへ既存中心で所属を付ける場合:

```python
# X_new: (新しいN, D)、centers: 学習済み (K, D)
U, M = assign(X_new, centers, alpha=1.1, beta=0.5, p=2, backend="numpy")
```

データを再スケールする際は `beta` も同じ倍率で変えてください。

## SOM-OLP

[最小例](../examples/som_olp.py) / [実装](../restart_v2/portable_accel/som_olp.py)

```python
fit_som_olp(X, R, *, gamma, lam, max_iters=100, tol=1e-4,
            pca_scale=2.0, backend="cdist", policy=None,
            initializer="original")
```

| 引数 | 指定内容 |
|---|---|
| `X` | `(N, D)` |
| `R` | 格子座標 `(M, Q)`。通常 Q=2 |
| `gamma` | 必須。有限で `>=0`。格子側コストの重み |
| `lam` | 必須。有限で `>0`。エントロピー項の重み |
| `max_iters` | 反復上限。通常は正の整数。`0` では `V=None` |
| `tol` | 目的関数の相対変化に使う非負の許容値 |
| `pca_scale` | PCA 初期ノード配置のスケール |
| `initializer` | `original` / `svd_lowrank`。どちらも同じ完全 SVD を使う |

バックエンド:

- `cdist`: 参照経路。`numpy` / `auto` もこの経路へ解決します
- `threadpool`: 行ブロックを分担。`ExecutionPolicy(threads=...)` の明示値が必要
- `gemm_guarded`, `gemm_centered`, `cdist_optimized`: 明示選択する比較用経路。いずれも threads の明示値が必要
- `svd_lowrank` 初期化にも threads の明示値が必要です。低ランク P0 評価を使い、条件によって直接計算へ戻ります

戻り値 dict:

- `V`: `(N, Q)` の埋め込み、`W`: `(M, D)` のプロトタイプ、`P`: `(N, M)` の所属度
- `history`: 目的関数履歴、`n_iter`: 反復数、`backend`: 実際の経路、`initializer_requested`: 指定した初期化

1反復の順序は **旧 P → V/W → コスト → 新 P** です。
元実装と同様に V/W は最後の反復に入る時点の P から求め、返す P はその後の更新結果です。
したがって `V == P @ R` を返却値同士で仮定しないでください。

### 初期値を揃えて比較する

```python
from portable_accel import initialize_som_olp, run_som_olp
W0, P0 = initialize_som_olp(X, R, lam=1.0, policy=policy)
a = run_som_olp(X, R, W0, P0, gamma=0.5, lam=1.0,
                backend="cdist", policy=policy)
b = run_som_olp(X, R, W0, P0, gamma=0.5, lam=1.0,
                backend="threadpool", policy=policy)
```

### 同じ X で繰り返す: PreparedSOM

[独立して実行できる例](../examples/prepared_som.py)

```python
from portable_accel import PreparedSOM
prepared = PreparedSOM(X, max_rank=R.shape[1], threads=1)
a = prepared.fit(R, gamma=0.5, lam=1.0)
b = prepared.fit(R, gamma=0.8, lam=1.2)
```

X の変更不能コピーと完全 SVD の因子を保持し、各 fit は新しい W0/P0 から開始します。
`.fit()` は `threadpool` 経路で、SVD 準備時間は含みません。単発呼出が速くなるという意味ではありません。
X・dtype・対応するランクを変える場合は作り直します。`.describe()` で保持メモリと準備時間を確認できます。

`prepare(X)` / `PreparedData` は別の API で、変更不能な入力と明示的な基本統計キャッシュを持ちます。
SVD を共有したい場合は `PreparedSOM` を選んでください。

## Trustworthiness / Continuity

[最小例](../examples/neighborhood_quality.py) / [実装](../restart_v2/portable_accel/metrics.py)

```python
joint_quality(X, Y, ks=5, *, backend="auto", policy=None,
              max_distance_bytes=None, return_stats=False, rank_method=None)
```

- X: `(N, D)`、Y: 同じサンプル・同じ行順の `(N, Q)`。`N >= 3`
- `ks`: 整数または整数の列。各 k は `1 <= k < N/2`。重複 k は要求順を保って1つにします
- `max_distance_bytes`: 最大の距離行列1枚が超えていたら、作成前にエラーにする任意の上限
- `return_stats=True`: `(qualities, stats)` を返す。`stats` は実際の経路や距離メモリ見積もりを含む

| backend | 既定 rank_method | 指定可能な rank_method |
|---|---|---|
| `numpy` | `broadcast` | `broadcast`, `searchsorted`, `sortsearch`, `full` |
| `sqrt_numpy` | `sortsearch` | `sortsearch`, `broadcast` |
| `numba` | `scan` | `scan`, `histogram`, `full` |
| `sqrt_numba`（別名 `numba_sqrt`） | `scan` | `scan`, `histogram` |

`auto` は Numba が利用可能なら `numba`、それ以外は `numpy`。
平方根省略系も、厳密順位を保つため必要に応じて境界を再計算します。
同距離ではインストール済み NumPy の実際の argsort 順序を保ち、独自の index 順を仮定しません。

戻り値は常に `Quality` レコードのリストです。各レコードは次の属性を持ちます。

- `.k`, `.trustworthiness`, `.continuity`
- `.trustworthiness_penalty`, `.continuity_penalty`: 正規化前の整数ペナルティ

ブロック化されるのは順位処理で、距離計算は全 `(N,N)` 行列です。
float64 の距離行列1枚だけでも `8*N*N` bytes。`N=70,000` なら約39.2 GBになり、ほかの領域は別です。
まず小規模に試し、抽出評価をするなら全点評価とは区別してください。
