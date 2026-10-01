# UbuKit RMCM 候補実装

ユーザーが指定した Rough Membership C-means（RMCM）の独立した追加候補です。既存の RCM / ExRCM、FCM、RMCM2 とは別のアルゴリズムです。

## インストール

ZIP を展開して、この `pyproject.toml` があるフォルダーで実行します。

```sh
python -m pip install .
```

通常実行に必要なのは NumPy、SciPy、threadpoolctl です。UbuKit 独自の C/C++ 拡張やビルド作業は不要です。依存ライブラリに対応 wheel がない環境では依存側のビルドが必要になる可能性があります。

任意の Numba 集計バックエンドとテストを使う場合:

```sh
python -m pip install '.[numba,test]'
```

Python 3.10 以上を対象としています。実測・配布検証は Linux x86-64 / Python 3.12 で行い、macOS / Windows は未検証です。

## 最短の使用例

```python
import numpy as np
from ubukit_rmcm import fit_rmcm

X = np.array([[0.0], [1.0], [3.0], [8.0], [9.0]])
result = fit_rmcm(X, n_clusters=2, delta=2.0, random_state=42)
print(result.centers)       # (K, D): 最後の重み付き重心
print(result.memberships)   # (N, K): その重心を作ったラフメンバシップ
print(result.labels)        # (N,): そのメンバシップを作った暫定最近傍ラベル
print(result.stop_reason)   # fixed_point / cycle / max_iter
```

`labels` は返却重心で追加再分類した値ではありません。最終ラベルが必要なら、返却重心との距離で別途分類してください。ただし、それは返却された `memberships` を生成したラベルとは異なることがあります。

## 指定された計算そのもの

1. 初めに一度だけ δ 近傍関係を構築する。距離はユークリッド距離、判定は `distance <= delta`。自分自身を含む
2. データから異なる K 個の行番号をランダムサンプリングして初期中心にする。値の重複は許す
3. 各データを最近傍中心へ暫定割り当てする。同距離なら最小のクラスタ番号
4. 点 i の近傍内でクラスタ c に暫定割り当てされた点の占有率を `R[i,c]` にする
5. `R[i,c]` をそのまま重みにして中心を更新する。べき乗・ファジファイアは入れない
6. 3–5 を繰り返す。δ 関係は更新しない

数式、計算量、有限精度上の注意は [THEORY.md](THEORY.md)、実測は [REPORT.md](REPORT.md) を参照してください。

RMCM は最適化問題に基づく方法ではなく、収束保証を置いていません。ユーザーが区別した目的関数に基づく RMCM2 は別の方法であり、この実装には含めていません。サイクル検出は RMCM の停止状態を正しく報告するためのものです。

## バックエンド

| backend | 計算内容 | 用途 |
|---|---|---|
| `numpy` | 近傍ラベルを `bincount` で数え、密 R を作り、`R.T @ X` | 読める NumPy スクラッチ基準 |
| `csr` | 疎行列 `R=P@H` を作って重心計算 | 強い直接実装・有限精度の比較対象 |
| `adjoint`（標準） | 固定 `P.T@X` と `P.T@1` を前計算し、ラベルごとに集計 | 反復中に R を作らない |
| `numba` | `adjoint` と同じ式の集計だけをシリアル JIT | 任意依存、初回コンパイルあり |

全バックエンドで同じ δ グラフ、直接距離の NumPy カーネル、初期化、空クラスタ処理、停止条件を使います。`numba` は `fastmath` や並列集計を使いません。既定値を自動性能判定で切り替える仕組みはありません。

全点近傍のときは全バックエンド共通で厳密な簡約を使います。占有されている中心を共通の全体平均にし、同じ数値を代入します。空クラスタの中心は保持します。これにより、数学的には同じ中心なのに和の順番だけで同距離判定が割れる退化ケースを避けます。

## 同じ近傍関係を再利用する

```python
from ubukit_rmcm import prepare_rmcm

prepared = prepare_rmcm(X, delta=2.0, backend='adjoint')
a = prepared.fit(2, random_state=0)
b = prepared.fit(2, random_state=1)
```

`prepare_rmcm` は X の独立スナップショットを持ちます。同じ X と δ に対して初期中心や K を変える場合、グラフ構築と前計算を再利用できます。`fit_rmcm` は毎回これらを含めます。両者の時間を混同しないでください。

## 停止条件と安全策

- 連続する `(labels, centers)` が完全に一致した場合だけ `fixed_point`、`converged=True`
- 過去の完全状態が再登場した場合は `cycle`、`converged=False` と周期を返す
- `cycle_window=32` が記憶する状態数。有限窓より長い周期を見逃す可能性がある。0 で周期検出を無効化できる
- `max_iter` 到達は `max_iter`。収束したと表示しない。`max_iter=1` では 1 回更新した結果を返す
- 空クラスタは直前の中心を保持する。自己近傍があるので、クラスタの粗い重みが 0 なのは暫定割り当てが 0 点のときだけ
- `delta=0` は同じ値の行すべてを近傍に含む。自己除外モードは提供しない
- 不正な形状・非有限値・複素数、距離の浮動小数点オーバーフロー／アンダーフローが分かった場合はエラーにする。無言で同距離扱いにしない
- `max_edges=10_000_000` は自己辺を含む有向候補辺数の上限。高密度グラフを大規模確保する前に停止する。実メモリのバイト上限ではない
- `max_edges` は R の大きさを制限しない。`numpy` は反復中に密な N×K 行列を持つ。全バックエンドで `return_memberships=True` は最終的に密な N×K 行列を返す
- 大きい K でメンバシップが不要なら `backend='adjoint', return_memberships=False` を使う。これもグラフ・前計算・距離作業領域・状態履歴のメモリは必要
- `threads=1` は BLAS を制限する。threadpoolctl の設定はプロセス共通なので、異なる制限を同時に使うスレッド間では呼び出し側で調整する

浮動小数点では結合法則が成り立たないため、`adjoint` は実数上では同じ式でも、`numpy` / `csr` とビット一致する保証はありません。最近傍境界の極端に近くでは、微小な中心差がラベル・その後の経路を変えることがあります。普遍的な軌道一致や収束保証を主張しません。

## テストと再測定

```sh
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.benchmark --numba --repeats 7 --output results/benchmark_final.json
python -m benchmarks.startup --numba --output results/startup.json
```

Numba を導入しない場合は `--numba` を省いてください。テストは利用可能なバックエンドを自動選択します。

再測定すると `results/benchmark_final.json` に生の反復時間、初期化・入力ハッシュ、コードハッシュ、反復回数、出力の数値比較を保存します。配布版には生ログを含めず、検証アーカイブにまとめています。
