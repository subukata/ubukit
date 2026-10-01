# 再構築版の使い方

これは再構築して検証した研究用コードです。以前に失われた配布物の復元品ではありません。Linux x86-64 / Python 3.12 で検証しています。Windows、macOS、ほかのCPUでは未検証です。

## インストール

ZIPを展開し、`restart_v2` に移動して実行します。

```sh
python -m pip install .
python -m pip install '.[numba]'  # Numba実装を使う場合
python -m unittest discover -s tests -v
```

`requirements-foundation.txt` はNumPy/SciPy等の標準依存だけ、`requirements-tested.txt` は検証時の任意依存も含む固定版です。Cython・コンパイラは通常のインストールには不要です。検証済みの純Python wheelと配布検証ログは `evidence/distribution/` にあります。

## 3つの処理

```python
import numpy as np
from portable_accel import ExecutionPolicy, fit_kmeans, fit_som_olp, joint_quality

rng = np.random.default_rng(4)
X = rng.normal(size=(300, 12))
R = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
policy = ExecutionPolicy(threads=4, block_rows=256, max_scratch_bytes=64*2**20)

km = fit_kmeans(X, X[:8], backend='numpy', policy=policy)
som = fit_som_olp(X, R, gamma=.5, lam=1., backend='threadpool',
                 initializer='svd_lowrank', policy=policy)
quality = joint_quality(X, som['V'], ks=[5, 10], backend='numpy', policy=policy)
```

- k-meansは初期中心を明示します。結果の `labels` は最終中心への割当、`inertia` はその距離和です。Numbaが使える場合は `backend='numba'` または `'numba_blas_vector'`、最終割当は `finalizer='numba'` を明示できます。空クラスタの扱い・終了条件・浮動小数点の同距離判定はsklearn全体との互換を保証しません。
- SOMの高速構成は `backend='threadpool', initializer='svd_lowrank'` を明示します。同じ完全SVDを使い、P0の計算を低ランク形へ変換します。元の初期化と参照カーネルも選択できます。
- 近傍指標はT/Cを同時に返します。Numba版は `backend='numba'`、平方根省略候補は `'sqrt_numba'` です。結果にはスコアと整数ペナルティが入り、複数kも指定できます。距離行列には依然O(N²)のメモリが必要です。

バックエンドの自動選択は利用可否による既定値であり、常に最速となる設定を推定する機能ではありません。計算の再現にはスレッド数、入力型、初期中心、初期化法、ブロックサイズも固定してください。

## 検証結果を読む場所

- `evidence/public_metrics/summary.json`: 実際の公開APIによる近傍指標比較
- `kmeans_candidate/results/matched_v2/`: 最終ラベル・慣性の計算込みk-means比較
- `som_candidate/FULL_FIT_REPORT.md`: 初期化込み70,000件SOM比較
- `evidence/distribution/validation.json`: 別環境へのwheelインストール、Numba不在動作、ソース一致確認
- `evidence/report_stable/`: 安定版時点の日本語レポート

計測値は指定データ・指定環境での値です。初回/JIT、暖機後、カーネルだけ、初期化込みを混同しないでください。Numbaを禁止した配布テストは12件中10成功・任意依存用2件スキップ、別途3 APIのスモーク確認も成功しました。

## 共通処理とメモリ

`prepare(X)` は元のXをコピーした変更不能スナップショットを所有します。元のXを変更しても保存済みノルム等は古くなりません。共通化そのものによる速度向上は主張していません。

`max_scratch_bytes` は明記した作業配列の予算であり、プロセス全体のメモリ上限ではありません。入出力、入力コピー、SVD、ライブラリの作業領域等は別です。スレッド制限はプロセス単位で作用するため、独立した同時実行には別プロセスを使ってください。

データセット、大きな実行時配列、仮想環境、JITキャッシュ、ネイティブのコンパイル済みバイナリはZIPに含めていません。取得元・生成方法・ハッシュは測定資料に記録しています。第三者のライセンスは `NOTICE.txt` と `LICENSE-SOM.txt` を確認してください。研究用プロジェクト全体の公開ライセンスは未選択です。

## 同じXで複数条件を試す場合

`PreparedSOM(X, max_rank=2, threads=9)` を明示すると、変更不能なXのコピーと完全SVDの因子を保持します。`prepared.fit(R, gamma=..., lam=..., policy=...)` は毎回新しいW0/P0から開始します。Xや型を変更するときは作り直してください。SVDのスレッド条件は準備時に固定され、fit側のスレッド数を変えてもSVDを再計算しません。

70,000件・3条件をまとめた実測では、毎回の新規コピーとSVDを含め40.008秒から33.089秒、約1.21倍でした。追加で約420MiBのスナップショット・因子を保持します。単発fitの高速化という意味ではなく、複数fitで準備を共用した結果です。旧安定版a1のwheel・測定済みソースと、再利用APIを追加したa2の検証は区別して収録しています。
