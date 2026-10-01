# 性能の要約と再計測

[使い方に戻る](../README.md)。2026-10-01 の測定から、利用時の選択に関係する結果だけをまとめています。
測定値は保証値ではありません。データ形状・初期値・反復数・スレッド数・JIT の有無を揃えて比較してください。

## まず選ぶ設定

- Python FCM: 単発・小規模は既定の `scipy`。高次元は `blas`、暖機後の低次元は任意の `numba_parallel` を比較する価値があります
- Python RCM / ExRCM: Numba なしなら `scipy` または `numpy`。長く使う場合は初回 JIT と分けて `numba` を比較してください
- JavaScript: UI では Worker を使います。下の速度は Node の同期実行で、ブラウザや Worker 通信込みの速度ではありません
- どの手法も `auto` は入力から最速の実装を予測する機能ではありません

## Python FCM

Linux x86-64 / AMD EPYC 9V74 / Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、Numba 0.68.0。
float64、`m=2`、1 worker、同じ明示初期所属度。暖機後3回を交互に測った中央値です。
収束条件は所属度の Frobenius 変化 `< 1e-5`。入力検証・初期値処理・最終目的関数と labels を含む fit 全体です。

| (N, D, K) | 経路 | fit 秒 | NumPy 基準に対する倍率 | scikit-fuzzy 基準に対する倍率 |
|---|---|---:|---:|---:|
| (100000, 8, 8) | `numba_parallel`, 1 worker | 0.784 | 3.82× | 2.06× |
| (10000, 128, 8) | `blas` | 0.111 | 8.14× | 1.58× |
| (10000, 16, 64) | `scipy` | 0.126 | 4.76× | 1.56× |

- NumPy 基準も O(NK) の逆べき正規化を使っています。所属度の正規化を意図的に O(NK²) にした基準ではありません
- 外部基準は **scikit-fuzzy 0.5.0 + 共通出力アダプタ**。返却ペアの目的関数・labels・配列方向を揃えた処理を含み、bare `cmeans` だけの比較ではありません
- 主測定6条件で反復数が一致。最大所属度差は約1.46e-11、最終目的関数の最大相対差は約3.6e-15
- 強制30回更新では、ほぼ等しい所属度の微差から hard label が変わる条件がありました。全条件のラベル bit 一致は保証しません
- ランダム初期化の高次元条件等では所属度がほぼ一様になる場合もあります。標準 FCM の同等な計算を速くする結果であり、クラスタ品質の改善を意味しません
- Numba は初回 JIT 時間・コンパイラのメモリを要します。上の表に冷スタート時間は含みません

対数 / logsumexp の安定化は極端条件で重要ですが、常に log/exp を計算する経路が最速になるわけではありません。
[数式と保護処理](../fcm_challenge/THEORY.md)、[外部実装との契約差](../fcm_challenge/COMPARATORS.md) を参照してください。

## Python RCM / ExRCM

同じ Linux x86-64 環境、float64、BLAS 1スレッド、同じ明示初期中心、同じ収束・最終所属度出力。
暖機後5回を順序を変えてペア測定した中央値。下は12条件での倍率の範囲です。

| 経路 | 単純 broadcast 参照実装に対する倍率 |
|---|---:|
| `numpy` | 1.24〜3.39× |
| `scipy` | 1.41〜7.39× |
| `numba` | 3.12〜6.35× |

条件は `(N,K,D)` = `(1024,4,8)`, `(12000,8,16)`, `(4096,16,64)`, `(2048,16,256)`、各 `p=1,2,3`。
外部ライブラリとの比較ではありません。全12条件×4実装で最終 mask / 所属度が一致し、全 fit が収束しました。
中心の最大絶対差は約1.33e-14です。浮動小数点の厳密な境界で、あらゆる評価順序の一致を保証するものではありません。

初回の Numba JIT コストは別です。通常の p=1 / p=2 の専用経路と、極端な p で使う対数領域の保護を併用します。
オーバーフローして誤った mask を返す単純な式は、同等精度の高速化比較として扱いません。

## JavaScript

Node.js 24.19.0 / V8 13.6.233.17 / Linux x64 / AMD EPYC 9V74。
単一の Node メインスレッド上で同期実行。比較先は同梱 `javascript/bench/` の単純 JavaScript 実装です。
外部の最速ライブラリ・Python・WASM・GPU との比較ではありません。

### FCM: O(NK) 同士で再計測

FCM は **2026-10-01 04:46:51 UTC 開始**の再計測です。
両実装とも TypedArray の作業バッファを再利用し、O(NK) の逆べき正規化と `m=2` の乗除算による専用処理を使います。
参照は重み・距離・所属度を別々のループで計算する標準的な実装です。

`(N,D,K)=(2000,8,8)`、同じ入力・明示初期所属度、固定12回更新。
3回の追加暖機後、7回の交互計測の中央値。最終 labels / objective / FPC / delta を含む fit 全体です。

| FCM 実装 | 中央値 ms |
|---|---:|
| O(NK) 参照実装 | 6.10 |
| UbuKit JS | 8.89 |
| 参照 / UbuKit | 0.69× |

**この公平な比較では FCM の高速化は確認できませんでした。**
測定データで中心・所属度の最大絶対差は0。目的関数・FPC・delta・labels・収束状態・反復数も照合しました。
両実装の完全な1反復は O(NKD)、所属度正規化は O(NK) です。
これは同条件の JS 同士の比較で、Python 対 JavaScript の速度比較ではありません。

### その他の手法: 既存の測定

以下は **04:12 UTC の元の測定**を保持しています。FCM の修正時に再計測したものではありません。
クラスタリングは `(N,D,K)=(2000,8,8)`、反復上限12、同じ明示初期値、暖機後5回の交互計測。
k-means / ExRCM は両実装12回、RCM は両実装4回です。

| 手法 | 参照中央値 ms | UbuKit JS ms | 参照 / UbuKit |
|---|---:|---:|---:|
| k-means | 31.97 | 2.49 | 12.85× |
| RCM | 20.83 | 2.63 | 7.91× |
| ExRCM | 37.70 | 6.07 | 6.21× |

SOM は `(N,D)=(400,8)`、6×6格子、固定8回更新。
T/C は `N=700`、元8次元 / 埋め込み2次元、k=[1,5,10]。
それぞれ3回暖機後、7回の交互計測です。

| 手法・計時範囲 | 参照中央値 ms | UbuKit JS ms | 参照 / UbuKit |
|---|---:|---:|---:|
| SOM、同じ初期値から更新部分 | 6.84 | 8.02 | 0.85× |
| SOM、共通の JS PCA 初期化込み | 7.34 | 7.89 | 0.93× |
| T/C、同距離なし | 171.52 | 43.58 | 3.94× |
| T/C、同距離あり | 152.47 | 43.67 | 3.49× |

**この SOM 条件では高速化していません。** メモリ・Worker・進捗対応を備えた移植として扱ってください。
SOM の共通 PCA 比較は元の NumPy SVD との速度比較ではありません。
ブラウザでの実計測は未完了です。Node Worker のテスト成功をブラウザ速度やブラウザ動作確認と読み替えないでください。

## 再計測する

利用時には不要ですが、比較用の短いスクリプトは同梱しています。結果ファイルは実行時に新しく生成します。
過去の全 raw log・途中段階のソース・生成済み wheel は追加パッケージに含めていません。

FCM（リポジトリルートから）:

```sh
python -m pip install './fcm_challenge[numba,test,benchmark]'
cd fcm_challenge
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.benchmark --repeats 3 --output results/benchmark_t1.json
```

RCM / ExRCM（別シェルでリポジトリルートから）:

```sh
python -m pip install './rcm_challenge[scipy,numba]'
cd rcm_challenge
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmark.py --repeats 5 --output results/benchmark.json
```

JavaScript（別シェルでリポジトリルートから）:

```sh
cd javascript
npm test
npm run bench
node bench/first-calls.js
# FCM だけを再計測する場合
node bench/remeasure-fcm-linear.js
```

これらの再計測コマンドは、小さな動作例より時間とメモリを使います。使い方の確認だけなら [examples/](../examples/) を実行してください。
測定中に他の重い処理を並走させず、初回 / 暖機後・入力生成の有無・Worker 通信の有無を分けて記録します。

## 既存の k-means / SOM-OLP / T/C

`portable_accel 0.2.0a2` の既存資料は [結果索引](../restart_v2/RESULT_INDEX.md) に保持しています。
旧候補や a1 の測定を a2 の新規測定として扱わないでください。
追加パッケージの利用に、既存アーカイブ内の実験を全て再実行する必要はありません。
