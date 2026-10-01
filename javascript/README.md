# UbuKit JavaScript 版（0.1.0 ローカルプレビュー）

デモサイトへ直接組み込める、外部依存なしの ES Modules / Float64Array CPU 実装です。

- k-means / FCM / RCM / ExRCM / SOM-OLP / 近傍評価（Trustworthiness・Continuity）
- 共通 `run` / `runAsync` / Web Worker API
- 行優先 TypedArray、再利用バッファ、不要な N×K×D テンソルを作らない計算
- 進捗通知、AbortSignal、Worker の即時終了によるキャンセル、クライアント再利用
- Python の明示初期値による照合 fixture と Node テスト、ブラウザ用デモ・検証画面

現時点では npm 公開・デモサイト配信はしていません。WASM/GPU 版ではありません。Node の検証とブラウザの検証は別に扱います。ブラウザの実測状況は `docs/VALIDATION.md` を参照してください。

Node のデモサイズ計測では、同条件の JS 参照実装に対して k-means 12.85×、RCM 7.91×、ExRCM 6.21×、近傍評価 3.94×。FCM は両者 O(NK) のメンバーシップ更新・TypedArray・m=2 最適化で再測定し、本体 8.89 ms／参照 6.10 ms（0.69×）でした。FCM と SOM-OLP（0.85×）は今回の暖機後計測で高速化を確認していません。FCM だけ再測定した時刻と、保持した他方式の測定時刻を分けて記録しています。比較条件は [性能要約](../docs/PERFORMANCE_JA.md) を参照してください。過去の raw data は利用に不要なため同梱せず、同梱スクリプトで再計測できます。

## すぐ試す

Node.js 20 以上で、依存インストールは不要です。

```sh
cd ubukit_js
npm test
npm run demo
```

自分のブラウザで http://localhost:8765/ を開きます。HTTP で配信してください。`file://` は ES Module / Worker 制約のため対象外です。検証画面は http://localhost:8765/demo/test.html 。

## ブラウザで使う（推奨）

```js
import { createWorkerClient } from './src/index.js';

const client = createWorkerClient();
const controller = new AbortController();
const input = {
  data: Float64Array.of(0, 0, 1, 1, 9, 9, 10, 10),
  nSamples: 4,
  nFeatures: 2
};

const result = await client.run('fcm', input, {
  nClusters: 2,
  m: 2,
  seed: 7,
  maxIterations: 100,
  tolerance: 1e-5,
  signal: controller.signal,
  onProgress: event => console.log(event)
});
console.log(result.centers, result.membership, result.labels);
// controller.abort(); // 実行中に呼ぶと AbortError。次の run は利用可能
client.dispose();
```

`src/` をそのまま同一 origin で配信できます。バンドラを使う場合は `new URL('./worker.js', import.meta.url)` の Worker 対応を確認し、必要なら `workerFactory` または `workerUrl` を指定してください。CSP の `worker-src` と JavaScript MIME type も必要です。ブラウザから外部にデータを送る処理は含みません。

`transferInput: true` は入力 ArrayBuffer の所有権を Worker に渡し、呼出側の配列を detach します。入力を再利用する通常のデモでは既定の `false` を使ってください。転送後にキャンセルすると入力バッファは戻りません。初期値・grid・embedding など options 内の配列は既定の structured clone です。結果は転送で返します。1 クライアントに同時実行できる仕事は 1 件で、並列ジョブには別クライアントが必要です。

## 共通 API

```js
import { run, runAsync, steps } from './src/index.js';
const result = run('kmeans', input, { nClusters: 2, seed: 7 });
const asyncResult = await runAsync('fcm', input, {
  nClusters: 2, timeBudgetMs: 8, blockRows: 256, signal
});
```

- 入力は `{ data: numeric TypedArray, nSamples, nFeatures }`。Float32Array 等は Float64Array に変換します
- `run` は同期処理で、UI をブロックします。デモでは Worker を推奨します
- `runAsync` は行ブロックの間でイベントループへ処理を返す代替です。単一行、初期化、固有値計算等の処理時間までは制限しません
- `steps` は generator。yield 値は進捗/チェックポイント、generator の return 値が最終結果です
- 入力配列は変更しません。共通 membership の形は N×K、centers は K×D、labels は N です
- `seed` は Mulberry32 の JS 固有系列。NumPy と同じ数値 seed だけでは同じ初期値になりません。比較には明示初期値を渡してください
- クラスタリングの `maxMemoryBytes` は主要配列の推定上限、既定 512 MiB。JavaScript オブジェクト・JIT・GC・Worker のコピー等を含むプロセス総メモリ上限ではありません

### kmeans

`nClusters`、`initCenters`（K×D）または `seed`、`maxIterations`（既定100）、`blockRows`。

Lloyd 法。等距離は小さいクラスタ番号、空クラスタは前の中心を保持。ラベル完全一致で停止します。`tolerance` は0のみ許可。最終中心に対する labels/inertia を再計算し、最終更新前ラベルは `coreLabels` に残します。

### fcm

`nClusters`、`m > 1`（既定2）、`initMembership`（N×K）または `initCenters`（K×D）、`maxIterations`、`tolerance`（既定1e-5）、`returnHistory`。

標準 FCM。中心重み U^m、メンバーシップは二乗距離の逆数べきで正規化。距離ゼロが複数ある点は、それらの中心へ等分します。`||U_new-U_old||_F < tolerance` で停止し、tolerance=0 は必ず指定回数。中心は最終 U_old から計算したもの、返却 U は U_new で、停止後の隠れた中心更新はありません。objective は返却ペアで評価します。空クラスタは前の中心を保持（初回はデータ平均）。

距離の x² 項は省きません。大きい m や小さい距離比の underflow はスケーリング/log-space で保護します。二乗距離の overflow、非ゼロ差が全て二乗 underflow する極端なデータは明示エラーで、データの再スケールが必要です。Python 版の自動 rescale 範囲を全て実装したとは主張しません。

### rcm / exrcm

`nClusters`、`initCenters` または `seed`、`alpha >= 1`（既定1.1）、`beta >= 0`（既定0）、`p > 0`（exrcm既定1）、`maxIterations`、`cycleWindow`（既定16、0で周期検出を無効化）。

Euclidean d を使い、各点について `d_ci^p <= alpha^p*d_min_i^p + beta^p` を満たす二値 mask を求め、点ごとに合計1へ正規化します。中心はその U の重み付き平均です。RCM は厳密に p=1。beta は距離と同じ単位です。p=2 は二乗距離で判定し、一般 p はスケーリング/log-space を使います。式の境界に任意の epsilon を足しません。

空クラスタは前の中心を保持。固定点、周期、上限回数を区別し `converged` / `stopReason` を返します。最終 U は常に返却中心から再計算します。周期検出の履歴にもメモリが必要です。JavaScript の public U は N×K で、Python ExRCM の C×N とは転置関係にあります。

### som-olp

`grid: {data, nSamples: M, nFeatures: Q}`、`gamma`、`lambda`、`maxIterations`、`tolerance`。`initialPrototypes`（M×D）と `initialMemberships`（N×M）を同時に渡すと初期化差を取り除いた比較ができます。

既定は JS PCA 初期化、`initializer: 'sample'` も利用できます。PCA は NumPy SVD と bit 一致を保証しません。prototypes/centers、memberships/membership、embedding、history を返します。元実装の更新順序を保持し、embedding は最終 P を更新する直前の P から求めた値です。詳しくは `docs/SOM_AND_NEIGHBORHOOD.md`。

### neighborhood

`embedding: {data, nSamples: N, nFeatures: Q}` と `k` または `ks` を指定。結果は `qualities: [{ k, trustworthiness, continuity, ... }]`。

厳密順位、同距離の順序は点の index。巨大な N×N の距離/順位行列を保持せず O(N+Kmax) の主要 scratch で計算します。ただし計算時間は O(N²) です。全 N×N を保持する方式は、Float64 距離だけでも N=70,000 で約39.2 GBになるためデモでは避けてください。この実装でも70,000点の厳密評価が軽くなるわけではありません。サンプリングするなら厳密全点評価と区別してください。

## Node.js Worker

```js
import { Worker } from 'node:worker_threads';
import { createWorkerClient } from './src/index.js';
const client = createWorkerClient({
  workerFactory: () => new Worker(new URL('./src/worker.js', import.meta.url), { type: 'module' })
});
try { console.log(await client.run('kmeans', input, { nClusters: 2 })); }
finally { client.dispose(); }
```

## 検証と計測

- `npm test` は Python fixture 照合・境界条件・キャンセル・Worker 再利用等
- `npm run bench` は Node の再計測。結果ファイルは実行時に生成します
- [性能要約](../docs/PERFORMANCE_JA.md) に比較条件と確認できた範囲をまとめています
- ブラウザ検証ページは6 APIの同期/Worker一致、進捗、UI heartbeat、実計算開始後のキャンセル、再利用、buffer detach を検査
- 同じデータ・初期値・停止条件・返却結果の契約で比較。冷スタートと暖機後、計算本体と Worker通信込みを混同しません
- Node の結果はブラウザ性能の代用ではありません。全データ・全ブラウザでの最速を保証しません

## ファイル

`src/` 本体、`demo/` 最小デモ/ブラウザ検証、`tests/` 自動テスト、`fixtures/` Python参照値、`bench/` 再現可能な計測、`docs/` 数値契約・検証・計測記録。

SOM-OLP 由来部分の MIT 表記は `LICENSES/SOM-OLP-MIT.txt` に保持しています。
