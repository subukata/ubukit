# JavaScript: 手法ごとの呼び出し方

[README に戻る](../README.md) / [全6手法を実行するファイル](../examples/javascript.mjs) / [詳しい契約](../javascript/README.md)

## 最初に

Node.js 20 以上で、リポジトリのルートから:

```sh
node examples/javascript.mjs
```

npm パッケージのインストールは不要です。ブラウザでは HTTP で ES Modules を配信します。
ここでは最小例として同期 `run` を使いますが、UI から呼ぶ場合は後述の Worker を使います。

以下の6つの例は、最初にこの共通入力を用意してから実行してください。
これはリポジトリのルートを配信した HTML 内の `<script type="module">`、または同じ位置の `.mjs` 用です。

```javascript
import { run } from './javascript/src/index.js';
const input = {
  data: Float64Array.of(0, 0, 0, 1, 1, 0, 8, 8, 8, 9, 9, 8),
  nSamples: 6, nFeatures: 2,
};
const initCenters = Float64Array.of(0, 0, 8, 8);  // K=2, D=2
```

配列は行優先で平坦化します。`data[i * nFeatures + j]` が点 i の特徴 j です。
所属度はクラスタリング3種で `(N, K)`、SOM では `(N, M)`。中心は `(K, D)`、SOM のノードは `(M, D)` です。
返却配列も平坦な TypedArray で、2次元の入れ子配列ではありません。

## k-means

```javascript
const km = run('kmeans', input, {
  nClusters: 2, initCenters, maxIterations: 100,
});
console.log(km.centers, km.labels, km.inertia, km.iterations);
```

`initCenters` を省略するなら `seed` を指定できます。`tolerance` は0のみで、ラベル一致で終了します。
返却 labels / inertia は最終中心への割当。更新前の割当は `coreLabels` です。

## FCM

```javascript
const fcm = run('fcm', input, {
  nClusters: 2, initCenters, m: 2,
  maxIterations: 100, tolerance: 1e-5,
});
console.log(fcm.membership, fcm.centers, fcm.objective, fcm.converged);
```

`m > 1`。明示初期値には `initMembership`（N×K）も使えます。
所属度変化の Frobenius norm が `tolerance` より小さくなれば終了。
`tolerance=0` は必ず指定回数。`returnHistory: true` なら `objectiveHistory` も返します。

## RCM

```javascript
const rcm = run('rcm', input, {
  nClusters: 2, initCenters, alpha: 1.1, beta: 0.5,
  maxIterations: 100,
});
console.log(rcm.mask, rcm.membership, rcm.stopReason, rcm.converged);
```

RCM は `p=1`。同距離の複数クラスタも含め、閾値を満たすクラスタへ所属度を等分します。
`mask` / `membership` は平坦な `(N, K)`。Python の rough_cmeans が返す `(K, N)` とは向きが逆です。

## ExRCM

```javascript
const exrcm = run('exrcm', input, {
  nClusters: 2, initCenters, p: 2, alpha: 1.1, beta: 0.5,
  maxIterations: 100, cycleWindow: 16,
});
console.log(exrcm.mask, exrcm.membership, exrcm.stopReason, exrcm.converged);
```

`p > 0`, `alpha >= 1`, `beta >= 0`。
判定は `d**p <= alpha**p * d_min**p + beta**p`。beta は常に距離と同じ単位です。
RCM / ExRCM は固定点・周期・上限回数を区別するので、`converged` と `stopReason` を確認します。
`cycleWindow=0` は周期検出を無効にします。

## SOM-OLP

```javascript
const som = run('som-olp', input, {
  grid: {
    data: Float64Array.of(0, 0, 0, 1, 1, 0, 1, 1),
    nSamples: 4, nFeatures: 2,
  },
  gamma: 0.5, lambda: 1,
  maxIterations: 50, tolerance: 1e-4,
});
console.log(som.embedding, som.prototypes, som.memberships, som.history);
```

`grid` は `(M,Q)` の格子、`gamma` は格子項の重み、`lambda > 0` はエントロピー項の重み。
Python の `lam` と JavaScript の `lambda` は名前が異なります。

返却 `embedding` は `(N,Q)`、`prototypes` は `(M,D)`、`memberships` は `(N,M)`。
既定の PCA 初期化は NumPy SVD と bit 一致を保証しません。
初期値を揃える場合は `initialPrototypes` と `initialMemberships` の両方を渡します。
元の更新順序を保持しているため、embedding は最終所属度を更新する直前の所属度から求めています。

## Trustworthiness / Continuity

```javascript
const quality = run('neighborhood', input, {
  embedding: { data: som.embedding, nSamples: 6, nFeatures: 2 },
  ks: [1, 2],
});
for (const q of quality.qualities) {
  console.log(q.k, q.trustworthiness, q.continuity);
}
```

元データと embedding は同じ行順。`k` または `ks` を指定し、`1 <= k < N/2` が必要です。
同距離の順位は点の index 順。Python API の NumPy argsort 規則とは異なる場合があります。
全 `(N,N)` 距離行列を保存しませんが、全点の厳密計算には O(N²) 時間が必要です。

## ブラウザでは Worker を使う

```javascript
import { createWorkerClient } from './javascript/src/index.js';
const client = createWorkerClient();
const controller = new AbortController();
try {
  const result = await client.run('fcm', input, {
    nClusters: 2, m: 2, seed: 4,
    signal: controller.signal,
    onProgress: event => console.log(event),
  });
  console.log(result.membership);
} finally {
  client.dispose();
}
// 実行中のキャンセルボタンから controller.abort() を呼びます。
```

1クライアントにつき同時に1件。キャンセルは `AbortError` で、終了後は再実行できます。
`transferInput` の既定値は false。true は入力 ArrayBuffer の所有権を Worker へ渡し、呼出側の配列が detach されるので注意してください。
単純なデモでは既定値のまま使います。CSP の `worker-src`、MIME、バンドラの Worker 対応も確認してください。

Node.js Worker、`runAsync`、進捗、メモリ上限の範囲は [パッケージ README](../javascript/README.md) を参照してください。
Node のテスト成功とブラウザ実機検証・ブラウザ性能は別です。[検証記録](../javascript/docs/VALIDATION.md) に実施範囲を残しています。
