# UbuKit

Fuzzy, rough and classical clustering, self-organizing maps, evaluation metrics
and TPE search for browsers and Node.js. No dependencies; types included.

```sh
npm install ubukit
```

```js
import * as ub from 'ubukit';

const X = [[0, 0], [0.1, 0.2], [4, 4], [4.2, 3.9]];  // rows, or { data: Float64Array, rows, cols }

const r = ub.fcm(X, 2, { m: 2, seed: 0 });           // also kmeans, efcm, rcm, rmcm
r.centers; r.membership; r.labels; r.nIter; r.converged; r.history;
ub.toRows(r.centers);                                 // matrices are { data, rows, cols }

const m = ub.somOlp(X, [3, 3], { lam: 0.5, gamma: 1 }); // also som, batchSom
ub.trustworthiness(X, m.embedding, 1);
ub.ari([0, 0, 1, 1], Array.from(r.labels));

const best = await ub.minimize(p => (p.x - 1) ** 2, { x: ub.uniform(-5, 5) }, { nTrials: 40 });
```

Fitting runs synchronously. Every fitting function has a generator in `steps`
that yields after each iteration; `runAsync` drives it without blocking the
event loop and supports `AbortSignal`:

```js
const result = await ub.runAsync(ub.steps.fcm(X, 2), { signal, onProgress: ({ iteration }) => {} });
```

For heavy work in a browser, import `ubukit` inside your own Web Worker.

Equations and references: <https://github.com/subukata/ubukit/blob/main/docs/algorithms.md>.
The Python package `ubukit` on PyPI has the same API in snake_case.
