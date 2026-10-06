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

Inputs should be finite and of ordinary scale; standardize features first.

Fitting runs synchronously. Every fitting function has a generator in `steps`
that yields `{ iteration, result }` after each iteration (each epoch for
maps). `result()` returns the Result the run would return had it stopped
there (centers, memberships, labels, embedding, history), so the learning
can be drawn as it happens, without interpolation; it costs nothing unless
called. Keep the generator to pause, step or continue:

```js
const run = ub.steps.somOlp(X, [10, 10], { lam: 0.5, gamma: 1, maxIter: 1000 });
function frame() {
  const { value, done } = run.next();     // one iteration
  draw(done ? value : value.result());    // done: value is the final Result
  if (!done) requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

The data are an input of every iteration: `run.next(points)` runs the next
iteration on new points and keeps everything else the run has learned
(prototypes, schedules, and SOM-OLP's memberships while the number of points
stays the same, point i being the same one as before), so the map follows
data that move. `maxIter: Infinity` lets it run for as long as data keep
coming:

```js
const run = ub.steps.somOlp(currentPoints(), [10, 10], { lam: 0.5, gamma: 1, maxIter: Infinity });
run.next();                                         // the first iteration
function frame() {
  draw(run.next(currentPoints()).value.result());   // the next one, on the points as they are now
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

`runAsync` drives a generator without blocking the event loop and supports
`AbortSignal`; `onProgress` receives the same `{ iteration, result }`:

```js
const result = await ub.runAsync(ub.steps.fcm(X, 2), { signal, onProgress: p => draw(p.result()) });
```

For heavy work in a browser, import `ubukit` inside your own Web Worker.

Equations and references: <https://github.com/subukata/ubukit/blob/main/docs/algorithms.md>.
The Python package `ubukit` on PyPI has the same API in snake_case.
