# UbuKit

Fuzzy, rough and classical clustering, self-organizing maps, evaluation metrics
and TPE search for browsers and Node.js. No dependencies; types included.

```sh
npm install ubukit
```

It needs Node.js 22 or newer, or a current browser.

```js
import * as ub from 'ubukit';

const X = [[0, 0], [0.1, 0.2], [4, 4], [4.2, 3.9]];  // rows, or { data: Float64Array, rows, cols }

const r = ub.fcm(X, 2, { m: 2, seed: 0 });           // also kmeans, efcm, rcm, rmcm
r.centers; r.membership; r.labels; r.nIter; r.stopReason; r.history;
ub.toRows(r.centers);                                 // matrices are { data, rows, cols }

const m = ub.somOlp(X, [3, 3], { lam: 0.5, gamma: 1 }); // also som, batchSom
ub.trustworthiness(X, m.embedding, 1);
ub.ari([0, 0, 1, 1], Array.from(r.labels));

const best = await ub.minimize(p => (p.x - 1) ** 2, { x: ub.uniform(-5, 5) }, { nTrials: 40 });
```

Every fit returns a Result: `centers` (K, D), `labels` (N), `membership`
(N, K; `null` for `kmeans`, `som` and `batchSom`), for the maps `embedding`
(N, Q) and `unitCoordinates` (K, Q), `nIter`, `stopReason` (`'tol'`,
`'schedule'` or `'limit'`) and `history`, the objective at each assignment
(empty for `rcm`, `rmcm`, `som` and `batchSom`, which have none). A map's
unit `j` sits at row `j` of `unitCoordinates`, `(row, col)` on a grid with
`j = row * cols + col`, and has the prototype in row `j` of `centers`.

A Result is the state of its last iteration: `labels` and `membership` come
from the last assignment and `centers` from the update after it, so
assigning the data to `centers` again, as `ub.apply` does, can change them,
most after an early stop (`'limit'`). `som` and `batchSom` instead give the
best-matching units of their final prototypes.

The stop reason says how a run stopped, not that its result is good: from
about five dimensions on, FCM with the default `m: 2` can put every
prototype at the mean of the data and still stop with `'tol'`. Use `m`
closer to 1 there (1.2, for example) and check that the centers are apart.
Inputs must be finite numbers of ordinary scale; standardize features first.
The equations, the stopping rules and when FCM and EFCM collapse are in
[docs/algorithms.md](https://github.com/subukata/ubukit/blob/v0.3.0/docs/algorithms.md).

Trained prototypes apply to new data, fixed: `ub.apply` has a function for
every method, which takes the data, the prototypes and the parameters they
were trained with, and returns the Result of the new data (`ub.steps.apply`
has the same functions as generators). The parameters have no defaults,
since only the trained ones are right: keep them in one object for both
calls.

```js
const added = [[0.5, 0.4], [3.9, 4.1]];
const placed = ub.apply.somOlp(added, m.centers, m.unitCoordinates, { lam: 0.5, gamma: 1 });
placed.embedding;                                      // the new points on the trained map
ub.apply.fcm(added, r.centers, { m: 2 }).membership;   // the m that r was trained with
```

Each method assigns once, except `somOlp`, whose memberships iterate with
the map fixed until they settle, so for the points the map was trained on
they need not end at the trained `embedding`; `rmcm`'s memberships depend on
the points applied together
([applying trained prototypes](https://github.com/subukata/ubukit/blob/v0.3.0/docs/algorithms.md#applying-trained-prototypes)).

A search that scores many embeddings of one `X` computes its distances once
with `new Neighborhood(X)` (8 N^2 bytes), which both metrics take in place
of `X`; an array of `k` gives a `Float64Array` with one value per `k`:

```js
const reference = new ub.Neighborhood(X);
ub.trustworthiness(reference, m.embedding, 1);
ub.continuity(reference, m.embedding, [1]);
```

Fitting runs synchronously. Every fitting function has a generator in `steps`
that yields `{ iteration, result }` after each iteration (each epoch for
maps); `result()` returns the Result the run would return had it stopped
there, and costs nothing unless called. Keep the generator to pause, step or
continue, and draw the learning as it happens:

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
iteration on new points and keeps what the run has learned (prototypes,
schedules, and SOM-OLP's memberships while point i stays the same one), so
the map follows data that move. `maxIter: Infinity` lets it run for as long
as data keep coming:

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

Worked uses, in Python with the same names (camelCase here): <https://github.com/subukata/ubukit/blob/v0.3.0/docs/recipes.md>.
The Python package `ubukit` on PyPI has the same API in snake_case.
