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
(N, K; `null` for `kmeans`, `som` and `batchSom`), `embedding` ((N, Q) map
positions, for the maps), `unitCoordinates` ((K, Q), for the maps; see below),
`nIter`, `stopReason` and `history`, the
objective at each iteration (empty for `rcm`, `rmcm`, `som` and `batchSom`,
which have none). `stopReason` is the stopping rule that holds at the
result's iteration: `'tol'` (no prototype moved more than the tolerance),
`'schedule'` (`som` and `batchSom`: every epoch has run) or `'limit'`
(`maxIter` was reached first), and `null` in a per-iteration result while
the run goes on; none of them says that the result is good.
In `rcm`, membership 1 puts a point in the lower approximation of a cluster;
a point shared equally among clusters (1/2, 1/3, ...) is in their boundary.
Inputs should be numbers, finite and of ordinary scale; standardize features first.

Trained prototypes apply to new data, fixed: `ub.apply` has a function for
every method, which takes the data, the prototypes (`centers`) and the
parameters they were trained with, and returns the Result of the new data,
`centers` unchanged. Retraining is the fitting function's job; these never
move the prototypes.

```js
const added = [[0.5, 0.4], [3.9, 4.1]];
const placed = ub.apply.somOlp(added, m.centers, m.unitCoordinates, { lam: 0.5, gamma: 1 });
placed.embedding;                                      // the new points on the trained map
ub.apply.fcm(added, r.centers, { m: 2 }).membership;
```

Each method assigns once (`stopReason === 'schedule'`), except `somOlp`,
whose memberships are iterated with the map fixed until they settle
(`'tol'`, or `'limit'` after `maxIter`, 1000 by default). They start from
the distances alone, so for the points a map was trained on they need not
end where training left them: keep the trained `embedding` for those, or
apply to them too for positions consistent with the final map. `rmcm` builds
its neighborhoods within the data given, so apply it to new points together.
`ub.steps.apply` has the same functions as generators.

A search that scores many embeddings of the same `X` computes the distances
of `X` once with a `Neighborhood` (8 N^2 bytes), which both metrics take in
place of `X` with the same values; an array of `k` gives a `Float64Array`
with one value per `k`:

```js
const reference = new ub.Neighborhood(X);
ub.trustworthiness(reference, m.embedding, 1);
ub.continuity(reference, m.embedding, [1]);
```

A map's `unitCoordinates` are the coordinates of its units as the run used
them, not normalized: `(row, col)` in grid steps on a rectangular grid (unit
`j = row * cols + col`, so a plot can take x = col and y = row), or the
`(K, Q)` matrix given to `som` or `somOlp`. Row `j` of `centers` is the
prototype of unit `j`, and `embedding` holds the coordinates of each point's
unit (`labels`) for `som` and `batchSom`, and `membership` times
`unitCoordinates` for `somOlp`.

From about five dimensions on, FCM with the default `m: 2` can collapse:
every prototype goes to the mean of the data, every membership to 1/K, and
the run still stops with `'tol'`. Use `m` closer to 1 there (1.2, for
example) and check that the centers are apart; `efcm` collapses likewise
when `tau` is too large ([degenerate solutions](https://github.com/subukata/ubukit/blob/main/docs/algorithms.md#degenerate-solutions-fcm-efcm)).

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
Worked uses, in Python with the same names (camelCase here): <https://github.com/subukata/ubukit/blob/main/docs/recipes.md>.
The Python package `ubukit` on PyPI has the same API in snake_case.
