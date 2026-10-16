# UbuKit

Fuzzy, rough and classical clustering, self-organizing maps, evaluation metrics
and TPE search on NumPy and SciPy.

```sh
pip install ubukit
```

```python
import numpy as np
import ubukit as ub

X = np.random.default_rng(0).normal(size=(500, 2))

r = ub.fcm(X, 3, m=2.0, seed=0)  # also kmeans, efcm, rcm, rmcm
r.centers, r.membership, r.labels, r.n_iter, r.stop_reason, r.history

m = ub.som_olp(X, (8, 8), lam=0.5, gamma=1.0)  # also som, batch_som
m.embedding  # latent positions

y = ub.kmeans(X, 3, seed=1).labels
ub.ari(y, r.labels), ub.ami(y, r.labels)
ub.trustworthiness(X, m.embedding, k=5), ub.continuity(X, m.embedding, k=5)

best = ub.minimize(
    lambda p: -ub.ari(y, ub.fcm(X, 3, m=p["m"], seed=0).labels),
    {"m": ub.uniform(1.1, 3.0)},
    n_trials=20,
)
```

Every fit returns a `Result`: `centers` (K, D), `labels` (N,), `membership`
(N, K; `None` for `kmeans`, `som` and `batch_som`), `embedding` ((N, Q) map
positions, for the maps), `unit_coordinates` ((K, Q), for the maps; see below),
`n_iter`, `stop_reason` and `history`, the
objective at each iteration (empty for `rcm`, `rmcm`, `som` and `batch_som`,
which have none). `stop_reason` is the stopping rule that holds at the
result's iteration: `"tol"` (no prototype moved more than the tolerance),
`"schedule"` (`som` and `batch_som`: every epoch has run) or `"limit"`
(`max_iter` was reached first), and `None` in a per-iteration result while
the run goes on; none of them says that the result is good.
In `rcm`, membership 1 puts a point in the lower approximation of a cluster;
a point shared equally among clusters (1/2, 1/3, ...) is in their boundary.
A pickled `Result` or `TPE` loads in the UbuKit version that wrote it; keep
results across versions as arrays. Inputs should be finite and of ordinary
scale; standardize features first.

Trained prototypes apply to new data, fixed: `ub.apply` has a function for
every method, which takes the data, the prototypes (`centers`) and the
parameters they were trained with, and returns the `Result` of the new data,
`centers` unchanged. Retraining is the fitting function's job; these never
move the prototypes.

```python
X_new = np.random.default_rng(1).normal(size=(20, 2))
new = ub.apply.som_olp(X_new, m.centers, m.unit_coordinates, lam=0.5, gamma=1.0)
new.embedding  # the new points on the trained map
ub.apply.fcm(X_new, r.centers, m=2.0).membership
```

Each method assigns once (`stop_reason == "schedule"`), except `som_olp`,
whose memberships are iterated with the map fixed until they settle
(`"tol"`, or `"limit"` after `max_iter`, 1000 by default). They start from
the distances alone, so for the points a map was trained on they need not
end where training left them: keep the trained `embedding` for those, or
apply to them too for positions consistent with the final map. `rmcm` builds
its neighborhoods within the data given, so apply it to new points together.
`ub.steps.apply` has the same functions as generators.

A search that scores many embeddings of the same `X` computes the distances
of `X` once with a `Neighborhood` (8 N^2 bytes), which both metrics take in
place of `X` with the same values; a list of `k` gives one value per `k`:

```python
reference = ub.Neighborhood(X)


def layout_loss(p):
    embedding = ub.som_olp(X, (8, 8), lam=p["lam"], gamma=1.0).embedding
    return -ub.trustworthiness(reference, embedding, k=10)


best = ub.minimize(layout_loss, {"lam": ub.loguniform(0.05, 5.0)}, n_trials=20)
ub.continuity(reference, m.embedding, k=[5, 10, 20])  # an array of three values
```

From about five dimensions on, FCM with the default `m = 2` can collapse:
every prototype goes to the mean of the data, every membership to 1/K, and
the run still stops with `"tol"`. Use `m` closer to 1 there (1.2, for
example) and check that the centers are apart; `efcm` collapses likewise
when `tau` is too large ([degenerate solutions](https://github.com/subukata/ubukit/blob/main/docs/algorithms.md#degenerate-solutions-fcm-efcm)).

A map's `unit_coordinates` are the coordinates of its units as the run used
them, not normalized: `(row, col)` in grid steps on a rectangular grid (unit
`j = row * cols + col`, so a plot can take x = col and y = row), or the
`(K, Q)` array given to `som` or `som_olp`. `centers[j]` is the prototype of
unit `j`, and `embedding` is `unit_coordinates[labels]` for `som` and
`batch_som` and `membership @ unit_coordinates` for `som_olp`.

`ub.steps` has every fitting function as a generator that yields after each
iteration (each epoch for maps). `p.result()` is the `Result` the run would
return had it stopped there, so the learning itself can be plotted or
recorded, without rerunning:

```python
frames = [p.result().embedding for p in ub.steps.som_olp(X, (8, 8), lam=0.5, gamma=1.0)]
```

The data are an input of every iteration: `run.send(X_new)` runs the next
iteration on new rows and keeps everything else the run has learned
(prototypes, schedules, and SOM-OLP's memberships while the number of rows
stays the same, row i being the same point as before). With `max_iter=None`
a run goes on for as long as data keep coming:

```python
run = ub.steps.som_olp(X, (8, 8), lam=0.5, gamma=1.0, max_iter=None)
p = next(run)
for t in range(1, 50):
    p = run.send(X + 0.05 * t)  # the data drift; the map follows from where it is
```

`som`, `trustworthiness`, `continuity` and `ami` also take `engine="numba"`,
which runs their loops compiled, with the same results: the online SOM
about 8 times faster, trustworthiness and continuity about 3 times per core
(25 times on 16 cores), and `ami` most when the labels have many distinct
cluster sizes.

```sh
pip install "ubukit[numba]"
```

```python
m = ub.som(X, (8, 8), engine="numba")
ub.trustworthiness(X, m.embedding, k=5, engine="numba")
```

Each kernel compiles on its first call in a process (0.3 to 1.3 s). The
parallel ones (trustworthiness, continuity, ami) use every core unless
`NUMBA_NUM_THREADS` limits them; their results do not depend on it.

Equations, defaults and references: <https://github.com/subukata/ubukit/blob/main/docs/algorithms.md>.
A JavaScript package with the same API in camelCase is available as `ubukit` on npm.
