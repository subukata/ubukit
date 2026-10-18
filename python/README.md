# UbuKit

Fuzzy, rough and classical clustering, self-organizing maps, evaluation metrics
and TPE search on NumPy and SciPy.

```sh
pip install ubukit
```

It needs Python 3.12 or newer.

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
(N, K; `None` for `kmeans`, `som` and `batch_som`), for the maps `embedding`
(N, Q) and `unit_coordinates` (K, Q), `n_iter`, `stop_reason` (`"tol"`,
`"schedule"` or `"limit"`) and `history`, the objective at each assignment
(empty for `rcm`, `rmcm`, `som` and `batch_som`, which have none). A map's
unit `j` sits at `unit_coordinates[j]`, `(row, col)` on a grid with
`j = row * cols + col`, and has the prototype `centers[j]`.

A Result is the state of its last iteration: `labels` and `membership` come
from the last assignment and `centers` from the update after it, so
assigning the data to `centers` again, as `ub.apply` does, can change them,
most after an early stop (`"limit"`). `som` and `batch_som` instead give the
best-matching units of their final prototypes.

The stop reason says how a run stopped, not that its result is good: from
about five dimensions on, FCM with the default `m = 2` can put every
prototype at the mean of the data and still stop with `"tol"`. Use `m`
closer to 1 there (1.2, for example) and check that the centers are apart.
Inputs must be finite numbers of ordinary scale; standardize features first.
A pickled `Result` or `TPE` is promised to load only in the version that
wrote it. The equations, the stopping rules and when FCM and EFCM collapse are in
[docs/algorithms.md](https://github.com/subukata/ubukit/blob/v0.2.0/docs/algorithms.md).

Trained prototypes apply to new data, fixed: `ub.apply` has a function for
every method, which takes the data, the prototypes and the parameters they
were trained with, and returns the `Result` of the new data (`ub.steps.apply`
has the same functions as generators). The parameters have no defaults,
since only the trained ones are right: keep them in one dictionary for
both calls.

```python
X_new = np.random.default_rng(1).normal(size=(20, 2))
new = ub.apply.som_olp(X_new, m.centers, m.unit_coordinates, lam=0.5, gamma=1.0)
new.embedding  # the new points on the trained map
ub.apply.fcm(X_new, r.centers, m=2.0).membership  # the m that r was trained with
```

Each method assigns once, except `som_olp`, whose memberships iterate with
the map fixed until they settle, so for the points the map was trained on
they need not end at the trained `embedding`; `rmcm`'s memberships depend on
the points applied together
([applying trained prototypes](https://github.com/subukata/ubukit/blob/v0.2.0/docs/algorithms.md#applying-trained-prototypes)).

A search that scores many embeddings of one `X` computes its distances once
with `Neighborhood(X)` (8 N^2 bytes), which both metrics take in place of
`X`; a list of `k` gives one value per `k`:

```python
reference = ub.Neighborhood(X)


def layout_loss(p):
    embedding = ub.som_olp(X, (8, 8), lam=p["lam"], gamma=1.0).embedding
    return -ub.trustworthiness(reference, embedding, k=10)


best = ub.minimize(layout_loss, {"lam": ub.loguniform(0.05, 5.0)}, n_trials=20)
ub.continuity(reference, m.embedding, k=[5, 10, 20])  # an array of three values
```

`ub.steps` has every fitting function as a generator that yields after each
iteration (each epoch for maps). `p.result()` is the `Result` the run would
return had it stopped there, so the learning itself can be plotted or
recorded:

```python
frames = [p.result().embedding for p in ub.steps.som_olp(X, (8, 8), lam=0.5, gamma=1.0)]
```

The data are an input of every iteration: `run.send(X_new)` runs the next
iteration on new rows and keeps what the run has learned (prototypes,
schedules, and SOM-OLP's memberships while row i stays the same point).
With `max_iter=None` a run goes on for as long as data keep coming:

```python
run = ub.steps.som_olp(X, (8, 8), lam=0.5, gamma=1.0, max_iter=None)
p = next(run)
for t in range(1, 50):
    p = run.send(X + 0.05 * t)  # the data drift; the map follows from where it is
```

`som`, `trustworthiness`, `continuity` and `ami` also take `engine="numba"`,
which runs their loops compiled, with the same results:

```sh
pip install "ubukit[numba]"
```

```python
m = ub.som(X, (8, 8), engine="numba")
ub.trustworthiness(X, m.embedding, k=5, engine="numba")
```

Each kernel compiles on its first call in a process. The parallel ones
(trustworthiness, continuity, ami) use every core unless `NUMBA_NUM_THREADS`
limits them; their results do not depend on it.

Worked uses: <https://github.com/subukata/ubukit/blob/v0.2.0/docs/recipes.md>.
A JavaScript package with the same API in camelCase is available as `ubukit` on npm.
