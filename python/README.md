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
r.centers, r.membership, r.labels, r.n_iter, r.converged, r.history

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

Every fit returns a `Result` with `centers`, `labels`, `membership` (`None` for
hard methods), `n_iter`, `converged`, `history` (objective per iteration) and
`embedding` (maps). Inputs should be finite and of ordinary scale; standardize
features first.

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
A JavaScript package with the same API is available as `ubukit` on npm.
