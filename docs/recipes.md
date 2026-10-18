# Recipes

Worked uses of UbuKit, in Python; JavaScript has the same functions in its
own spelling (`ub.apply.somOlp`, `unitCoordinates`). This file follows the
main branch: a recipe marked *Unreleased* needs the development version,
and the recipes of a release are at its tag
(`https://github.com/subukata/ubukit/tree/v<version>/docs`). The blocks run
as written, in order, after this one:

```python
import numpy as np

import ubukit as ub

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c, 0.6, (200, 2)) for c in ([0, 0], [4, 0], [1, 4])])
```

## A first experiment, end to end

Split the data first and compute every statistic on the training part
only; fit and apply with one dictionary of settings; and save what applying
needs (the centers, the settings, the preprocessing, the order of the
features and the version that made them) as arrays and JSON, which
`np.load` reads without pickles. A map also needs its `unit_coordinates`.

```python
import io
import json

y = np.repeat([0, 1, 2], 200)  # X's groups, to score against
features = ["length", "width"]  # the columns, in their order
order = np.random.default_rng(1).permutation(len(X))
train, new = order[:450], order[450:]

mean = X[train].mean(axis=0)
scale = X[train].std(axis=0)
scale[scale == 0] = 1.0  # a constant feature is only centered
settings = {"m": 1.5}
fit = ub.fcm((X[train] - mean) / scale, 3, **settings, seed=0)
print(fit.stop_reason, ub.ari(y[train], fit.labels))

saved = io.BytesIO()  # or a file: np.savez("model.npz", ...)
np.savez(
    saved,
    centers=fit.centers,
    mean=mean,
    scale=scale,
    features=features,
    settings=json.dumps(settings),
    version=ub.__version__,
)

model = np.load(io.BytesIO(saved.getvalue()))  # or np.load("model.npz")
assert list(model["features"]) == features
standardized = (X[new] - model["mean"]) / model["scale"]
applied = ub.apply.fcm(standardized, model["centers"], **json.loads(str(model["settings"])))
print(ub.ari(y[new], applied.labels))
before = ub.apply.fcm((X[new] - mean) / scale, fit.centers, **settings)
assert np.array_equal(applied.membership, before.membership)  # saved exactly

# Drawn with matplotlib, which UbuKit does not depend on:
# import matplotlib.pyplot as plt
# plt.scatter(*X[new].T, c=applied.labels, s=8)
# plt.scatter(*(model["centers"] * model["scale"] + model["mean"]).T, c="k", marker="x")
```

## SOM-OLP's scales and diagnostics

SOM-OLP adds a squared distance in the data and one on the grid,
$\lVert x_i - w_j \rVert^2 + \gamma \lVert v_i - r_j \rVert^2$, and divides
the sum by $\lambda$. So $\lambda$ is in squared data units, and $\gamma$
converts squared grid units into squared data units: scaling the data by
$a$ and the grid by $b$ gives the same memberships when $\lambda$ is
scaled by $a^2$ and $\gamma$ by $a^2 / b^2$. Choosing both as fractions of
the data's variance (with one grid step as the grid's unit) carries a
choice over to data of another scale. With $\gamma = 0$, a $\lambda$ from
about twice the largest variance of the data draws every prototype to the
mean ("Degenerate solutions" in [algorithms.md](algorithms.md)); the
latent term changes that threshold, so check the spread of the prototypes.

```python
variance = X.var(axis=0).sum()  # the mean squared distance to the mean
collapse = 2 * np.linalg.eigvalsh(np.cov(X.T, bias=True)).max()  # gamma = 0: lam above
m = ub.som_olp(X, (8, 8), lam=0.02 * variance, gamma=0.02 * variance)
```

Two readings of the memberships tell how hard they are and whether the
points snap to the units:

```python
P = m.membership
entropy = -(P * np.log(np.where(P > 0, P, 1.0))).sum(axis=1)
effective_units = np.exp(entropy)  # 1: one unit; K: all units alike
to_unit = np.linalg.norm(m.embedding[:, None] - m.unit_coordinates[None], axis=2).min(axis=1)
print(np.median(effective_units), np.mean(to_unit < 0.1))  # how many points sit on a unit
```

## Watching rough c-means converge

RCM has no objective, so its `history` is empty; its generator shows how
many points change their memberships at each iteration. Compare whole rows:
a point can change its shares without changing its strongest cluster.

```python
previous = None
for p in ub.steps.rcm(X, 5, alpha=1.3, seed=0):  # five clusters for three blobs: slow
    U = p.result().membership
    if previous is not None:
        print(p.iteration, int(np.any(U != previous, axis=1).sum()))
    previous = U
```

## Memberships of held-out data

Train on one part and apply to another: the centers stay as trained, and
the memberships follow the trained model's own rule, with its parameters;
one dictionary keeps them the same in both calls.

```python
order = rng.permutation(len(X))
calibration, test = X[order[:400]], X[order[400:]]
settings = {"alpha": 1.2, "beta": 0.0, "p": 1.0}
trained = ub.rcm(calibration, 3, **settings, seed=0)
held_out = ub.apply.rcm(test, trained.centers, **settings)
print(held_out.membership[:5])  # a boundary point shares 1/2 or 1/3
```

## Adding points to a trained map

Retraining (`ub.som_olp` on all the points) moves the map; applying keeps
it and places the new points on it.

```python
m = ub.som_olp(X, (8, 8), lam=0.5, gamma=1.0)
new_points = rng.normal(2.0, 1.5, (10, 2))
placed = ub.apply.som_olp(new_points, m.centers, m.unit_coordinates, lam=0.5, gamma=1.0)
positions = np.vstack([m.embedding, placed.embedding])
print(placed.stop_reason)  # "tol"; "limit" if max_iter (1000) came first
```

Each new point's position starts from its distances alone and settles with
the map fixed. Its update depends on it alone, but one rule stops all the
points given together, so a point applied alone and with others can end in
different places, mostly when it lies midway between two units. The
trained points keep where training left them; applying to them too gives positions consistent with the final map,
which can differ ("Applying trained prototypes" in
[algorithms.md](algorithms.md)).

## Federated rough c-means

Clients keep their rows: each computes the memberships of its rows for the
shared centers and sends two sums, and a server divides them. From the same
start, it computes the centralized run's updates in exact arithmetic; in
floating point the sums differ in their last digits, which can tip a point
at the edge of an admissible set, and the runs can then part.

```python
clients = np.array_split(rng.permutation(X), 4)
start = X[[0, 200, 400]]
centers = start
for _ in range(300):
    sums, masses = np.zeros_like(centers), np.zeros(len(centers))
    for part in clients:  # on each client
        U = ub.apply.rcm(part, centers, alpha=1.2, beta=0.0, p=1.0).membership
        sums += U.T @ part  # sends sum_i u_ic x_i
        masses += U.sum(axis=0)  # and sum_i u_ic
    new = centers.copy()  # on the server; a center without mass stays
    new[masses > 0] = sums[masses > 0] / masses[masses > 0, None]
    if np.array_equal(new, centers):
        break
    centers = new
central = ub.rcm(X, 3, alpha=1.2, init=start)
print(np.abs(centers - central.centers).max())  # here the same, to rounding
```

For FCM the weights are $u^m$, and `U**m` underflows for large $m$: FCM's
own update divides each column of $U$ by its maximum before the power.
Federated, every client must use the same divisor: the server first
gathers the clients' column maxima of `U` and sends back their maximum `M`,
and each client sends the sums of `W = (U / M) ** m` (`W.T @ part` and
`W.sum(axis=0)`).

## Separating points on screen

Points can overlap: `som` and `batch_som` put the points of a unit on its
coordinates, and SOM-OLP's positions can lie close together. Spread them
for display only, and compute metrics on the embedding itself.

```python
shown = m.embedding + rng.normal(0.0, 0.15, m.embedding.shape)  # drawing only
score = ub.trustworthiness(X, m.embedding, k=10)  # not on `shown`
```

## Resuming a search

*Unreleased; planned for 0.3.0.* A TPE's proposals follow from its seed and
the trials told, so a search resumes from its result, in another process
too: a new `TPE` with the same space, settings and seed, told the trials in
order, goes on as the search would have.

```python
space = {"x": ub.uniform(-5, 5), "y": ub.uniform(-5, 5)}


def loss(p):
    return (p["x"] - 1) ** 2 + p["y"] ** 2


first = ub.minimize(loss, space, 20)  # first.seed is the seed drawn

search = ub.TPE(space, seed=first.seed)
for params, value in zip(first.params, first.values):
    search.tell(params, value)
for _ in range(20):
    params = search.ask()
    search.tell(params, loss(params))
assert search.result().params == ub.minimize(loss, space, 40, seed=first.seed).params
```
