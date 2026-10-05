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

Equations, defaults and references: <https://github.com/subukata/ubukit/blob/main/docs/algorithms.md>.
A JavaScript package with the same API is available as `ubukit` on npm.
