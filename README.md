<p align="center">
  <img src="docs/assets/ubukit-logo-b.png" alt="UbuKit" width="420">
</p>

# UbuKit

Fuzzy, rough and classical clustering, self-organizing maps, evaluation metrics
and TPE search, for Python and JavaScript.

| | |
|---|---|
| Clustering | k-means, fuzzy c-means (FCM), entropy-regularized FCM, rough c-means (RCM/ExRCM), rough membership c-means (RMCM) |
| Maps | online SOM, batch SOM, SOM with optimized latent positions (SOM-OLP) |
| Metrics | adjusted Rand index, adjusted mutual information, trustworthiness, continuity |
| Search | multivariate Tree-structured Parzen Estimator |

Both packages share names, parameters and equations ([docs/algorithms.md](docs/algorithms.md)),
each in its language's spelling (`batch_som` and `max_iter` in Python,
`batchSom` and `maxIter` in JavaScript).
Python depends only on NumPy and SciPy (Numba optionally, for compiled
kernels); JavaScript has no dependencies.

## Python

```sh
pip install ubukit
```

It needs Python 3.12 or newer. Usage: [python/README.md](python/README.md).

```python
import numpy as np
import ubukit as ub

X = np.random.default_rng(0).normal(size=(500, 2))
r = ub.fcm(X, 3, m=2.0, seed=0)
r.centers, r.membership, r.labels, r.stop_reason
```

## JavaScript

```sh
npm install ubukit
```

It needs Node.js 22 or newer, or a current browser. Usage:
[js/README.md](js/README.md).

```js
import { fcm } from 'ubukit';

const X = [[0, 0], [0.1, 0.2], [4, 4], [4.2, 3.9]];  // rows, or { data, rows, cols }
const r = fcm(X, 2, { m: 2, seed: 0 });               // r.centers is { data: Float64Array, rows, cols }
```

A stop reason says how a run stopped, not that its result is good: from
about five dimensions on, FCM with `m = 2` can put every prototype at the
mean of the data and still stop with `"tol"`
([degenerate solutions](docs/algorithms.md#degenerate-solutions-fcm-efcm)).

## More

[docs/recipes.md](docs/recipes.md) has worked uses, such as SOM-OLP's
scales, held-out data, points added to a trained map and federated rough
c-means.

## Development

[DESIGN.md](DESIGN.md) explains what UbuKit is and why,
[CONTRIBUTING.md](CONTRIBUTING.md) how to change it, and
[CHANGELOG.md](CHANGELOG.md) what changed. Licensed under [MIT](LICENSE);
three small expressions in ARI and AMI follow scikit-learn under BSD-3-Clause
([THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)).
