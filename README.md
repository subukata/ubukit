<p align="center">
  <img src="docs/assets/ubukit-logo-b.png" alt="UbuKit" width="520">
</p>

# UbuKit

A Python and JavaScript library for clustering, self-organizing maps, evaluation metrics, and lightweight parameter search.

- **Clustering:** k-means assigns each sample to one cluster; fuzzy c-means (FCM) assigns degrees of membership
- **Rough clustering:** rough c-means (RCM) and extended rough c-means (ExRCM) allow overlapping cluster assignments; rough membership c-means (RMCM) derives membership from fixed-radius neighborhoods
- **Self-organizing maps (SOM):** online and batch training (BatchSOM) map samples to a grid for visualization
- **Self-organizing maps with optimized latent positions (SOM-OLP):** learn continuous sample positions using a supplied grid
- **Evaluation:** trustworthiness and continuity check neighborhood preservation; adjusted Rand index (ARI) and adjusted mutual information (AMI) measure chance-adjusted agreement between cluster labels
- **Parameter search:** Tree-structured Parzen Estimator (TPE) search uses earlier trial results to suggest parameter values; random search samples without that feedback

UbuKit is a pre-release preview and is not yet published to PyPI or npm. Follow [getting started](docs/getting-started.md) to build and install the packages locally.

## Python

```python
import numpy as np
import ubukit

X = np.array([[0.0], [0.1], [4.0], [4.1]], dtype=np.float64)
result = ubukit.fit_fcm(X, 2, random_state=1, max_iter=20, backend="numpy")
print(result["labels"])

som = ubukit.som_batch(X, grid_shape=(4, 3), epochs=2, random_state=1)
print(som["embedding"])
```

Use `import ubukit` for the public Python API. See the [Python reference](python/README.md) for inputs, result shapes, and optional backends.

## JavaScript

```javascript
import { run } from 'ubukit-js';

const X = { data: new Float64Array([0, 0.1, 4, 4.1]), nSamples: 4, nFeatures: 1 };
const result = run('kmeans', X, { nClusters: 2, seed: 1, maxIterations: 20 });
console.log(result.labels);
```

Use ES modules in Node.js or the browser, with no runtime npm dependencies. See the [JavaScript reference](javascript/README.md) for data layouts, sessions, and Workers.

## Guides

- [Getting started](docs/getting-started.md)
- [Choosing an API](docs/api.md)
- [Numerical contracts and limits](docs/numerics.md)
- [Understanding performance](docs/performance.md)
- [Examples](examples/README.md)

## Development

See [development and verification](docs/contributing.md) for contributor instructions.

## License

UbuKit project contributions use the [MIT License](LICENSE). Third-party terms and pending-origin component exclusions are described in [license scope](LICENSE-SCOPE.txt). The private preview is not approved for public release.
