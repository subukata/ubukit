# Choosing an API

Use the [Python reference](../python/README.md) or [JavaScript reference](../javascript/README.md) for public entry points, parameters, and result contracts.

- **Clustering:** k-means assigns each sample to one cluster; fuzzy c-means (FCM) assigns degrees of membership
- **Rough clustering:** rough c-means (RCM) and extended rough c-means (ExRCM) allow overlapping cluster assignments; rough membership c-means (RMCM) derives membership from fixed-radius neighborhoods
- **Self-organizing maps (SOM):** online and batch training (BatchSOM) map samples to a grid for visualization
- **Self-organizing maps with optimized latent positions (SOM-OLP):** learn continuous sample positions using a supplied grid
- **Evaluation:** trustworthiness and continuity check neighborhood preservation; adjusted Rand index (ARI) and adjusted mutual information (AMI) measure chance-adjusted agreement between cluster labels
- **Parameter search:** Tree-structured Parzen Estimator (TPE) search uses earlier trial results to suggest parameter values; random search samples without that feedback

Each map algorithm has its own update order, initialization, and membership semantics.

## Python

Import the public API from `ubukit`. Paths under `ubukit._impl` are private implementation details and are not a stable public API.

See the [Python reference](../python/README.md), [SOM guide](../python/SOM.md), and [optimization guide](../python/OPTIMIZATION.md).

## JavaScript

Import from `ubukit-js` or one of the subpaths declared in its `package.json`. Use the [JavaScript reference](../javascript/README.md) to select synchronous calls, incremental sessions, or Worker clients.

See the [SOM guide](../javascript/SOM.md), [external metrics guide](../javascript/EXTERNAL_METRICS.md), and [optimization guide](../javascript/OPTIMIZATION.md).

## Data and numerical contracts

Python and JavaScript expose different container types and option names. Check the language-specific result layouts before exchanging data. Shared reference tests cover specific cross-language contracts; they do not guarantee universal bit-identical results.

Read the [numerical contracts and limits](numerics.md) and try the [examples](../examples/README.md).
