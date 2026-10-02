# Choosing an API

Use the [Python reference](../python/README.md) or [JavaScript reference](../javascript/README.md) for public entry points, parameters, and result contracts.

- **Clustering:** k-means, FCM, RCM / ExRCM, and RMCM
- **Self-organizing maps:** online SOM, BatchSOM, and SOM-OLP. Each algorithm has its own update order, initialization, and membership semantics
- **Evaluation:** trustworthiness and continuity for neighborhood quality; ARI and AMI for comparing cluster labels
- **Parameter search:** lightweight TPE or random search

## Python

Import the public API from `ubukit`. Paths under `ubukit._impl` are private implementation details and are not a stable public API.

See the [Python reference](../python/README.md), [SOM guide](../python/SOM.md), and [optimization guide](../python/OPTIMIZATION.md).

## JavaScript

Import from `ubukit-js` or one of the subpaths declared in its `package.json`. Use the [JavaScript reference](../javascript/README.md) to select synchronous calls, incremental sessions, or Worker clients.

See the [SOM guide](../javascript/SOM.md), [external metrics guide](../javascript/EXTERNAL_METRICS.md), and [optimization guide](../javascript/OPTIMIZATION.md).

## Data and numerical contracts

Python and JavaScript expose different container types and option names. Check the language-specific result layouts before exchanging data. Shared reference tests cover specific cross-language contracts; they do not guarantee universal bit-identical results.

Read the [numerical contracts and limits](numerics.md) and try the [examples](../examples/README.md).
