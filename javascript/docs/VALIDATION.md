# Validation status

## Executed

- Node.js v24.19.0 on Linux x64
- Standard FCM against Python explicit-membership fixture, m=1.3, 2, 3.5, seven fixed iterations; mixed absolute/relative tolerance 2e-12
- ExRCM against Python explicit-center fixture, p=.5, 1, 2, 3, 1000; centers tolerance 2e-12, membership exact
- k-means against Python public numpy core / sklearn finalizer fixture; no ambiguous distance ties, same explicit centers
- SOM-OLP and exact neighborhood metrics against independent Python fixtures, as detailed in SOM_AND_NEIGHBORHOOD.md
- Node Worker thread: result parity, progress, copied input, transferred input detachment, error recovery, pre-abort, busy state, abort termination, reuse, dispose
- Cooperative runAsync yields to timer while preserving result
- Numeric validation, zero distances, tiny/huge fuzzy m, fractional-power underflow recovery, tiny ExRCM p, memory guards

The final aggregate count is 58 passing tests. Four added tests verify the corrected linear-normalization Float64Array FCM benchmark reference against production/Python fixtures and zero-distance handling. Run `npm test` to reproduce. Historical console logs are not required by the package. See the [performance summary](../../docs/PERFORMANCE_JA.md) for the measured scope.

## Browser not yet verified in this environment

The actual cloud-browser attempt to load `http://localhost:8765/demo/test.html` was blocked by the client with `net::ERR_BLOCKED_BY_CLIENT` before the page loaded. This is an access/environment restriction, not evidence of algorithm or UI failure. It was not bypassed. No deployment was performed.

The runnable browser verification harness remains included. It exercises all six API names (five algorithm categories, with RCM/ExRCM separate), module Web Workers, synchronous/worker result identity, progress, main-thread timer responsiveness, cancellation after work has started, worker reuse, and transferred-buffer detachment.

Node Worker results must not be described as browser measurements. Browser performance and visual QA are pending until this harness can be opened in a supported local browser/preview environment.

## Remaining limits

- Python and JavaScript floating arithmetic can differ near exact thresholds or ties; the declared tie/threshold rules are preserved, not universal bit identity
- JS PCA uses a covariance/Jacobi eigensolver and has an explicit demo-size initialization guard; use shared prepared initialization for strict cross-language kernel comparisons
- Finite data whose squared differences overflow or collapse nonzero differences to zero are rejected by clustering; rescale such input
- maxMemoryBytes bounds estimated primary numeric arrays, not total process RSS, JIT workspace or Worker structured-clone copies
- WASM/GPU are not implemented or benchmarked; no universal fastest-implementation claim
