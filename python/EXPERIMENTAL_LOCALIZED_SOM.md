> Namespace dev6: this remains an unsupported private experiment under `ubukit._impl`; it is not a new facade export. The dev5 numerical implementation is unchanged.

# Experimental, opt-in SOM-OLP probability-tail localization

The dev5 candidate added the localization experiment. In this namespace-only dev6
candidate it lives at `ubukit._impl.portable_accel.som_olp_localized`.
Existing exports, backend implementations, and public defaults are unchanged.
The preserved verified dev4 source and built distributions remain the baseline.
This is not a supported public API or a universally faster replacement.

Explicit use:

```python
from ubukit._impl.portable_accel.som_olp_localized import fit_som_olp_localized
result = fit_som_olp_localized(X, R, gamma=gamma, lam=lam,
    backend="cdist_optimized", policy=policy)
```

`run_som_olp_localized` accepts existing W0/P0. The candidate activates only when
initial probabilities trigger the dev4 extreme-range check, coordinates and
parameters do not, and a direct NumPy/SciPy backend was requested. Other requests
retain the original implementation. Small scratch caps also retain the original
path when the vectorized candidate cannot fit its primary buffers.

Tiny nonempty prototype-column masses (<1e-140), prototype means near signed
cancellation (|W| <= 1e-10 times the feature maximum), and grid reductions near
cancellation (|V| <= 2e-10 times the grid-coordinate maximum) are recomputed from
exact binary products. Other means use float64 BLAS. These are conservative
heuristic guards, not dimension-independent error proofs or a guarantee that all
means are exact. Unsafe cost rows use the original extended-exponent arithmetic.
Their low-cost threshold scales by max(1,gamma) and the coordinate dimension so
gamma-amplified grid-square underflow cannot be hidden by a larger data cost.
Min-before-temperature normalization, small entropy tails, original-unit
objectives and stopping, lagged W/V output order, and detached ownership remain.

Ordinary direct costs and unguarded means can differ from the fully scalar cold
path by float64 rounding. No bitwise parity, globally closer trajectory,
universal speedup, or arbitrary-precision solver is promised. Keep this opt-in
until broader adversarial, high-dimensional, cross-platform and workload review.
The complete original cold implementation remains available through the existing
`run_som_olp` and `fit_som_olp` functions.

The primary scratch cap excludes validated inputs, outputs, Python integer
reductions and library/runtime workspace, as documented by the result metadata.
Only NumPy/SciPy standard dependencies are needed; optional accelerated backends
are neither removed nor made mandatory.
