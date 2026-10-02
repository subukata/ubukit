"""SOM cost normalization preserving ordinary divide-then-shift arithmetic."""
import numpy as np

_MAX = float(np.finfo(np.float64).max)
_EPS = float(np.finfo(np.float64).eps)
_MAX_EPS = _MAX * _EPS


def normalize_costs_inplace(cost, lam):
    """Normalize finite cost rows in place; return original minima and totals.

    Ordinary rows retain the historical divide-then-shift operation order.
    Rows whose division would overflow or erase a temperature-scale gap
    under a large common offset are shifted while costs are finite.
    Overflow to negative infinity for a nonminimal shifted cost is harmless:
    every row still has a finite zero logit and a positive normalizer.
    """
    minimum = cost.min(axis=1)
    maximum = cost.max(axis=1)
    if not (np.isfinite(minimum).all() and np.isfinite(maximum).all()):
        raise ValueError("SOM costs overflowed; rescale inputs or parameters")
    limit = _MAX * lam if lam < 1.0 else np.inf
    unsafe = (minimum <= -limit) | (maximum >= limit)
    # If the common quotient reaches float64 integer resolution, adjacent
    # finite costs can round to the same quotient before the softmax shift.
    # Keep historical ordinary arithmetic, but subtract before dividing here.
    precision_limit = lam / _EPS if lam <= _MAX_EPS else np.inf
    unsafe |= np.abs(minimum) >= precision_limit
    with np.errstate(over="ignore", invalid="ignore"):
        for row in np.flatnonzero(unsafe):
            np.subtract(cost[row], minimum[row], out=cost[row])
        np.divide(cost, -lam, out=cost)
        cost -= cost.max(axis=1, keepdims=True)
        np.exp(cost, out=cost)
    total = cost.sum(axis=1)
    cost /= total[:, None]
    return minimum, total


def probabilities_from_costs(cost, lam):
    normalize_costs_inplace(cost, lam)
    return cost


def repair_tiny_mass_prototypes(X, P, den, W):
    """Repair only nonempty columns whose tiny mass can underflow P.T @ X.

    A tiny probability may first appear after an ordinary first iteration, so
    the public initial-range dispatch alone cannot detect this case. The
    ordinary numerator/update remains unchanged for every other column.
    """
    denominator = np.ravel(den)
    small = (denominator > 0) & (denominator < 1e-140)
    if np.any(small):
        from ._som_extreme import _weighted_mean
        for j in np.flatnonzero(small):
            for f in range(X.shape[1]):
                W[j, f] = _weighted_mean(X[:, f], P[:, j])
