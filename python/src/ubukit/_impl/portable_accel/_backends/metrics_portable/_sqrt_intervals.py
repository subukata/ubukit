"""NumPy-only rounded-sqrt intervals retained from the recovered candidate."""
import numpy as np


def _validate_squared(values):
    values = np.asarray(values)
    if values.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise TypeError('Squared distances must have dtype float32 or float64')
    if np.isnan(values).any() or (values < 0).any():
        raise ValueError('Squared distances must be nonnegative and contain no NaNs')
    return values


def sqrt_intervals(values):
    """Inclusive input endpoints mapping to each query's rounded sqrt.

    Endpoints and every sqrt/nextafter operation retain the input float dtype.
    Walk adjacent representable nonnegative floats until the actual NumPy sqrt
    value changes. The loops have no approximate tolerance or iteration cutoff.
    Zero and +inf terminate explicitly. Subnormals are retained, not flushed.

    A correctly rounded sqrt is monotone. Thus each equal-output set is an
    interval in the finite ordered floating-point input set; finding the first
    different value on each side certifies its complete preimage. The algorithm
    does not rely on an assumed bound on the number of floats in the interval.
    """
    values = _validate_squared(values)
    roots = np.sqrt(values)
    lower, upper = values.copy(), values.copy()
    # Normalize signed zero; comparisons and sqrt ranks treat both zeros equal.
    lower[values == 0] = 0
    upper[values == 0] = 0
    shape = values.shape
    lower, upper, roots = lower.reshape(-1), upper.reshape(-1), roots.reshape(-1)
    active = np.flatnonzero(lower > 0)
    zero = np.array(0, dtype=values.dtype)
    infinity = np.array(np.inf, dtype=values.dtype)
    with np.errstate(under='ignore', over='ignore'):
        while active.size:
            candidate = np.nextafter(lower[active], zero)
            same = np.sqrt(candidate) == roots[active]
            active = active[same]
            lower[active] = candidate[same]
            active = active[lower[active] > 0]
        active = np.flatnonzero(np.isfinite(upper))
        while active.size:
            candidate = np.nextafter(upper[active], infinity)
            same = np.sqrt(candidate) == roots[active]
            active = active[same]
            upper[active] = candidate[same]
            active = active[np.isfinite(upper[active])]
    return lower.reshape(shape), upper.reshape(shape)

