"""Assignment-only, strict float64 finalization for restart-v2 k-means.

This is NEW v2 code. The recovered Lloyd core is unchanged. The target-neutral
register8 intrinsic is reused solely to calculate distances to final centers.
There is no centroid accumulation/update, square root, norm expansion or
fastmath. Feature components accumulate in increasing feature order. Lowest
center index wins equal computed squared distances. This is not a claim of
bitwise sklearn equivalence at near ties or after sklearn's recentering.
"""
from contextlib import contextmanager
from numbers import Integral
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from .kmeans_numba import _nearest8


@njit(cache=True, parallel=True)
def _assignment_register8(X, centers):
    """Require finite, nonempty, C-contiguous float64 arrays with matching D."""
    n, d = X.shape
    k = centers.shape[0]
    padded_k = ((k + 7) // 8) * 8
    centers_t = np.full((d, padded_k), np.inf, dtype=np.float64)
    for c in range(k):
        for j in range(d):
            centers_t[j, c] = centers[c, j]
    labels = np.empty(n, dtype=np.int64)
    squared_distances = np.empty(n, dtype=np.float64)
    for i in prange(n):
        best = 0
        best_distance = np.inf
        for c in range(0, k, 8):
            distance, unused, lane = _nearest8(X, i, centers_t, c)
            if distance < best_distance:
                best_distance = distance
                best = c + lane
        labels[i] = best
        squared_distances[i] = best_distance
    return labels, squared_distances


def _matrix(value, name):
    value = np.asarray(value)
    if value.dtype != np.float64:
        raise TypeError(f'{name} must have dtype float64')
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f'{name} must be a nonempty two-dimensional array')
    if not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite')
    return np.ascontiguousarray(value)


@contextmanager
def _threads(threads):
    old = get_num_threads()
    if threads is None:
        count = old
    else:
        if isinstance(threads, (bool, np.bool_)) or not isinstance(threads, Integral):
            raise TypeError('threads must be a positive integer or None')
        count = int(threads)
        if count < 1:
            raise ValueError('threads must be positive')
    set_num_threads(count)
    try:
        yield
    finally:
        set_num_threads(old)


def finalize(X, centers, threads=1):
    """Return labels, inertia and per-row squared distances to final centers.

    Validation/conversion, transposition, output allocation, assignment and
    inertia reduction are all performed by this call and belong inside any
    end-to-end fit timer. No input or center is mutated. The fixed NumPy sum of
    the distance vector makes inertia independent of Numba thread count.
    """
    X = _matrix(X, 'X')
    centers = _matrix(centers, 'centers')
    if X.shape[1] != centers.shape[1]:
        raise ValueError('X and centers must have matching feature dimensions')
    with _threads(threads):
        labels, squared_distances = _assignment_register8(X, centers)
    if not np.isfinite(squared_distances).all():
        raise ValueError('squared distances overflowed; rescale inputs')
    with np.errstate(over='ignore', invalid='ignore'):
        inertia = float(np.sum(squared_distances, dtype=np.float64))
    if not np.isfinite(inertia):
        raise ValueError('inertia overflowed; rescale inputs')
    return {'labels': labels, 'inertia': inertia,
            'squared_distances': squared_distances,
            'label_contract': 'nearest-final-centers; strict direct float64; first computed tie',
            'finalizer': 'register8_assignment_only'}
