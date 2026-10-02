"""Assignment-only, strict float64 finalization for restart-v2 k-means.

The target-neutral strict SIMD intrinsics are reused solely to calculate
four rows of distances to final centers, with adaptive 4/8/16-center tiles.
There is no centroid accumulation/update, square root, norm expansion or
fastmath. Feature components accumulate in increasing feature order. Lowest
center index wins equal computed squared distances. This is not a claim of
bitwise sklearn equivalence at near ties or after sklearn's recentering.
"""
from contextlib import contextmanager
from numbers import Integral
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from .kmeans_numba import (_nearest8, _nearest4rows4, _nearest8rows4,
                           _nearest16rows4, _select_vector_width)


@njit(cache=True, parallel=True)
def _assignment_register8(X, centers, vector_width=8):
    """Require finite, nonempty, C-contiguous float64 arrays with matching D."""
    n, d = X.shape
    k = centers.shape[0]
    padded_k = max(((k + 7) // 8) * 8,
                   ((k + vector_width - 1) // vector_width) * vector_width)
    centers_t = np.full((d, padded_k), np.inf, dtype=np.float64)
    for j in range(d):
        for c in range(k):
            centers_t[j, c] = centers[c, j]
    labels = np.empty(n, dtype=np.int64)
    squared_distances = np.empty(n, dtype=np.float64)
    for group in prange((n + 3) // 4):
        i = 4 * group
        if i + 3 < n:
            d0, d1, d2, d3 = np.inf, np.inf, np.inf, np.inf
            b0, b1, b2, b3 = 0, 0, 0, 0
            for c in range(0, k, vector_width):
                if vector_width == 4:
                    a0,a1,a2,a3,l0,l1,l2,l3 = _nearest4rows4(X, i, centers_t, c)
                elif vector_width == 16:
                    a0,a1,a2,a3,l0,l1,l2,l3 = _nearest16rows4(X, i, centers_t, c)
                else:
                    a0,a1,a2,a3,l0,l1,l2,l3 = _nearest8rows4(X, i, centers_t, c)
                if a0 < d0: d0, b0 = a0, c + l0
                if a1 < d1: d1, b1 = a1, c + l1
                if a2 < d2: d2, b2 = a2, c + l2
                if a3 < d3: d3, b3 = a3, c + l3
            labels[i], labels[i+1], labels[i+2], labels[i+3] = b0,b1,b2,b3
            squared_distances[i], squared_distances[i+1], squared_distances[i+2], squared_distances[i+3] = d0,d1,d2,d3
        else:
            for row in range(i, n):
                best = 0
                best_distance = np.inf
                for c in range(0, k, 8):
                    distance, unused, lane = _nearest8(X, row, centers_t, c)
                    if distance < best_distance:
                        best_distance = distance
                        best = c + lane
                labels[row] = best
                squared_distances[row] = best_distance
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
        labels, squared_distances = _assignment_register8(X, centers, _select_vector_width(len(centers), X.shape[1]))
    if not np.isfinite(squared_distances).all():
        raise ValueError('squared distances overflowed; rescale inputs')
    with np.errstate(over='ignore', invalid='ignore'):
        inertia = float(np.sum(squared_distances, dtype=np.float64))
    if not np.isfinite(inertia):
        raise ValueError('inertia overflowed; rescale inputs')
    return {'labels': labels, 'inertia': inertia,
            'squared_distances': squared_distances,
            'label_contract': 'nearest-final-centers; strict direct float64; first computed tie',
            # Historical result identifier retained for compatibility.
            'finalizer': 'register8_assignment_only'}
