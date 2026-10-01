"""Elide the full square-root pass without changing rounded-root ranks.

The strict contract is the current NumPy float32/float64 sqrt ufunc, NumPy's
installed default argsort, and sklearn's full Euclidean pairwise computation.
No squared-distance tie policy is silently substituted for rounded-root ties.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
from numba import njit, prange
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .metrics_numba import _ks, _quality, _validate_pair


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


@njit(cache=True, parallel=True)
def _threshold_scan(D, lower, upper):
    b, n = D.shape
    m = lower.shape[1]
    ranks = np.empty((b, m), dtype=np.int64)
    tied = np.zeros(b, dtype=np.bool_)
    for i in prange(b):
        any_ties = False
        for z in range(m):
            lo, hi = lower[i, z], upper[i, z]
            less, at_most = 0, 0
            for j in range(n):
                d = D[i, j]
                less += d < lo
                at_most += d <= hi
            ranks[i, z] = less + 1
            any_ties |= at_most - less > 1
        tied[i] = any_ties
    return ranks, tied


@njit(cache=True, parallel=True)
def _threshold_histogram(D, lower, upper):
    """One binary search per distance, including repeated query bins."""
    b, n = D.shape
    m = lower.shape[1]
    ranks = np.empty((b, m), dtype=np.int64)
    tied = np.zeros(b, dtype=np.bool_)
    for i in prange(b):
        order = np.argsort(lower[i])
        sorted_lower = lower[i, order]
        sorted_upper = upper[i, order]
        hist = np.zeros(m + 1, dtype=np.int64)
        eq = np.zeros(m, dtype=np.int64)
        for j in range(n):
            d = D[i, j]
            lo, hi = 0, m
            # First interval with a lower endpoint strictly greater than d.
            while lo < hi:
                mid = (lo + hi) // 2
                if d < sorted_lower[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            hist[lo] += 1
            if lo > 0 and d <= sorted_upper[lo - 1]:
                eq[lo - 1] += 1
        count = 0
        for z in range(m):
            count += hist[z]
            ranks[i, order[z]] = count + 1
            if eq[z] > 1:
                tied[i] = True
    return ranks, tied


def sqrt_queried_ranks(D_squared, Q, *, method='scan', return_stats=False):
    """Match argsort(np.sqrt(D_squared)) query ranks, including root ties.

    The caller excludes self using an infinite diagonal. No dtype conversion
    occurs. Distinct squared values in the same rounded-sqrt bin count as ties.
    A tied query triggers the actual NumPy default sort on that rooted row.
    Duplicated query indices do not themselves create a distance tie.
    """
    D = _validate_squared(D_squared)
    Q = np.asarray(Q)
    if D.ndim != 2 or Q.ndim != 2 or len(D) != len(Q) or D.shape[1] < 1:
        raise ValueError('D and Q must be 2D with matching rows and nonempty D columns')
    if Q.dtype.kind not in 'iu':
        raise TypeError('Q must contain integer indices')
    if Q.size and (Q.min() < 0 or Q.max() >= D.shape[1]):
        raise ValueError('Q contains an out-of-bounds index')
    if method not in ('scan', 'histogram'):
        raise ValueError('method must be scan or histogram')
    lower, upper = sqrt_intervals(np.take_along_axis(D, Q, axis=1))
    ranks, tied = (_threshold_scan if method == 'scan' else _threshold_histogram)(D, lower, upper)
    for i in np.flatnonzero(tied):
        # Must preserve float32 here: casting before sqrt can split root ties.
        order = np.argsort(np.sqrt(D[i]))
        inv = np.empty(D.shape[1], dtype=np.int64)
        inv[order] = np.arange(1, D.shape[1] + 1)
        ranks[i] = inv[Q[i]]
    if return_stats:
        return ranks, {'query_count': Q.size, 'fallback_rows': int(tied.sum()),
                       'total_rows': len(D)}
    return ranks


def _squared_distance_block(Z, start, stop, *, full):
    D = (pairwise_distances(Z, metric='euclidean', squared=True) if full else
         pairwise_distances(Z[start:stop], Z, metric='euclidean', squared=True))
    if not np.isfinite(D).all():
        raise ValueError('The metric produced non-finite squared distances; check overflow')
    D[np.arange(stop - start), np.arange(start, stop)] = np.inf
    return D


def joint_sklearn_sqrt(X, Y, ks=5, *, rank_method='scan', block_size=None):
    """Strict sklearn metric values, with rounded-sqrt interval rank queries.

    Always returns a list of the frozen core's Quality records. Full distances
    are the strict compatibility path. Blocks retain their usual near-tie
    caveat because their Gram multiplication call shape differs from sklearn.
    NN selection stays unchanged, separately per k; all integer penalties and
    normalization follow the frozen core. Spaces are processed sequentially.
    """
    X, Y = _validate_pair(X, Y)
    n, kvals = len(X), _ks(ks, len(X))
    if rank_method not in ('scan', 'histogram'):
        raise ValueError('rank_method must be scan or histogram')
    full = block_size is None
    if not full and (not isinstance(block_size, Integral) or block_size < 1):
        raise ValueError('block_size must be a positive integer or None')
    bsize = n if full else int(block_size)
    NX = np.concatenate([NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
                         for k in kvals], axis=1)
    NY = np.concatenate([NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(return_distance=False)
                         for k in kvals], axis=1)
    ends = np.cumsum(kvals)
    starts = np.r_[0, ends[:-1]]
    penalties = np.zeros((len(kvals), 2), dtype=np.int64)
    for axis, (Z, neighbors) in enumerate(((X, NY), (Y, NX))):
        for start in range(0, n, bsize):
            stop = min(n, start + bsize)
            D = _squared_distance_block(Z, start, stop, full=full)
            R = sqrt_queried_ranks(D, neighbors[start:stop], method=rank_method)
            for z, (k, lo, hi) in enumerate(zip(kvals, starts, ends)):
                penalties[z, axis] += np.maximum(R[:, lo:hi] - k, 0).sum(dtype=np.int64)
            del D, R
    return [_quality(k, n, *penalties[z]) for z, k in enumerate(kvals)]


def trustworthiness_continuity_sqrt(X, Y, n_neighbors=5, **kwargs):
    """Public scalar/iterable API matching trustworthiness_continuity."""
    results = joint_sklearn_sqrt(X, Y, n_neighbors, **kwargs)
    return results[0] if isinstance(n_neighbors, Integral) else results
