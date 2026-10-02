"""New-v2 NumPy-only squared-distance queries preserving rounded-sqrt ties."""
from numbers import Integral
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from ._neighbor_queries import neighbor_queries
from ._numpy import _ks, _validate_pair, _quality
from ._sqrt_intervals import _validate_squared, sqrt_intervals


def sqrt_queried_ranks_numpy(D_squared, Q, *, method='sortsearch', block_size=32,
                             max_scratch_bytes=32 * 2**20):
    D = _validate_squared(D_squared)
    Q = np.asarray(Q)
    if D.ndim != 2 or Q.ndim != 2 or len(D) != len(Q) or D.shape[1] < 1:
        raise ValueError('D and Q must be 2D with matching rows and nonempty D columns')
    if Q.dtype.kind not in 'iu':
        raise TypeError('Q must contain integer indices')
    if Q.size and (Q.min() < 0 or Q.max() >= D.shape[1]):
        raise ValueError('Q contains an out-of-bounds index')
    if method not in ('sortsearch', 'broadcast'):
        raise ValueError('method must be sortsearch or broadcast')
    if not isinstance(block_size, Integral) or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    if not isinstance(max_scratch_bytes, Integral) or max_scratch_bytes < 1:
        raise ValueError('max_scratch_bytes must be a positive integer')
    rows, n, m = len(D), D.shape[1], Q.shape[1]
    R = np.empty(Q.shape, dtype=np.int64)
    if not Q.size:
        return R
    bsize = int(block_size)
    qsize = m
    if method == 'broadcast':
        if max_scratch_bytes < n:
            raise ValueError('max_scratch_bytes must be at least the number of distance columns')
        qsize = min(m, int(max_scratch_bytes) // n)
        bsize = min(bsize, max(1, int(max_scratch_bytes) // (n * qsize)))
        scratch = np.empty((min(bsize, rows), qsize, n), dtype=np.bool_)
    for start in range(0, rows, bsize):
        stop = min(rows, start + bsize)
        lower, upper = sqrt_intervals(np.take_along_axis(D[start:stop], Q[start:stop], axis=1))
        tied = np.zeros(stop - start, dtype=np.bool_)
        if method == 'sortsearch':
            ordered = np.sort(D[start:stop], axis=1)
            for i in range(stop - start):
                less = np.searchsorted(ordered[i], lower[i], side='left')
                at_most = np.searchsorted(ordered[i], upper[i], side='right')
                R[start + i] = less + 1
                tied[i] = np.any(at_most - less > 1)
            del ordered
        else:
            for lo in range(0, m, qsize):
                hi = min(m, lo + qsize)
                work = scratch[:stop-start, :hi-lo]
                np.less(D[start:stop, None, :], lower[:, lo:hi, None], out=work)
                less = np.count_nonzero(work, axis=2)
                R[start:stop, lo:hi] = less + 1
                np.less_equal(D[start:stop, None, :], upper[:, lo:hi, None], out=work)
                tied |= np.any(np.count_nonzero(work, axis=2) - less > 1, axis=1)
        for local in np.flatnonzero(tied):
            i = start + local
            # Root before sorting, in the original dtype, exactly as sklearn.
            order = np.argsort(np.sqrt(D[i]))
            inverse = np.empty(n, dtype=np.int64)
            inverse[order] = np.arange(1, n + 1)
            R[i] = inverse[Q[i]]
    return R


def joint_sklearn_sqrt_numpy(X, Y, ks=5, *, rank_method='sortsearch', block_size=32,
                             max_scratch_bytes=32 * 2**20):
    X, Y = _validate_pair(X, Y)
    n, kvals = len(X), _ks(ks, len(X))
    NX = neighbor_queries(X, kvals)
    NY = neighbor_queries(Y, kvals)
    ends = np.cumsum(kvals)
    starts = np.r_[0, ends[:-1]]
    penalties = np.zeros((len(kvals), 2), dtype=np.int64)
    for axis, (Z, neighbors) in enumerate(((X, NY), (Y, NX))):
        D = pairwise_distances(Z, metric='euclidean', squared=True)
        for start in range(0, n, 256):
            if not np.isfinite(D[start:start+256]).all():
                raise ValueError('Euclidean squared distances overflowed; rescale the input')
        np.fill_diagonal(D, np.inf)
        R = sqrt_queried_ranks_numpy(D, neighbors, method=rank_method,
            block_size=block_size, max_scratch_bytes=max_scratch_bytes)
        for z, (k, lo, hi) in enumerate(zip(kvals, starts, ends)):
            penalties[z, axis] = np.maximum(R[:, lo:hi] - k, 0).sum(dtype=np.int64)
        del D, R
    return [_quality(k, n, *penalties[z]) for z, k in enumerate(kvals)]
