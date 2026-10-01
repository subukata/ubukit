"""V2 rank-only blocking around recovered strict Numba query kernels."""
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from ._numpy import _ks, _validate_pair, _quality


def joint_strict_numba(X, Y, ks=5, *, rank_method='scan', block_rows=256,
                       sqrt_elision=False):
    X, Y = _validate_pair(X, Y)
    n, kvals = len(X), _ks(ks, len(X))
    if sqrt_elision:
        from ._sqrt_numba import sqrt_queried_ranks
        if rank_method not in ('scan', 'histogram'):
            raise ValueError('sqrt Numba rank_method must be scan or histogram')
    else:
        from ._numba_core import queried_ranks
        if rank_method not in ('scan', 'histogram', 'full'):
            raise ValueError('Numba rank_method must be scan, histogram, or full')
    NX = np.concatenate([NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
                         for k in kvals], axis=1)
    NY = np.concatenate([NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(return_distance=False)
                         for k in kvals], axis=1)
    ends = np.cumsum(kvals)
    starts = np.r_[0, ends[:-1]]
    penalties = np.zeros((len(kvals), 2), dtype=np.int64)
    for axis, (Z, neighbors) in enumerate(((X, NY), (Y, NX))):
        # A single FULL self-distance call per space. Blocking starts only below.
        D = pairwise_distances(Z, metric='euclidean', squared=True) if sqrt_elision else pairwise_distances(Z, metric='euclidean')
        for start in range(0, n, 256):
            if not np.isfinite(D[start:start+256]).all():
                raise ValueError('Euclidean distances overflowed; rescale the input')
        np.fill_diagonal(D, np.inf)
        for start in range(0, n, block_rows):
            stop = min(n, start + block_rows)
            if sqrt_elision:
                R = sqrt_queried_ranks(D[start:stop], neighbors[start:stop], method=rank_method)
            else:
                R = queried_ranks(D[start:stop], neighbors[start:stop], method=rank_method, tie_policy='sklearn')
            for z, (k, lo, hi) in enumerate(zip(kvals, starts, ends)):
                penalties[z, axis] += np.maximum(R[:, lo:hi] - k, 0).sum(dtype=np.int64)
            del R
        del D
    return [_quality(k, n, *penalties[z]) for z, k in enumerate(kvals)]
