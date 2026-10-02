"""Exact joint trustworthiness/continuity using standard binary dependencies.

This module imports only Python's standard library, NumPy and scikit-learn.
It does not import the laboratory's JIT/native implementations. It preserves
scikit-learn's full Euclidean distance calculation, input floating dtype,
neighbor selection separately for each k, and NumPy's actual default argsort
permutation on tied rows. See README.md for version and memory qualifications.
"""
from __future__ import annotations

from numbers import Integral
from typing import NamedTuple

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from ._neighbor_queries import neighbor_queries


class Quality(NamedTuple):
    k: int
    trustworthiness: float
    continuity: float
    trustworthiness_penalty: int
    continuity_penalty: int


def _ks(ks, n):
    vals = (ks,) if isinstance(ks, Integral) else tuple(ks)
    if not vals or any(not isinstance(k, Integral) or k < 1 or k >= n / 2 for k in vals):
        raise ValueError("Every k must be an integer satisfying 1 <= k < n / 2")
    return tuple(dict.fromkeys(int(k) for k in vals))


def _validate_pair(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError("X and Y must be 2D arrays with the same number of rows")
    if len(X) < 3 or X.shape[1] < 1 or Y.shape[1] < 1:
        raise ValueError("Inputs must contain at least three samples and one feature")
    for Z in (X, Y):
        if Z.dtype.kind not in "fiu":
            raise TypeError("Inputs must contain real numbers")
        for start in range(0, len(Z), 256):
            if not np.isfinite(Z[start:start + 256]).all():
                raise ValueError("Inputs must be finite")
    return X, Y


def _quality(k, n, pt, pc):
    # Preserve sklearn's normalization and operation order exactly.
    pt, pc = int(pt), int(pc)
    factor = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))
    return Quality(k, 1.0 - pt * factor, 1.0 - pc * factor, pt, pc)


def _validate_options(method, block_size, max_scratch_bytes):
    if method not in ("full", "broadcast", "searchsorted", "sortsearch"):
        raise ValueError("rank_method must be full, broadcast, searchsorted, or sortsearch")
    if not isinstance(block_size, Integral) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not isinstance(max_scratch_bytes, Integral) or max_scratch_bytes < 1:
        raise ValueError("max_scratch_bytes must be a positive integer")


def _full_ranks(D, Q):
    order = np.argsort(D, axis=1)
    inv_dtype = np.int32 if D.shape[1] <= np.iinfo(np.int32).max else np.int64
    inverse = np.empty(D.shape, dtype=inv_dtype)
    np.put_along_axis(inverse, order, np.arange(1, D.shape[1] + 1), axis=1)
    return np.take_along_axis(inverse, Q, axis=1)


def _repair_ties(D, Q, ranks, tied):
    # The fallback sorts the original rooted dtype and uses the installed
    # unstable argsort, not an invented stable/index tie policy.
    for i in np.flatnonzero(tied):
        order = np.argsort(D[i])
        inverse = np.empty(D.shape[1], dtype=np.int64)
        inverse[order] = np.arange(1, D.shape[1] + 1)
        ranks[i] = inverse[Q[i]]


def _searchsorted_ranks(D, Q):
    """O(B*N*log K) NumPy histogram, with exact tied-query detection."""
    ranks = np.empty(Q.shape, dtype=np.int64)
    tied_rows = 0
    for i in range(len(D)):
        # Repeated queried indices/values are not automatically distance ties.
        thresholds, inverse = np.unique(D[i, Q[i]], return_inverse=True)
        bins = np.searchsorted(thresholds, D[i], side="right")
        counts = np.bincount(bins, minlength=len(thresholds) + 1)
        less = np.cumsum(counts[:-1], dtype=np.int64)
        # bin z contains values >= threshold[z-1], < threshold[z].
        # Count exactly equal values independently before using rank=less+1.
        previous = thresholds[np.maximum(bins - 1, 0)]
        equal_mask = (bins > 0) & (D[i] == previous)
        equal_counts = np.bincount(bins[equal_mask], minlength=len(thresholds) + 1)[1:]
        if (equal_counts > 1).any():
            order = np.argsort(D[i])
            full_inverse = np.empty(D.shape[1], dtype=np.int64)
            full_inverse[order] = np.arange(1, D.shape[1] + 1)
            ranks[i] = full_inverse[Q[i]]
            tied_rows += 1
        else:
            ranks[i] = less[inverse] + 1
    return ranks, tied_rows


def _sortsearch_ranks(D, Q):
    """Sort numeric values, then binary-search query ranks; repair all ties."""
    ordered_values = np.sort(D, axis=1)
    ranks = np.empty(Q.shape, dtype=np.int64)
    tied_rows = 0
    n = D.shape[1]
    for i in range(len(D)):
        targets = D[i, Q[i]]
        lower = np.searchsorted(ordered_values[i], targets, side="left")
        # All targets occur in D. An equal next value proves a repeated query
        # distance without a second binary search. The last entry has no next.
        tied = (lower + 1 < n) & (ordered_values[i, np.minimum(lower + 1, n - 1)] == targets)
        if tied.any():
            order = np.argsort(D[i])
            inverse = np.empty(n, dtype=np.int64)
            inverse[order] = np.arange(1, n + 1)
            ranks[i] = inverse[Q[i]]
            tied_rows += 1
        else:
            ranks[i] = lower + 1
    return ranks, tied_rows


def queried_ranks_numpy(D, Q, *, method="broadcast", block_size=32,
                        max_scratch_bytes=32 * 2**20, return_stats=False):
    """Return 1-based ranks at Q, matching np.argsort(D, axis=1) exactly.

    D must already contain rooted distances with excluded self (+inf), if
    applicable. It can be rectangular and is not modified. Repeated queries
    and arbitrary ties are supported. block_size splits only rank work: no
    pairwise-distance calculation is changed. The broadcast bool scratch is
    capped at max_scratch_bytes, including when one full K*N row exceeds it;
    query columns are then chunked too. Full matrices and result arrays are
    outside this scratch cap. A tied query triggers a full-row sort.
    """
    _validate_options(method, block_size, max_scratch_bytes)
    D, Q = np.asarray(D), np.asarray(Q)
    if D.ndim != 2 or Q.ndim != 2 or len(D) != len(Q) or D.shape[1] < 1:
        raise ValueError("D and Q must be 2D with matching rows and nonempty D columns")
    if D.dtype.kind not in "fiu":
        raise TypeError("D must contain real distances")
    for start in range(0, len(D), 256):
        if np.isnan(D[start:start + 256]).any():
            raise ValueError("D must not contain NaNs")
    if Q.dtype.kind not in "iu":
        raise TypeError("Q must contain integer indices")
    if Q.size and (Q.min() < 0 or Q.max() >= D.shape[1]):
        raise ValueError("Q contains an out-of-bounds index")
    rows, n, k = len(D), D.shape[1], Q.shape[1]
    ranks = np.empty(Q.shape, dtype=np.int64)
    stats = {"query_count": int(Q.size), "fallback_rows": 0,
             "total_rows": rows, "max_bool_scratch_bytes": 0,
             "row_block_size": min(int(block_size), rows), "query_block_size": k}
    if not Q.size:
        return (ranks, stats) if return_stats else ranks
    bsize = int(block_size)
    qsize = k
    if method == "broadcast":
        # One query requires a length-N boolean row. Reject impossibly small
        # caps rather than quietly exceeding the advertised memory limit.
        if max_scratch_bytes < n:
            raise ValueError("max_scratch_bytes must be at least D.shape[1] for broadcast")
        qsize = min(k, int(max_scratch_bytes) // n)
        bsize = min(bsize, max(1, int(max_scratch_bytes) // (n * qsize)))
        scratch = np.empty((min(bsize, rows), qsize, n), dtype=np.bool_)
        stats["max_bool_scratch_bytes"] = int(scratch.nbytes)
    stats["row_block_size"] = min(bsize, rows)
    stats["query_block_size"] = qsize
    for start in range(0, rows, bsize):
        stop = min(rows, start + bsize)
        if method == "full":
            ranks[start:stop] = _full_ranks(D[start:stop], Q[start:stop])
        elif method in ("searchsorted", "sortsearch"):
            ranker = _searchsorted_ranks if method == "searchsorted" else _sortsearch_ranks
            R, fallback = ranker(D[start:stop], Q[start:stop])
            ranks[start:stop] = R
            stats["fallback_rows"] += fallback
        else:
            # Track distinct fallback rows even if query columns are chunked.
            block_tied = np.zeros(stop - start, dtype=np.bool_)
            for lo in range(0, k, qsize):
                hi = min(k, lo + qsize)
                q = Q[start:stop, lo:hi]
                targets = np.take_along_axis(D[start:stop], q, axis=1)
                work = scratch[:stop - start, :hi - lo]
                np.less(D[start:stop, None, :], targets[:, :, None], out=work)
                ranks[start:stop, lo:hi] = np.count_nonzero(work, axis=2) + 1
                np.equal(D[start:stop, None, :], targets[:, :, None], out=work)
                block_tied |= (np.count_nonzero(work, axis=2) > 1).any(axis=1)
            _repair_ties(D[start:stop], Q[start:stop], ranks[start:stop], block_tied)
            stats["fallback_rows"] += int(np.count_nonzero(block_tied))
    return (ranks, stats) if return_stats else ranks


def joint_sklearn_numpy(X, Y, ks=5, *, rank_method="broadcast", block_size=32,
                        max_scratch_bytes=32 * 2**20, return_stats=False):
    """Exact Euclidean joint metrics; always returns a list of Quality records.

    Distances use one full rooted sklearn pairwise_distances call per space,
    sequentially. block_size affects only rank work, so this retains the full
    matrix numerical contract; peak distance storage is still O(N**2).
    Every requested k uses sklearn's own neighbor search in each space. No
    largest-k prefix assumption is made, even when neighbor ties are present.
    Multiple k share distance and rank work. Duplicate k values are deduplicated
    in request order. No JIT, custom extension or build step is required.
    """
    _validate_options(rank_method, block_size, max_scratch_bytes)
    X, Y = _validate_pair(X, Y)
    n, kvals = len(X), _ks(ks, len(X))
    NX = neighbor_queries(X, kvals)
    NY = neighbor_queries(Y, kvals)
    ends = np.cumsum(kvals)
    starts = np.r_[0, ends[:-1]]
    penalties = np.zeros((len(kvals), 2), dtype=np.int64)
    stats = {"rank_method": rank_method, "spaces": []}
    for axis, (Z, neighbors) in enumerate(((X, NY), (Y, NX))):
        D = pairwise_distances(Z, metric="euclidean")
        for start in range(0, n, 256):
            if not np.isfinite(D[start:start + 256]).all():
                raise ValueError("Euclidean distances overflowed; rescale the input")
        np.fill_diagonal(D, np.inf)
        R, rank_stats = queried_ranks_numpy(D, neighbors, method=rank_method,
            block_size=block_size, max_scratch_bytes=max_scratch_bytes, return_stats=True)
        rank_stats["distance_dtype"] = str(D.dtype)
        rank_stats["distance_bytes"] = int(D.nbytes)
        stats["spaces"].append(rank_stats)
        for z, (k, lo, hi) in enumerate(zip(kvals, starts, ends)):
            penalties[z, axis] = np.maximum(R[:, lo:hi] - k, 0).sum(dtype=np.int64)
        del D, R
    results = [_quality(k, n, *penalties[z]) for z, k in enumerate(kvals)]
    return (results, stats) if return_stats else results


def trustworthiness_continuity(X, Y, n_neighbors=5, **kwargs):
    """Return one Quality for scalar k, or a list for iterable k.

    With return_stats=True, return (Quality-or-list, stats).
    """
    result = joint_sklearn_numpy(X, Y, n_neighbors, **kwargs)
    if kwargs.get("return_stats", False):
        qualities, stats = result
        return (qualities[0] if isinstance(n_neighbors, Integral) else qualities), stats
    return result[0] if isinstance(n_neighbors, Integral) else result
