"""Joint exact trustworthiness/continuity, with explicit tie contracts.

No approximate nearest-neighbor search is used. See README.md for the distinction
between NumPy/sklearn's historical tie behavior and deterministic index ties.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Literal

import numpy as np
from numba import njit, prange
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class Quality:
    k: int
    trustworthiness: float
    continuity: float
    trustworthiness_penalty: int
    continuity_penalty: int


def _ks(ks: int | Iterable[int], n: int) -> tuple[int, ...]:
    vals = (ks,) if isinstance(ks, Integral) else tuple(ks)
    if not vals or any(not isinstance(k, Integral) or k < 1 or k >= n / 2 for k in vals):
        raise ValueError("Every k must be an integer satisfying 1 <= k < n / 2")
    return tuple(dict.fromkeys(int(k) for k in vals))


def _validate_pair(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError("X and Y must be 2D arrays with the same number of rows")
    if len(X) < 3:
        raise ValueError("Inputs must be finite and contain at least three samples")
    # Row chunks avoid an additional n*n boolean temporary for precomputed data.
    for Z in (X, Y):
        for start in range(0, len(Z), 256):
            if not np.isfinite(Z[start:start + 256]).all():
                raise ValueError("Inputs must be finite and contain at least three samples")
    return X, Y


def _quality(k, n, pt, pc):
    # Deliberately use sklearn's normalization and operation order.
    factor = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))
    return Quality(k, 1.0 - int(pt) * factor, 1.0 - int(pc) * factor, int(pt), int(pc))


def sklearn_reference(X, Y, ks=5):
    """Two actual sklearn calls per k, plus exact diagnostic integer penalties.

    The diagnostic penalty pass is intentionally separate; do not time this
    function as the sklearn baseline. Use ``sklearn_scores`` for timings.
    """
    X, Y = _validate_pair(X, Y)
    kvals = _ks(ks, len(X))
    diagnostics = joint_sklearn(X, Y, kvals, rank_method="full")
    for q in diagnostics:
        t = trustworthiness(X, Y, n_neighbors=q.k)
        c = trustworthiness(Y, X, n_neighbors=q.k)
        if t != q.trustworthiness or c != q.continuity:
            raise AssertionError((q, t, c))
    return diagnostics


def sklearn_scores(X, Y, ks=5):
    """Timing baseline: exactly two sklearn trustworthiness calls per k."""
    return [(trustworthiness(X, Y, n_neighbors=k),
             trustworthiness(Y, X, n_neighbors=k)) for k in _ks(ks, len(X))]


@njit(cache=True, parallel=True)
def _scan_ranks(D, Q):
    b, n = D.shape
    m = Q.shape[1]
    ranks = np.empty((b, m), dtype=np.int64)
    tied = np.zeros(b, dtype=np.bool_)
    for i in prange(b):
        any_ties = False
        for z in range(m):
            target = D[i, Q[i, z]]
            less, equal = 0, 0
            for j in range(n):
                d = D[i, j]
                less += d < target
                equal += d == target
            ranks[i, z] = less + 1
            any_ties |= equal > 1
        tied[i] = any_ties
    return ranks, tied


@njit(cache=True, parallel=True)
def _histogram_ranks(D, Q):
    """O(B N log M) rank queries; ties are flagged for exact policy repair."""
    b, n = D.shape
    m = Q.shape[1]
    ranks = np.empty((b, m), dtype=np.int64)
    tied = np.zeros(b, dtype=np.bool_)
    for i in prange(b):
        vals = np.empty(m, dtype=D.dtype)
        for z in range(m):
            vals[z] = D[i, Q[i, z]]
        order = np.argsort(vals)
        sorted_vals = vals[order]
        hist = np.zeros(m + 1, dtype=np.int64)
        eq = np.zeros(m, dtype=np.int64)
        for j in range(n):
            d = D[i, j]
            lo, hi = 0, m
            while lo < hi:
                mid = (lo + hi) // 2
                if d < sorted_vals[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            hist[lo] += 1
            if lo > 0 and sorted_vals[lo - 1] == d:
                eq[lo - 1] += 1
        count = 0
        for z in range(m):
            count += hist[z]
            ranks[i, order[z]] = count + 1
            if eq[z] > 1:
                tied[i] = True
    return ranks, tied


@njit(cache=True, parallel=True)
def _index_tie_repair(D, Q, ranks, tied):
    for i in prange(len(D)):
        if tied[i]:
            for z in range(Q.shape[1]):
                idx = Q[i, z]
                value = D[i, idx]
                for j in range(idx):
                    ranks[i, z] += D[i, j] == value


def queried_ranks(D, Q, *, method="histogram", tie_policy="index"):
    """Rank only Q, exactly relative to the supplied floating-point distances.

    D excludes self by an infinite diagonal. Q contains non-self indices.
    ``sklearn`` reproduces NumPy's default argsort on every tied query row.
    ``index`` orders equal distances by increasing sample index.
    """
    D, Q = np.asarray(D), np.asarray(Q)
    if D.ndim != 2 or Q.ndim != 2 or D.shape[0] != Q.shape[0] or D.shape[1] < 1:
        raise ValueError("D and Q must be 2D with matching rows and nonempty D columns")
    if Q.dtype.kind not in "iu":
        raise TypeError("Q must contain integer indices")
    if Q.size and (Q.min() < 0 or Q.max() >= D.shape[1]):
        raise ValueError("Q contains an out-of-bounds index")
    if D.dtype.kind not in "fiu" or np.isnan(D).any():
        raise ValueError("D must contain real distances without NaNs")
    if tie_policy not in {"index", "sklearn"}:
        raise ValueError("tie_policy must be index or sklearn")
    if method == "full":
        order = np.argsort(D, axis=1, kind="stable" if tie_policy == "index" else "quicksort")
        inv = np.empty(D.shape, dtype=np.int32 if D.shape[1] < 2**31 else np.int64)
        np.put_along_axis(inv, order, np.arange(1, D.shape[1] + 1), axis=1)
        return np.take_along_axis(inv, Q, axis=1)
    if method not in {"scan", "histogram"}:
        raise ValueError("rank method must be full, scan, or histogram")
    ranks, tied = (_scan_ranks if method == "scan" else _histogram_ranks)(D, Q)
    if tie_policy == "index":
        _index_tie_repair(D, Q, ranks, tied)
    elif tie_policy == "sklearn":
        # Reproduce the installed NumPy quicksort's actual permutation, rather
        # than inventing a stable ordering for equal distances.
        for i in np.flatnonzero(tied):
            order = np.argsort(D[i])
            inv = np.empty(D.shape[1], dtype=np.int64)
            inv[order] = np.arange(1, D.shape[1] + 1)
            ranks[i] = inv[Q[i]]
    else:
        raise ValueError("tie_policy must be index or sklearn")
    return ranks


@njit(cache=True)
def _penalties_for_ks(R, requested_ks):
    """All requested k in O(B K log M + M), instead of O(B sum(ks))."""
    order = np.argsort(requested_ks)
    ks = requested_ks[order]
    m = len(ks)
    intercept = np.zeros(m + 1, dtype=np.int64)
    slope = np.zeros(m + 1, dtype=np.int64)
    for z in range(R.shape[1]):
        first = np.searchsorted(ks, z + 1)
        if first == m:
            break
        for i in range(R.shape[0]):
            r = R[i, z]
            last = np.searchsorted(ks, r)
            if last > first:
                intercept[first] += r
                intercept[last] -= r
                slope[first] += 1
                slope[last] -= 1
    result = np.empty(m, dtype=np.int64)
    a, b = 0, 0
    for j in range(m):
        a += intercept[j]
        b += slope[j]
        result[order[j]] = a - ks[j] * b
    return result


@njit(cache=True, parallel=True)
def _index_queried_ranks(D, Q, own_neighbors, histogram=True, reuse_overlap=True):
    """Exact composite-key ranks; share already-known own top-K ranks."""
    b, n = D.shape
    k = Q.shape[1]
    ranks = np.empty((b, k), dtype=np.int64)
    for i in prange(b):
        own_order = np.argsort(own_neighbors[i])
        own_sorted = own_neighbors[i, own_order]
        missing_cols = np.empty(k, dtype=np.int64)
        m = 0
        for z in range(k):
            q = Q[i, z]
            known = False
            if reuse_overlap:
                pos = np.searchsorted(own_sorted, q)
                if pos < k and own_sorted[pos] == q:
                    ranks[i, z] = own_order[pos] + 1
                    known = True
            if not known:
                missing_cols[m] = z
                m += 1
        if m == 0:
            continue
        if not histogram:
            for u in range(m):
                z = missing_cols[u]
                q = Q[i, z]
                target = D[i, q]
                count = 0
                for j in range(n):
                    d = D[i, j]
                    count += d < target or (d == target and j < q)
                ranks[i, z] = count + 1
            continue
        # Stable value sort after index sort = lexicographic (distance,index).
        qids = np.empty(m, dtype=np.int64)
        for u in range(m):
            qids[u] = Q[i, missing_cols[u]]
        index_order = np.argsort(qids)
        vals = np.empty(m, dtype=D.dtype)
        for u in range(m):
            vals[u] = D[i, qids[index_order[u]]]
        value_order = np.argsort(vals, kind="mergesort")
        original_cols = np.empty(m, dtype=np.int64)
        sorted_vals = np.empty(m, dtype=D.dtype)
        sorted_ids = np.empty(m, dtype=np.int64)
        for u in range(m):
            v = value_order[u]
            original_cols[u] = missing_cols[index_order[v]]
            sorted_vals[u] = vals[v]
            sorted_ids[u] = qids[index_order[v]]
        hist = np.zeros(m + 1, dtype=np.int64)
        for j in range(n):
            d = D[i, j]
            lo, hi = 0, m
            while lo < hi:
                mid = (lo + hi) // 2
                if d < sorted_vals[mid] or (d == sorted_vals[mid] and j < sorted_ids[mid]):
                    hi = mid
                else:
                    lo = mid + 1
            hist[lo] += 1
        count = 0
        for u in range(m):
            count += hist[u]
            ranks[i, original_cols[u]] = count + 1
    return ranks


def _index_neighbors(D, k):
    """Partial select, sort k only, repair boundary ties by sample index."""
    n = D.shape[1]
    partition = np.argpartition(D, k, axis=1)
    Q = partition[:, :k].copy()
    selected = np.take_along_axis(D, Q, axis=1)
    threshold = selected.max(axis=1)
    next_value = np.take_along_axis(D, partition[:, k:k + 1], axis=1)[:, 0]
    tied_rows = np.flatnonzero(threshold == next_value)
    for i in tied_rows:
        strict = np.flatnonzero(D[i] < threshold[i])
        equal = np.flatnonzero(D[i] == threshold[i])
        Q[i] = np.concatenate((strict, equal[:k - len(strict)]))
    selected = np.take_along_axis(D, Q, axis=1)
    permutation = np.lexsort((Q, selected), axis=1)
    return np.take_along_axis(Q, permutation, axis=1)


def _metric_params(Z, metric, params):
    params = {} if params is None else dict(params)
    # Same data-derived defaults as sklearn.metrics.pairwise_distances(Z).
    if metric == "seuclidean" and "V" not in params:
        params["V"] = np.var(Z, axis=0, ddof=1)
    elif metric == "mahalanobis" and "VI" not in params:
        params["VI"] = np.linalg.inv(np.cov(Z.T)).T
    return params


def _distance_block(Z, start, stop, metric, *, full=False, metric_params=None):
    if metric == "precomputed":
        if Z.shape[0] != Z.shape[1] or (Z[start:stop] < 0).any():
            raise ValueError("Precomputed distances must be square and non-negative")
        if Z.dtype.kind in "iu" and (Z[start:stop] > 2**53).any():
            raise ValueError("Precomputed integer distances above 2**53 are not safely representable")
        D = Z[start:stop].copy()
    else:
        kwds = {} if metric_params is None else metric_params
        D = (pairwise_distances(Z, metric=metric, **kwds) if full else
             pairwise_distances(Z[start:stop], Z, metric=metric, **kwds))
    if not np.isfinite(D).all():
        raise ValueError("The metric produced non-finite distances; check degenerate input or overflow")
    # pairwise_distances may return integer input unchanged for precomputed.
    if D.dtype.kind != "f":
        D = D.astype(np.float64)
    D[np.arange(stop - start), np.arange(start, stop)] = np.inf
    return D


def joint_fullrank(X, Y, ks=5, *, block_size=256,
                   metric_x="euclidean", metric_y="euclidean",
                   metric_params_x=None, metric_params_y=None):
    """Joint full-sort reference that reuses each sort for neighbors and ranks.

    Distances and ranks are row-blocked. Index tie semantics match joint_exact.
    Unlike two sklearn calls, no separate nearest-neighbor distance pass occurs.
    """
    X, Y = _validate_pair(X, Y)
    n, kvals = len(X), _ks(ks, len(X))
    kmax = max(kvals)
    params_x = _metric_params(X, metric_x, metric_params_x)
    params_y = _metric_params(Y, metric_y, metric_params_y)
    full = block_size is None
    if not full and (not isinstance(block_size, Integral) or block_size < 1):
        raise ValueError("block_size must be a positive integer or None")
    bsize = n if full else int(block_size)
    penalties = np.zeros((len(kvals), 2), dtype=np.int64)
    rank_dtype = np.int32 if n < 2**31 else np.int64
    for start in range(0, n, bsize):
        stop = min(n, start + bsize)
        DX = _distance_block(X, start, stop, metric_x, full=full, metric_params=params_x)
        IX = np.argsort(DX, axis=1, kind="stable")
        del DX
        NX = IX[:, :kmax].copy()
        RX = np.empty(IX.shape, dtype=rank_dtype)
        np.put_along_axis(RX, IX, np.arange(1, n + 1), axis=1)
        del IX
        DY = _distance_block(Y, start, stop, metric_y, full=full, metric_params=params_y)
        IY = np.argsort(DY, axis=1, kind="stable")
        del DY
        NY = IY[:, :kmax]
        query_RX = np.take_along_axis(RX, NY, axis=1)
        del RX
        RY = np.empty(IY.shape, dtype=rank_dtype)
        np.put_along_axis(RY, IY, np.arange(1, n + 1), axis=1)
        query_RY = np.take_along_axis(RY, NX, axis=1)
        del RY, IY, NY
        penalties[:, 0] += _penalties_for_ks(query_RX, np.asarray(kvals, dtype=np.int64))
        penalties[:, 1] += _penalties_for_ks(query_RY, np.asarray(kvals, dtype=np.int64))
        del query_RX, query_RY, NX
    return [_quality(k, n, *penalties[z]) for z, k in enumerate(kvals)]


def joint_exact(X, Y, ks=5, *, rank_method="histogram", block_size=256,
                metric_x="euclidean", metric_y="euclidean", reuse_overlap=True,
                metric_params_x=None, metric_params_y=None):
    """Exact joint metric with deterministic (distance, sample-index) ties.

    Computes each distance block once per space, selects only max(ks) neighbors,
    and ranks only those neighbors in the opposite space. Multiple k share all
    geometric work. Peak working storage is O(block_size*n + n*max(ks)); the
    implementation reduces scores blockwise so the n*max(ks) term is not stored.

    ``block_size=None`` uses sklearn's dense self-distance calculation. For
    feature arrays, blockwise Gram-matrix rounding can differ near exact ties;
    supplied precomputed matrices provide an engine-independent distance input.
    """
    if rank_method == "full":
        return joint_fullrank(X, Y, ks, block_size=block_size,
                              metric_x=metric_x, metric_y=metric_y,
                              metric_params_x=metric_params_x, metric_params_y=metric_params_y)
    X, Y = _validate_pair(X, Y)
    n, kvals = len(X), _ks(ks, len(X))
    kmax = max(kvals)
    params_x = _metric_params(X, metric_x, metric_params_x)
    params_y = _metric_params(Y, metric_y, metric_params_y)
    full = block_size is None
    if not full and (not isinstance(block_size, Integral) or block_size < 1):
        raise ValueError("block_size must be a positive integer or None")
    bsize = n if full else int(block_size)
    penalties = np.zeros((len(kvals), 2), dtype=np.int64)
    for start in range(0, n, bsize):
        stop = min(n, start + bsize)
        DX = _distance_block(X, start, stop, metric_x, full=full, metric_params=params_x)
        DY = _distance_block(Y, start, stop, metric_y, full=full, metric_params=params_y)
        NX, NY = _index_neighbors(DX, kmax), _index_neighbors(DY, kmax)
        if rank_method not in {"scan", "histogram"}:
            raise ValueError("rank method must be full, scan, or histogram")
        RX = _index_queried_ranks(DX, NY, NX, rank_method == "histogram", reuse_overlap)
        RY = _index_queried_ranks(DY, NX, NY, rank_method == "histogram", reuse_overlap)
        penalties[:, 0] += _penalties_for_ks(RX, np.asarray(kvals, dtype=np.int64))
        penalties[:, 1] += _penalties_for_ks(RY, np.asarray(kvals, dtype=np.int64))
        del DX, DY, NX, NY, RX, RY
    return [_quality(k, n, *penalties[z]) for z, k in enumerate(kvals)]


def joint_sklearn(X, Y, ks=5, *, rank_method="histogram", block_size=None):
    """Joint scores preserving sklearn 1.8/installed NumPy tie behavior.

    NearestNeighbors is invoked independently for each k, because sklearn's
    selected tied neighborhood is not guaranteed to be a prefix of max-k.
    Full distance matrices (default) reproduce sklearn's numerical distances.
    Setting block_size bounds distance memory, but changes the distance-kernel
    call shape and therefore cannot promise bit-exact scores for near ties.
    """
    X, Y = _validate_pair(X, Y)
    n, kvals = len(X), _ks(ks, len(X))
    NX = np.concatenate([NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False) for k in kvals], axis=1)
    NY = np.concatenate([NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(return_distance=False) for k in kvals], axis=1)
    ends = np.cumsum(kvals)
    starts = np.r_[0, ends[:-1]]
    full = block_size is None
    if not full and (not isinstance(block_size, Integral) or block_size < 1):
        raise ValueError("block_size must be a positive integer or None")
    bsize = n if full else int(block_size)
    penalties = np.zeros((len(kvals), 2), dtype=np.int64)
    # Process spaces sequentially: never retain both dense distance matrices.
    for axis, (Z, neighbors) in enumerate(((X, NY), (Y, NX))):
        for start in range(0, n, bsize):
            stop = min(n, start + bsize)
            D = _distance_block(Z, start, stop, "euclidean", full=full)
            R = queried_ranks(D, neighbors[start:stop], method=rank_method, tie_policy="sklearn")
            for z, (k, lo, hi) in enumerate(zip(kvals, starts, ends)):
                penalties[z, axis] += np.maximum(R[:, lo:hi] - k, 0).sum(dtype=np.int64)
            del D, R
    return [_quality(k, n, *penalties[z]) for z, k in enumerate(kvals)]


def trustworthiness_continuity(X, Y, n_neighbors=5, *, compatibility="sklearn",
                              rank_method="histogram", block_size=None, **kwargs):
    """Recommended entry point; strict sklearn compatibility is the default.

    A scalar n_neighbors returns one Quality; an iterable returns a list.
    ``compatibility='index'`` selects shared-neighborhood deterministic exact
    metrics; pass block_size (e.g. 256) for memory-bounded execution.
    """
    if compatibility == "sklearn":
        if kwargs:
            raise TypeError("Strict sklearn compatibility uses Euclidean metrics in both spaces")
        result = joint_sklearn(X, Y, n_neighbors, rank_method=rank_method, block_size=block_size)
    elif compatibility == "index":
        result = joint_exact(X, Y, n_neighbors, rank_method=rank_method, block_size=block_size, **kwargs)
    else:
        raise ValueError("compatibility must be sklearn or index")
    return result[0] if isinstance(n_neighbors, Integral) else result
