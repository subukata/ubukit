"""Numba kernels for ``engine="numba"``: the loops that NumPy cannot vectorize.

Each kernel transcribes, loop for loop, the NumPy code it replaces, which
stays the reference; tests/test_numba.py requires the same results. This
module is imported only when a caller asks for ``engine="numba"`` and caches
nothing on disk, so each kernel compiles on its first call in a process.
Parallel kernels split the work by rows whose results are independent and
leave every sum over rows to NumPy, so the results do not depend on the
number of threads.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=False)
def som_epoch(X, W, R, order, start, steps, s0, s1, lr, lr_end):
    """One epoch of ``som``: update the prototypes W by the samples in ``order``."""
    d = X.shape[1]
    k, q = R.shape
    W = W.copy()
    diff = np.empty((k, d))
    for j in range(order.shape[0]):
        i = order[j]
        f = (start + j) / (steps - 1) if steps > 1 else 0.0
        s, eta = s0 ** (1.0 - f) * s1**f, lr ** (1.0 - f) * lr_end**f
        bmu, best = 0, np.inf
        for u in range(k):
            dist = 0.0
            for c in range(d):
                diff[u, c] = X[i, c] - W[u, c]
                dist += diff[u, c] * diff[u, c]
            if dist < best:
                bmu, best = u, dist
        for u in range(k):
            g = 0.0
            for c in range(q):
                t = (R[u, c] - R[bmu, c]) / s
                g += t * t
            h = eta * math.exp(-0.5 * g)
            for c in range(d):
                W[u, c] += h * diff[u, c]
    return W


@njit(cache=False)
def _distances(A, i):
    """Squared distances from row i to every row of A, by direct differences; inf at i."""
    n, d = A.shape
    out = np.empty(n)
    for j in range(n):
        s = 0.0
        for f in range(d):
            t = A[i, f] - A[j, f]
            s += t * t
        out[j] = s
    out[i] = np.inf
    return out


@njit(cache=False)
def _row(A, stored, i):
    """Squared distances from row i to every row of A (A's row i when stored), inf at i."""
    if stored:
        return A[i].copy()
    return _distances(A, i)


@njit(parallel=True, cache=False)
def trustworthiness_penalties(A, a_stored, B, b_stored, ks):
    """Per point i and each k in ks, the sum over its k nearest neighbors j in B of
    max(rank of j in A - k, 0); A and B are points, or their distances when stored.

    Neighbors and ranks follow the (distance, index) order, as in ``trustworthiness``.
    """
    n = A.shape[0]
    top = ks.max()
    penalty = np.zeros((n, ks.shape[0]), np.int64)
    for i in prange(n):
        da, db = _row(A, a_stored, i), _row(B, b_stored, i)
        # The top smallest of db, kept sorted by insertion: the k nearest for
        # each k are the first k.
        nearest = np.empty(top, np.int64)
        size = 0
        for j in range(n):
            if size == top and db[j] >= db[nearest[top - 1]]:
                continue
            p = min(size, top - 1)
            while p > 0 and db[nearest[p - 1]] > db[j]:
                nearest[p] = nearest[p - 1]
                p -= 1
            nearest[p] = j
            size = min(size + 1, top)
        for r in range(top):
            j = nearest[r]
            rank = 1
            for m in range(n):
                if da[m] < da[j] or (da[m] == da[j] and m < j):
                    rank += 1
            for q in range(ks.shape[0]):
                if r < ks[q]:
                    penalty[i, q] += max(rank - ks[q], 0)
    return penalty


@njit(parallel=True, cache=False)
def expected_mi_terms(n, av, ac, bv, bc, lf):
    """The terms of ``_expected_mutual_information``, one per distinct size in ``av``.

    ``lf[t]`` is log t! for t = 0..n.
    """
    terms = np.zeros(av.shape[0])
    for a in prange(av.shape[0]):
        x = av[a]
        for b in range(bv.shape[0]):
            y = bv[b]
            mean, width = x * y / n, math.sqrt(35 * min(x, y))
            low = max(1, x + y - n, math.ceil(mean - width))
            high = min(x, y, math.floor(mean + width))
            base = lf[x] + lf[y] + lf[n - x] + lf[n - y] - lf[n]
            total = 0.0
            for nij in range(low, high + 1):
                log_p = base - lf[nij] - lf[x - nij] - lf[y - nij] - lf[n - x - y + nij]
                total += nij / n * math.log(n * nij / (x * y)) * math.exp(log_p)
            terms[a] += ac[a] * bc[b] * total
    return terms
