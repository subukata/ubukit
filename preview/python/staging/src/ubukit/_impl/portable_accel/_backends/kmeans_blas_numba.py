"""Portable blocked-BLAS Lloyd with a fused Numba assignment/mean pass.

There are no target-specific intrinsics or build steps.  Numba calls the BLAS
shipped with SciPy.  Its threads stay at one because Numba distributes disjoint
row partitions.  A conservative expanded-distance ambiguity guard repairs
near-ties by direct squared differences.  This is an engineering guard, not a
formal interval proof for all floating-point inputs.
"""
from __future__ import annotations

from contextlib import contextmanager
import operator
import numpy as np
import scipy.linalg  # Load the exact BLAS library used by Numba before capture.
from numba import njit, prange, get_num_threads, set_num_threads
from threadpoolctl import ThreadpoolController

_controller = ThreadpoolController()
_EPS = np.finfo(np.float64).eps


@njit(cache=True, inline='always')
def _direct(X, centers, row):
    best = 0
    best_distance = np.inf
    for c in range(centers.shape[0]):
        distance = 0.0
        for j in range(X.shape[1]):
            delta = X[row, j] - centers[c, j]
            distance += delta * delta
        if distance < best_distance:
            best_distance = distance
            best = c
    return best


@njit(cache=True, parallel=True)
def _lloyd(X, init, xnorm, max_iter, blocks, block_rows):
    n, d = X.shape
    block_rows = min(block_rows, (n + blocks - 1) // blocks)
    k = init.shape[0]
    centers = init.copy()
    labels = np.full(n, -1, dtype=np.int64)
    sums = np.empty((blocks, k, d), dtype=np.float64)
    counts = np.empty((blocks, k + 8), dtype=np.int64)
    stats = np.empty((blocks, 8), dtype=np.int64)
    scratch = np.empty((blocks, block_rows, k), dtype=np.float64)
    cnorm = np.empty(k, dtype=np.float64)
    total_fallbacks = 0
    iteration = 0
    for iteration in range(1, max_iter + 1):
        max_cnorm = 0.0
        for c in range(k):
            value = 0.0
            for j in range(d):
                value += centers[c, j] * centers[c, j]
            cnorm[c] = 0.5 * value
            max_cnorm = max(max_cnorm, value)
        centers_t = centers.T
        for b in prange(blocks):
            sums[b].fill(0.0)
            counts[b].fill(0)
            changed = 0
            fallbacks = 0
            first = n * b // blocks
            end = n * (b + 1) // blocks
            for start in range(first, end, block_rows):
                stop = min(start + block_rows, end)
                rows = stop - start
                # C-contiguous first-axis slices do not copy X or the workspace.
                scores = np.ascontiguousarray(scratch[b, :rows, :])
                np.dot(X[start:stop], centers_t, scores)
                for r in range(rows):
                    i = start + r
                    best = -np.inf
                    second = -np.inf
                    label = 0
                    for c in range(k):
                        score = scores[r, c] - cnorm[c]
                        if score > best:
                            second = best
                            best = score
                            label = c
                        elif score > second:
                            second = score
                    bound = 64.0 * _EPS * (d + 1) * (xnorm[i] + max_cnorm + 1.0)
                    if not np.isfinite(best) or not np.isfinite(bound) or best - second <= bound:
                        label = _direct(X, centers, i)
                        fallbacks += 1
                    if labels[i] != label:
                        changed += 1
                        labels[i] = label
                    counts[b, label] += 1
                    for j in range(d):
                        sums[b, label, j] += X[i, j]
            stats[b, 0] = changed
            stats[b, 1] = fallbacks
        for c in range(k):
            count = 0
            for b in range(blocks):
                count += counts[b, c]
            if count:
                # Interchange the independent feature loop for contiguous SIMD
                # loads while preserving increasing-block addition per feature.
                for j in range(d):
                    centers[c, j] = sums[0, c, j]
                for b in range(1, blocks):
                    for j in range(d):
                        centers[c, j] += sums[b, c, j]
                for j in range(d):
                    centers[c, j] /= count
        changed = 0
        for b in range(blocks):
            changed += stats[b, 0]
            total_fallbacks += stats[b, 1]
        if changed == 0:
            break
    return centers, labels, iteration, total_fallbacks


def _validate(X, init, max_iter, block_rows):
    X, init = np.asarray(X), np.asarray(init)
    if X.dtype != np.float64 or init.dtype != np.float64:
        raise TypeError('X and init must have dtype float64')
    if X.ndim != 2 or init.ndim != 2 or X.shape[1] != init.shape[1]:
        raise ValueError('matching two-dimensional arrays required')
    if min(X.shape) < 1 or init.shape[0] < 1:
        raise ValueError('nonempty arrays required')
    if not X.flags.c_contiguous or not init.flags.c_contiguous:
        raise ValueError('C-contiguous arrays required')
    if not np.isfinite(X).all() or not np.isfinite(init).all():
        raise ValueError('finite arrays required')
    max_iter, block_rows = operator.index(max_iter), operator.index(block_rows)
    if min(max_iter, block_rows) < 1:
        raise ValueError('max_iter and block_rows must be positive')
    return X, init, max_iter, block_rows


@contextmanager
def execution_threads(threads, blas_threads=1):
    old = get_num_threads()
    active = old if threads is None else operator.index(threads)
    if active < 1:
        raise ValueError('threads must be positive')
    set_num_threads(active)
    try:
        with _controller.limit(limits=active if blas_threads is None else blas_threads, user_api='blas'):
            yield active
    finally:
        set_num_threads(old)


def run(X, init, max_iter=20, threads=None, block_rows=256):
    """Run Lloyd with per-call validation, X norms and scratch included."""
    X, init, max_iter, block_rows = _validate(X, init, max_iter, block_rows)
    xnorm = np.einsum('nd,nd->n', X, X)
    with execution_threads(threads) as active:
        result = _lloyd(X, init, xnorm, max_iter, min(active, len(X)), block_rows)
    return {'centers': result[0], 'labels': result[1], 'n_iter': result[2],
            'guard_fallbacks': result[3]}
