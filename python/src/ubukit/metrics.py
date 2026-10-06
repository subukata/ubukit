"""External agreement (ARI, AMI) and neighborhood preservation (trustworthiness, continuity)."""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.special import gammaln

from ._core import Engine, as_matrix, check_int, numba_kernels

type Average = Literal["arithmetic", "geometric", "min", "max"]

_AVERAGES: dict[str, Callable[[float, float], float]] = {
    "arithmetic": lambda a, b: (a + b) / 2,
    "geometric": lambda a, b: np.sqrt(a * b),
    "min": min,
    "max": max,
}


def ari(labels_true: ArrayLike, labels_pred: ArrayLike) -> float:
    """Adjusted Rand index (Hubert & Arabie, 1985)."""
    n, a, b, _, _, nij = _contingency(labels_true, labels_pred)
    # Exact integer pair counts: same/same, diff/same, same/diff, diff/diff.
    sa, sb, s = int(a @ a), int(b @ b), int(nij @ nij)
    tp, fp, fn = s - n, sb - s, sa - s
    tn = n * n - sa - sb + s
    if fp == 0 and fn == 0:
        return 1.0
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def ami(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    *,
    average: Average = "arithmetic",
    engine: Engine = "numpy",
) -> float:
    """Adjusted mutual information (Vinh, Epps & Bailey, 2010).

    ``average`` normalizes by the arithmetic, geometric, min or max of the
    two label entropies. ``engine="numba"`` computes the expected mutual
    information, the costly part, in parallel compiled loops.
    """
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {sorted(_AVERAGES)}")
    kernels = numba_kernels(engine)
    n, a, b, rows, cols, nij = _contingency(labels_true, labels_pred)
    if len(a) == len(b) <= 1:
        return 1.0
    mi = float(np.sum(nij / n * (np.log(nij) + np.log(n) - np.log(a[rows]) - np.log(b[cols]))))
    emi = _expected_mutual_information(n, a, b, kernels)
    denominator = _AVERAGES[average](_entropy(a, n), _entropy(b, n)) - emi
    eps = np.finfo(np.float64).eps
    denominator = min(denominator, -eps) if denominator < 0 else max(denominator, eps)
    return (mi - emi) / denominator


def trustworthiness(X: ArrayLike, Y: ArrayLike, k: int = 5, *, engine: Engine = "numpy") -> float:
    """Trustworthiness of embedding Y of X (Venna & Kaski, 2001).

    Penalizes points that are among the k nearest neighbors in Y but not in X,
    by their rank in X. Distances are computed from direct differences, so
    exact ties (e.g. SOM grid coordinates) stay exact and are ordered by index.
    ``engine="numba"`` computes the same penalties in parallel compiled loops.
    """
    kernels = numba_kernels(engine)
    X, Y = as_matrix(X), as_matrix(Y, "Y")
    n = len(X)
    if len(Y) != n:
        raise ValueError("X and Y must have the same number of rows")
    k = check_int(k, "k", 1)
    if not 2 * k < n:
        raise ValueError("k must satisfy 1 <= k < n / 2")
    if kernels:
        penalty = int(kernels.trustworthiness_penalties(X, Y, k).sum())
    else:
        penalty = _penalty(X, Y, k)
    return 1.0 - 2.0 * penalty / (n * k * (2.0 * n - 3.0 * k - 1.0))


def continuity(X: ArrayLike, Y: ArrayLike, k: int = 5, *, engine: Engine = "numpy") -> float:
    """Continuity of embedding Y of X: trustworthiness with the roles swapped."""
    return trustworthiness(Y, X, k, engine=engine)


def _penalty(X: np.ndarray, Y: np.ndarray, k: int) -> int:
    """Sum over points of max(rank in X - k, 0) for their k nearest neighbors in Y."""
    n = len(X)
    index = np.arange(n)
    block = max(1, (1 << 22) // (n * (k + 2)))
    penalty = 0
    for start in range(0, n, block):
        rows = index[start : start + block]
        dx = cdist(X[rows], X, "sqeuclidean")
        dy = cdist(Y[rows], Y, "sqeuclidean")
        dx[np.arange(len(rows)), rows] = np.inf
        dy[np.arange(len(rows)), rows] = np.inf
        neighbors = _nearest(dy, k)
        target = np.take_along_axis(dx, neighbors, axis=1)[:, :, None]
        # 1-based rank in X under the (distance, index) order.
        rank = 1 + (dx[:, None, :] < target).sum(axis=2)
        rank += ((dx[:, None, :] == target) & (index < neighbors[:, :, None])).sum(axis=2)
        penalty += int(np.maximum(rank - k, 0).sum())
    return penalty


def _nearest(d: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the k smallest entries per row, ties broken by index (unordered)."""
    kth = np.partition(d, k - 1, axis=1)[:, k - 1 : k]
    below = d < kth
    tied = d == kth
    room = k - below.sum(axis=1, keepdims=True)
    chosen = below | (tied & (np.cumsum(tied, axis=1) <= room))
    return np.nonzero(chosen)[1].reshape(len(d), k)


def _contingency(labels_true, labels_pred):
    """Return n, row sums, column sums and the nonzero cells (row, col, count)."""
    t, p = np.asarray(labels_true), np.asarray(labels_pred)
    if t.ndim != 1 or p.ndim != 1 or len(t) != len(p):
        raise ValueError("labels must be 1-D arrays of equal length")
    u = np.unique(t, return_inverse=True)[1]
    v = np.unique(p, return_inverse=True)[1]
    a, b = np.bincount(u), np.bincount(v)
    kb = max(len(b), 1)
    cells, nij = np.unique(u * kb + v, return_counts=True)
    return len(t), a, b, cells // kb, cells % kb, nij


def _entropy(counts: np.ndarray, n: int) -> float:
    p = counts / n
    return float(-np.sum(p * np.log(p)))


def _expected_mutual_information(
    n: int, a: np.ndarray, b: np.ndarray, kernels: ModuleType | None = None
) -> float:
    """E[MI] under the hypergeometric model, grouped by distinct marginal sizes.

    Each sum runs over n_ij within sqrt(35 min(x, y)) of its mean x y / n;
    by Hoeffding's inequality the hypergeometric mass outside is below
    2 exp(-70), so the result equals the full sum to rounding.
    """
    av, ac = np.unique(a, return_counts=True)
    bv, bc = np.unique(b, return_counts=True)
    if kernels:
        return float(kernels.expected_mi_terms(float(n), av, ac, bv, bc).sum())
    lg_n = gammaln(n + 1)
    emi = 0.0
    for x, cx in zip(av.tolist(), ac.tolist(), strict=True):
        for y, cy in zip(bv.tolist(), bc.tolist(), strict=True):
            mean, width = x * y / n, math.sqrt(35 * min(x, y))
            low = max(1, x + y - n, math.ceil(mean - width))
            high = min(x, y, math.floor(mean + width))
            nij = np.arange(low, high + 1, dtype=np.float64)
            if not len(nij):
                continue
            log_p = (
                gammaln(x + 1)
                + gammaln(y + 1)
                + gammaln(n - x + 1)
                + gammaln(n - y + 1)
                - lg_n
                - gammaln(nij + 1)
                - gammaln(x - nij + 1)
                - gammaln(y - nij + 1)
                - gammaln(n - x - y + nij + 1)
            )
            terms = nij / n * np.log(n * nij / (x * y)) * np.exp(log_p)
            emi += cx * cy * float(terms.sum())
    return emi
