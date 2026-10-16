"""External agreement (ARI, AMI) and neighborhood preservation (trustworthiness, continuity)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Literal, overload

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
    # This expression and its fn = fp = 0 case follow scikit-learn (BSD-3-Clause;
    # see THIRD_PARTY_NOTICES.txt).
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
    # The single-cluster case and the signed epsilon below follow scikit-learn
    # (BSD-3-Clause; see THIRD_PARTY_NOTICES.txt).
    if len(a) == len(b) <= 1:
        return 1.0
    mi = float(np.sum(nij / n * (np.log(nij) + np.log(n) - np.log(a[rows]) - np.log(b[cols]))))
    emi = _expected_mutual_information(n, a, b, kernels)
    denominator = _AVERAGES[average](_entropy(a, n), _entropy(b, n)) - emi
    eps = np.finfo(np.float64).eps
    denominator = min(denominator, -eps) if denominator < 0 else max(denominator, eps)
    return (mi - emi) / denominator


class Neighborhood:
    """The pairwise distances of N points, computed once to score many embeddings of them.

    ``trustworthiness`` and ``continuity`` take it in place of the points and
    give the same values without computing their distances again, which is
    most of the work when the points have many features: a search that
    scores many embeddings Y of one X builds ``Neighborhood(X)`` once. It
    holds the (N, N) squared distances, from direct differences as the
    metrics compute them, in 8 N^2 bytes (72 MB for N = 3,000), and never
    changes.
    """

    __slots__ = ("_distances",)

    def __init__(self, X: ArrayLike) -> None:
        X = as_matrix(X)
        distances = cdist(X, X, "sqeuclidean")
        np.fill_diagonal(distances, np.inf)  # a point is not its own neighbor
        distances.flags.writeable = False
        self._distances = distances

    def __repr__(self) -> str:
        return f"Neighborhood(n={len(self._distances)})"


@overload
def trustworthiness(
    X: ArrayLike | Neighborhood, Y: ArrayLike | Neighborhood, k: int = 5, *, engine: Engine = ...
) -> float: ...
@overload
def trustworthiness(
    X: ArrayLike | Neighborhood,
    Y: ArrayLike | Neighborhood,
    k: Sequence[int],
    *,
    engine: Engine = ...,
) -> np.ndarray: ...
def trustworthiness(
    X: ArrayLike | Neighborhood,
    Y: ArrayLike | Neighborhood,
    k: int | Sequence[int] = 5,
    *,
    engine: Engine = "numpy",
) -> float | np.ndarray:
    """Trustworthiness of embedding Y of X (Venna & Kaski, 2001).

    Penalizes points that are among the k nearest neighbors in Y but not in X,
    by their rank in X. Distances are computed from direct differences, so
    exact ties (e.g. SOM grid coordinates) stay exact and are ordered by index.
    ``k`` may be a sequence, giving an array with one value per entry, in its
    order, for the work of the largest. X or Y may be a ``Neighborhood`` of
    the points, whose distances are then not computed again.
    ``engine="numba"`` computes the same penalties in parallel compiled loops.
    """
    return _trustworthiness(X, Y, k, engine)


@overload
def continuity(
    X: ArrayLike | Neighborhood, Y: ArrayLike | Neighborhood, k: int = 5, *, engine: Engine = ...
) -> float: ...
@overload
def continuity(
    X: ArrayLike | Neighborhood,
    Y: ArrayLike | Neighborhood,
    k: Sequence[int],
    *,
    engine: Engine = ...,
) -> np.ndarray: ...
def continuity(
    X: ArrayLike | Neighborhood,
    Y: ArrayLike | Neighborhood,
    k: int | Sequence[int] = 5,
    *,
    engine: Engine = "numpy",
) -> float | np.ndarray:
    """Continuity of embedding Y of X: trustworthiness with the roles swapped."""
    return _trustworthiness(Y, X, k, engine)


def _trustworthiness(X: Any, Y: Any, k: Any, engine: Engine) -> float | np.ndarray:
    """``trustworthiness(X, Y, k)``, which ``continuity`` takes with X and Y swapped."""
    kernels = numba_kernels(engine)
    (A, a_stored), (B, b_stored) = _points(X, "X"), _points(Y, "Y")
    n = len(A)
    if len(B) != n:
        raise ValueError("X and Y must have the same number of rows")
    ks = _neighbor_counts(k, n)
    if kernels:
        counts = np.array(ks, np.int64)
        penalties = kernels.trustworthiness_penalties(A, a_stored, B, b_stored, counts).sum(axis=0)
    else:
        penalties = _penalties(A, a_stored, B, b_stored, ks)
    values = [
        1.0 - 2.0 * int(penalty) / (n * q * (2.0 * n - 3.0 * q - 1.0))
        for penalty, q in zip(penalties, ks, strict=True)
    ]
    return values[0] if np.ndim(k) == 0 else np.array(values)


def _points(Z: Any, name: str) -> tuple[np.ndarray, bool]:
    """The rows of Z, or the distances of a Neighborhood, with whether they are distances."""
    if isinstance(Z, Neighborhood):
        return Z._distances, True
    return as_matrix(Z, name), False


def _neighbor_counts(k: Any, n: int) -> list[int]:
    """The numbers of neighbors asked for: one, or a non-empty sequence of them."""
    ks = [check_int(q, "k", 1) for q in ([k] if np.ndim(k) == 0 else list(k))]
    if not ks:
        raise ValueError("k must be an integer or a non-empty sequence of integers")
    if not all(2 * q < n for q in ks):
        raise ValueError("k must satisfy 1 <= k < n / 2")
    return ks


def _penalties(
    A: np.ndarray, a_stored: bool, B: np.ndarray, b_stored: bool, ks: list[int]
) -> list[int]:
    """Per k in ks, the sum over points of max(rank in A - k, 0) for their k nearest in B.

    A and B are points, or their distances when stored. The neighbors in B
    come sorted by (distance, index), so the k nearest for each k are the
    first k of the largest k's, and one count of their ranks serves every k.
    """
    n, top = len(A), max(ks)
    index = np.arange(n)
    block = max(1, (1 << 22) // (n * (top + 2)))
    penalty = [0] * len(ks)
    for start in range(0, n, block):
        rows = index[start : start + block]
        da, db = _rows(A, a_stored, rows), _rows(B, b_stored, rows)
        neighbors = _nearest(db, top)
        target = np.take_along_axis(da, neighbors, axis=1)[:, :, None]
        # 1-based rank in A under the (distance, index) order.
        rank = 1 + (da[:, None, :] < target).sum(axis=2)
        rank += ((da[:, None, :] == target) & (index < neighbors[:, :, None])).sum(axis=2)
        for q, k in enumerate(ks):
            penalty[q] += int(np.maximum(rank[:, :k] - k, 0).sum())
    return penalty


def _rows(A: np.ndarray, stored: bool, rows: np.ndarray) -> np.ndarray:
    """Squared distances from the given rows to every row, inf at the row itself."""
    if stored:
        return A[rows]
    d = cdist(A[rows], A, "sqeuclidean")
    d[np.arange(len(rows)), rows] = np.inf
    return d


def _nearest(d: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the k smallest entries per row, in (value, index) order."""
    kth = np.partition(d, k - 1, axis=1)[:, k - 1 : k]
    below = d < kth
    tied = d == kth
    room = k - below.sum(axis=1, keepdims=True)
    chosen = below | (tied & (np.cumsum(tied, axis=1) <= room))
    columns = np.nonzero(chosen)[1].reshape(len(d), k)  # increasing index in each row
    order = np.argsort(np.take_along_axis(d, columns, axis=1), axis=1, kind="stable")
    return np.take_along_axis(columns, order, axis=1)


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
    lf = gammaln(np.arange(n + 1) + 1.0)  # log t! for t = 0..n, looked up below
    if kernels:
        return float(kernels.expected_mi_terms(n, av, ac, bv, bc, lf).sum())
    emi = 0.0
    for x, cx in zip(av.tolist(), ac.tolist(), strict=True):
        # The windows of x with every size y, flattened into one array of n_ij.
        mean, width = x * bv / n, np.sqrt(35 * np.minimum(x, bv))
        low = np.maximum(np.maximum(1, x + bv - n), np.ceil(mean - width)).astype(np.intp)
        high = np.minimum(np.minimum(x, bv), np.floor(mean + width)).astype(np.intp)
        size = np.maximum(high - low + 1, 0)
        pair = np.repeat(np.arange(len(bv)), size)
        nij = low[pair] + np.arange(len(pair)) - np.repeat(np.cumsum(size) - size, size)
        y = bv[pair]
        log_p = (
            lf[x]
            + lf[y]
            + lf[n - x]
            + lf[n - y]
            - lf[n]
            - lf[nij]
            - lf[x - nij]
            - lf[y - nij]
            - lf[n - x - y + nij]
        )
        terms = nij / n * np.log(n * nij / (x * y)) * np.exp(log_p)
        emi += cx * float(np.bincount(pair, terms, minlength=len(bv)) @ bc)
    return emi
