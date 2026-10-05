"""Shared numerics: validation, distances, initialization and the alternating engine.

Every partitional method in UbuKit is one loop,

    D = ||x_i - v_c||^2  ->  U = assign(D)  ->  V = update(U),

and differs only in ``assign`` (and, for SOMs, ``update``). Data are centered
once before fitting, which keeps the Gram-identity distances accurate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy import sparse

TINY = np.finfo(np.float64).tiny
# "Ordinary scale": row norms below 1e150 keep every squared distance between
# data, prototypes and grid points (~1e301 at most) far from float64 overflow.
MAX_SQ_NORM = 1e300


@dataclass(frozen=True, slots=True, eq=False)
class Result:
    """A fitted clustering or map.

    ``labels`` and ``membership`` come from the last assignment step, i.e. for
    the prototypes before the final update; they coincide at a fixed point.

    Attributes:
        centers: (K, D) prototypes in input coordinates.
        labels: (N,) index of the strongest membership (nearest prototype for hard methods).
        membership: (N, K) soft or rough memberships; ``None`` for hard methods.
        n_iter: iterations performed (epochs for SOMs).
        converged: whether the stopping rule was met (SOMs: the schedule completed).
        history: objective value per iteration; empty when the method has none.
        embedding: (N, Q) map coordinates for SOMs, otherwise ``None``.
    """

    centers: np.ndarray
    labels: np.ndarray
    membership: np.ndarray | None
    n_iter: int
    converged: bool
    history: np.ndarray
    embedding: np.ndarray | None = None


def as_matrix(X, name: str = "X") -> np.ndarray:
    """Return X as a finite, non-empty, C-contiguous float64 matrix of ordinary scale."""
    A = np.ascontiguousarray(X, dtype=np.float64)
    if A.ndim != 2 or 0 in A.shape:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if not np.isfinite(A).all():
        raise ValueError(f"{name} must contain only finite values")
    if not sq_norms(A).max() < MAX_SQ_NORM:
        raise ValueError(f"{name} is too large in scale for float64 distances; standardize it")
    return A


def check_int(value, name: str, low: int, high: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < low
        or (high is not None and value > high)
    ):
        bound = f">= {low}" if high is None else f"in [{low}, {high}]"
        raise ValueError(f"{name} must be an integer {bound}")
    return int(value)


def check_float(value, name: str, low: float = -np.inf, *, strict: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < low
        or (strict and value == low)
    ):
        raise ValueError(f"{name} must be a finite number {'>' if strict else '>='} {low}")
    return float(value)


def sq_norms(X: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", X, X)


def sqdist(X: np.ndarray, C: np.ndarray, xx=None, cc=None) -> np.ndarray:
    """Squared Euclidean distances (N, K) via the Gram identity, clipped at zero."""
    D = X @ C.T
    D *= -2.0
    D += (sq_norms(X) if xx is None else xx)[:, None]
    D += sq_norms(C) if cc is None else cc
    return np.maximum(D, 0.0, out=D)


def softmax_rows(L: np.ndarray) -> np.ndarray:
    """Row-wise softmax of the logits L, in place."""
    L -= L.max(axis=1, keepdims=True)
    np.exp(L, out=L)
    L /= L.sum(axis=1, keepdims=True)
    return L


def weighted_mean(X: np.ndarray, W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Means of X weighted by the columns of W; columns without mass keep V."""
    mass = W.sum(axis=0)
    out = V.copy()
    ok = mass > 0
    out[ok] = (W.T @ X)[ok] / mass[ok, None]
    return out


def label_sums(X: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-label sums of X rows and label counts (one sparse product)."""
    n = len(labels)
    H = sparse.csr_matrix((np.ones(n), labels, np.arange(n + 1)), shape=(n, k))
    return np.asarray(H.T @ X), np.bincount(labels, minlength=k)


def label_mean(X: np.ndarray, labels: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Per-label means of X; empty labels keep V."""
    sums, counts = label_sums(X, labels, len(V))
    out = V.copy()
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok, None]
    return out


def kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (Arthur & Vassilvitskii, 2007)."""
    n = len(X)
    xx = sq_norms(X)
    chosen = [int(rng.integers(n))]
    closest = sqdist(X, X[chosen], xx)[:, 0]
    for _ in range(1, k):
        total = closest.sum()
        if total > 0:
            i = int(np.searchsorted(np.cumsum(closest), rng.random() * total, side="right"))
        else:
            i = int(rng.integers(n))
        chosen.append(min(i, n - 1))
        np.minimum(closest, sqdist(X, X[chosen[-1:]], xx)[:, 0], out=closest)
    return X[chosen].copy()


def prepare(X, k, init, seed):
    """Validate and center X; return (centered X, mean, initial centers)."""
    X = as_matrix(X)
    k = check_int(k, "k", 1, len(X))
    mean = X.mean(axis=0)
    Xc = X - mean
    if isinstance(init, str):
        if init != "k-means++":
            raise ValueError("init must be 'k-means++' or a (k, n_features) array")
        V = kmeans_plus_plus(Xc, k, np.random.default_rng(seed))
        # Start from the seeds' cell means: a center sitting exactly on a data
        # point would give that point full weight in fuzzy updates with large m.
        V = label_mean(Xc, sqdist(Xc, V).argmin(axis=1), V)
    else:
        V = as_matrix(init, "init")
        if V.shape != (k, X.shape[1]):
            raise ValueError(f"init must have shape ({k}, {X.shape[1]})")
        V = V - mean
    return Xc, mean, V


Assign = Callable[[np.ndarray, np.ndarray | None, int], np.ndarray]
Update = Callable[[np.ndarray, np.ndarray, int], np.ndarray]
Objective = Callable[[np.ndarray, np.ndarray], float]


def alternate(
    X: np.ndarray,
    V: np.ndarray,
    assign: Assign,
    update: Update,
    *,
    max_iter: int,
    tol: float | None,
    objective: Objective | None = None,
):
    """Iterate D -> U -> V until the prototypes stop moving, or max_iter.

    ``assign(D, U_prev, t)`` returns integer labels or float memberships,
    ``update(U, V_prev, t)`` the new prototypes, and the optional
    ``objective(D, U)`` one history value. The loop stops when no
    prototype coordinate moves more than ``tol`` times the RMS radius of X;
    ``tol=0`` therefore means an exact fixed point and ``tol=None`` runs a
    fixed schedule. Returns (V, U, n_iter, converged, history).
    """
    xx = sq_norms(X)
    step = None if tol is None else tol * float(np.sqrt(xx.mean()))
    U = None
    history: list[float] = []
    for t in range(max_iter):
        D = sqdist(X, V, xx)
        U = assign(D, U, t)
        if objective is not None:
            history.append(objective(D, U))
        V, V_prev = update(U, V, t), V
        if step is not None and float(np.max(np.abs(V - V_prev))) <= step:
            return V, U, t + 1, True, np.asarray(history, dtype=float)
    return V, U, max_iter, False, np.asarray(history, dtype=float)
