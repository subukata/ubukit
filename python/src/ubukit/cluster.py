"""Partitional clustering on the shared engine.

The fuzzy and rough c-means define how memberships follow from squared
distances and run the standard step; k-means runs Lloyd's step with
Hamerly's bounds. See docs/algorithms.md for the update equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.spatial import cKDTree

from ._core import (
    TINY,
    Data,
    Init,
    Loop,
    Result,
    Steps,
    as_data,
    check_float,
    check_int,
    check_max_iter,
    fitted,
    iterate,
    label_mean,
    lloyd,
    softmax_rows,
    sq_norms,
    sqdist,
    start,
    weighted_mean,
)


def kmeans(
    X: ArrayLike,
    k: int,
    *,
    init: Init = "k-means++",
    max_iter: int | None = 300,
    seed: int | None = None,
) -> Steps:
    """Lloyd's k-means; distance ties go to the lowest center index.

    Iterates with Hamerly's (2010) bounds, which skip only distance
    computations that cannot change a label, so the iterates are Lloyd's.

    Args:
        X: (N, D) data.
        k: number of clusters.
        init: ``"k-means++"`` or a (k, D) array of initial centers.
        max_iter: iteration limit (None: none); stops earlier at a fixed point.
        seed: seed for k-means++.
    """
    data, V = start(X, k, init, seed)

    def view(loop):
        return fitted(loop, loop.state[0].copy())

    max_iter = check_max_iter(max_iter)
    return (yield from iterate(data, V, _hamerly, max_iter=max_iter, tol=0.0, view=view))


def _hamerly(
    data: Data, V: np.ndarray, state: tuple[np.ndarray, np.ndarray] | None, t: int
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], float]:
    """Lloyd's step with Hamerly's bounds; the state is (labels, lower bounds).

    A point keeps its label without computing its other distances when its
    exact distance to its center is below both half the distance from that
    center to the nearest other one (which proves the center nearest, for any
    labeling) and a lower bound on its distance to every other center: the
    second-nearest distance when last computed, minus how far the other
    centers have moved since. Points within 1e-9 of a bound are recomputed,
    so rounding never decides a label. The bounds hold for one data set, so
    new data start them again (iterate keeps no state by default).
    """
    X, xx = data.X, data.xx
    n, k = len(X), len(V)
    # Copies: a state once returned is never modified (see iterate).
    labels, lower = (
        (state[0].copy(), state[1].copy()) if state else (np.zeros(n, np.intp), np.zeros(n))
    )
    vv = sq_norms(V)
    # Exact squared distances to the assigned centers: the objective, and
    # the quantity the bounds are compared with.
    own = xx + vv[labels] - 2.0 * np.einsum("ij,ij->i", X, V[labels])
    np.maximum(own, 0.0, out=own)
    C = sqdist(V, V, vv, vv)
    np.fill_diagonal(C, np.inf)
    bound = np.maximum(0.5 * np.sqrt(C.min(axis=1))[labels], lower)
    far = np.flatnonzero(own >= (bound * (1.0 - 1e-9)) ** 2)
    if len(far):
        D = sqdist(X[far], V, xx[far], vv)
        nearest, rows = D.argmin(axis=1), np.arange(len(far))
        labels[far], own[far] = nearest, D[rows, nearest]
        D[rows, nearest] = np.inf  # the second-nearest distance is what remains
        lower[far] = np.sqrt(D.min(axis=1))
    V_new = label_mean(X, labels, V)
    if k > 1:
        moved = np.sqrt(sq_norms(V_new - V))
        top = int(moved.argmax())
        lower = lower - np.where(labels == top, np.partition(moved, -2)[-2], moved[top])
    return V_new, (labels, lower), float(own.sum())


def fcm(
    X: ArrayLike,
    k: int,
    *,
    m: float = 2.0,
    init: Init = "k-means++",
    max_iter: int | None = 300,
    tol: float = 1e-6,
    seed: int | None = None,
) -> Steps:
    """Fuzzy c-means (Bezdek).

    Memberships u_ic are proportional to d_ic^(-2/(m-1)), evaluated as powers
    of the ratios d_min / d_ic <= 1, so every fuzzifier m > 1 uses the same
    stable path. Stops when no center moves more than ``tol`` times the RMS
    radius of X. In high dimensions a large m (from about D/(D-2) for
    isotropic data) can draw every center to the mean of X; see "Degenerate
    solutions" in docs/algorithms.md.
    """
    m = check_float(m, "m", 1.0, strict=True)
    data, V = start(X, k, init, seed)

    def assign(_, D):
        # (d_min / d)^(2/(m-1)), normalized: the softmax of -log d^2 / (m - 1)
        # without logarithms or exponentials (none at all for m = 2).
        D = np.maximum(D, TINY)
        U = D.min(axis=1, keepdims=True) / D
        U **= 1.0 / (m - 1.0)
        U /= U.sum(axis=1, keepdims=True)
        return U

    def update(data, U, V):
        # Scaling each column by its maximum leaves the means unchanged and
        # keeps u^m from underflowing to all zeros for large m.
        return weighted_mean(data.X, (U / np.maximum(U.max(axis=0), TINY)) ** m, V)

    def objective(D, U):
        # At the memberships of these distances, sum_c u_ic^m d_ic^2 equals
        # d_min^2 u_max^(m-1) for each point.
        return float(D.min(axis=1) @ U.max(axis=1) ** (m - 1.0))

    step = lloyd(assign, update, objective)
    max_iter, tol = check_max_iter(max_iter), check_float(tol, "tol", 0.0)
    return (yield from iterate(data, V, step, max_iter=max_iter, tol=tol, view=_soft))


def efcm(
    X: ArrayLike,
    k: int,
    *,
    tau: float = 1.0,
    init: Init = "k-means++",
    max_iter: int | None = 300,
    tol: float = 1e-6,
    seed: int | None = None,
) -> Steps:
    """Entropy-regularized fuzzy c-means (Miyamoto).

    Minimizes sum u d^2 + tau * sum u log u, giving u_ic = softmax_c(-d_ic^2 / tau).
    A tau of at least twice the largest variance of X can draw every center to
    the mean of X; see "Degenerate solutions" in docs/algorithms.md.
    """
    tau = check_float(tau, "tau", 0.0, strict=True)
    data, V = start(X, k, init, seed)

    def objective(D, U):
        # At the memberships of these distances, sum_c u_ic d_ic^2 + tau u_ic log u_ic
        # equals d_min^2 + tau log u_max for each point: N logarithms, not N K.
        return float(np.sum(D.min(axis=1) + tau * np.log(U.max(axis=1))))

    step = lloyd(
        assign=lambda _, D: softmax_rows(D * (-1.0 / tau)),
        update=lambda data, U, V: weighted_mean(data.X, U, V),
        objective=objective,
    )
    max_iter, tol = check_max_iter(max_iter), check_float(tol, "tol", 0.0)
    return (yield from iterate(data, V, step, max_iter=max_iter, tol=tol, view=_soft))


def rcm(
    X: ArrayLike,
    k: int,
    *,
    alpha: float = 1.1,
    beta: float = 0.0,
    p: float = 1.0,
    init: Init = "k-means++",
    max_iter: int | None = 300,
    seed: int | None = None,
) -> Steps:
    """Rough c-means; ``p != 1`` gives the extended ExRCM.

    Cluster c is admissible for x_i when d_ic^p <= (alpha d_i,min)^p + beta^p,
    and x_i shares unit membership equally among its admissible clusters.
    Stops at a fixed point of the memberships.
    """
    alpha = check_float(alpha, "alpha", 1.0)
    beta = check_float(beta, "beta", 0.0)
    p = check_float(p, "p", 0.0, strict=True)
    data, V = start(X, k, init, seed)

    def assign(_, D):
        d = np.sqrt(D)
        a = alpha * d.min(axis=1, keepdims=True)
        # radius = ((alpha d_min)^p + beta^p)^(1/p), scaled to avoid under/overflow.
        large, small = np.maximum(a, beta), np.minimum(a, beta)
        ratio = np.divide(small, large, out=np.zeros_like(large), where=large > 0)
        with np.errstate(over="ignore"):
            radius = large * (1.0 + ratio**p) ** (1.0 / p)
        mask = d <= radius
        return mask / mask.sum(axis=1, keepdims=True)

    step = lloyd(assign, update=lambda data, U, V: weighted_mean(data.X, U, V))
    max_iter = check_max_iter(max_iter)
    return (yield from iterate(data, V, step, max_iter=max_iter, tol=0.0, view=_soft))


def rmcm(
    X: ArrayLike,
    k: int,
    delta: float,
    *,
    init: Init = "k-means++",
    max_iter: int | None = 300,
    max_edges: int = 10_000_000,
    seed: int | None = None,
) -> Steps:
    """Rough membership c-means.

    With P the row-normalized delta-neighborhood graph (self included) and H
    the one-hot nearest-center assignment, memberships are R = P H.

    Args:
        delta: neighborhood radius (Euclidean, inclusive).
        max_edges: refuse graphs with more directed edges than this.
    """
    delta = check_float(delta, "delta", 0.0)
    max_edges = check_int(max_edges, "max_edges", 1)

    def prepare(rows, n_features=None):
        data = as_data(rows, n_features)
        P = _neighborhood(data.X, delta, max_edges)
        return _Neighbors(data.X, data.mean, data.xx, data.radius, P)

    data, V = start(X, k, init, seed, prepare)

    def assign(data, D):
        n = len(D)
        H = sparse.csr_matrix((np.ones(n), D.argmin(axis=1), np.arange(n + 1)), shape=D.shape)
        return (data.P @ H).toarray()

    step = lloyd(assign, update=lambda data, U, V: weighted_mean(data.X, U, V))
    max_iter = check_max_iter(max_iter)
    return (
        yield from iterate(data, V, step, max_iter=max_iter, tol=0.0, view=_soft, prepare=prepare)
    )


@dataclass(frozen=True, slots=True, eq=False)
class _Neighbors(Data):
    """The data of rmcm with their row-normalized delta-neighborhood graph P."""

    P: sparse.csr_matrix


def _soft(loop: Loop) -> Result:
    """The view of soft and rough clusterings: memberships are the state."""
    U = loop.state.copy()
    return fitted(loop, U.argmax(axis=1), U)


def _neighborhood(X: np.ndarray, delta: float, max_edges: int) -> sparse.csr_matrix:
    """Row-normalized adjacency of ||x_i - x_j|| <= delta, self loops included."""
    tree = cKDTree(X)
    edges = int(tree.count_neighbors(tree, delta))
    if edges > max_edges:
        raise ValueError(f"delta={delta} gives {edges} edges > max_edges={max_edges}")
    pairs = tree.query_pairs(delta, output_type="ndarray")
    n = len(X)
    i = np.concatenate([np.arange(n), pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([np.arange(n), pairs[:, 1], pairs[:, 0]])
    degree = np.bincount(i, minlength=n)
    return sparse.csr_matrix((1.0 / degree[i], (i, j)), shape=(n, n))
