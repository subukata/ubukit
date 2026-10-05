"""Partitional clustering on the shared alternating engine.

Each method below only defines how memberships follow from squared distances.
See docs/algorithms.md for the update equations.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.special import xlogy

from ._core import (
    TINY,
    Result,
    alternate,
    check_float,
    check_int,
    label_mean,
    prepare,
    softmax_rows,
    weighted_mean,
)


def kmeans(X, k, *, init="k-means++", max_iter=300, seed=None) -> Result:
    """Lloyd's k-means; distance ties go to the lowest center index.

    Args:
        X: (N, D) data.
        k: number of clusters.
        init: ``"k-means++"`` or a (k, D) array of initial centers.
        max_iter: iteration limit; stops earlier at a fixed point.
        seed: seed for k-means++.
    """
    X, mean, V = prepare(X, k, init, seed)
    V, labels, n_iter, converged, history = alternate(
        X,
        V,
        assign=lambda D, _, t: D.argmin(axis=1),
        update=lambda labels, V, t: label_mean(X, labels, V),
        objective=lambda D, labels: float(D[np.arange(len(D)), labels].sum()),
        max_iter=check_int(max_iter, "max_iter", 1),
        tol=0.0,
    )
    return Result(V + mean, labels, None, n_iter, converged, history)


def fcm(X, k, *, m=2.0, init="k-means++", max_iter=300, tol=1e-6, seed=None) -> Result:
    """Fuzzy c-means (Bezdek).

    Memberships u_ic are proportional to d_ic^(-2/(m-1)), evaluated as a softmax
    of log-distances, so every fuzzifier m > 1 uses the same stable path.
    Stops when no center moves more than ``tol`` times the RMS radius of X.
    """
    m = check_float(m, "m", 1.0, strict=True)
    X, mean, V = prepare(X, k, init, seed)

    def assign(D, _, t):
        L = np.log(np.maximum(D, TINY))
        L *= -1.0 / (m - 1.0)
        return softmax_rows(L)

    def update(U, V, t):
        # Scaling each column by its maximum leaves the means unchanged and
        # keeps u^m from underflowing to all zeros for large m.
        return weighted_mean(X, (U / np.maximum(U.max(axis=0), TINY)) ** m, V)

    V, U, n_iter, converged, history = alternate(
        X,
        V,
        assign,
        update,
        objective=lambda D, U: float(np.sum(U**m * D)),
        max_iter=check_int(max_iter, "max_iter", 1),
        tol=check_float(tol, "tol", 0.0),
    )
    return Result(V + mean, U.argmax(axis=1), U, n_iter, converged, history)


def efcm(X, k, *, tau=1.0, init="k-means++", max_iter=300, tol=1e-6, seed=None) -> Result:
    """Entropy-regularized fuzzy c-means (Miyamoto).

    Minimizes sum u d^2 + tau * sum u log u, giving u_ic = softmax_c(-d_ic^2 / tau).
    """
    tau = check_float(tau, "tau", 0.0, strict=True)
    X, mean, V = prepare(X, k, init, seed)
    V, U, n_iter, converged, history = alternate(
        X,
        V,
        assign=lambda D, _, t: softmax_rows(D * (-1.0 / tau)),
        update=lambda U, V, t: weighted_mean(X, U, V),
        objective=lambda D, U: float(np.sum(U * D) + tau * np.sum(xlogy(U, U))),
        max_iter=check_int(max_iter, "max_iter", 1),
        tol=check_float(tol, "tol", 0.0),
    )
    return Result(V + mean, U.argmax(axis=1), U, n_iter, converged, history)


def rcm(X, k, *, alpha=1.1, beta=0.0, p=1.0, init="k-means++", max_iter=300, seed=None) -> Result:
    """Rough c-means; ``p != 1`` gives the extended ExRCM.

    Cluster c is admissible for x_i when d_ic^p <= (alpha d_i,min)^p + beta^p,
    and x_i shares unit membership equally among its admissible clusters.
    Stops at a fixed point of the memberships.
    """
    alpha = check_float(alpha, "alpha", 1.0)
    beta = check_float(beta, "beta", 0.0)
    p = check_float(p, "p", 0.0, strict=True)
    X, mean, V = prepare(X, k, init, seed)

    def assign(D, _, t):
        d = np.sqrt(D)
        a = alpha * d.min(axis=1, keepdims=True)
        # radius = ((alpha d_min)^p + beta^p)^(1/p), scaled to avoid under/overflow.
        large, small = np.maximum(a, beta), np.minimum(a, beta)
        ratio = np.divide(small, large, out=np.zeros_like(large), where=large > 0)
        with np.errstate(over="ignore"):
            radius = large * (1.0 + ratio**p) ** (1.0 / p)
        mask = d <= radius
        return mask / mask.sum(axis=1, keepdims=True)

    V, U, n_iter, converged, history = alternate(
        X,
        V,
        assign,
        update=lambda U, V, t: weighted_mean(X, U, V),
        max_iter=check_int(max_iter, "max_iter", 1),
        tol=0.0,
    )
    return Result(V + mean, U.argmax(axis=1), U, n_iter, converged, history)


def rmcm(X, k, delta, *, init="k-means++", max_iter=300, max_edges=10_000_000, seed=None) -> Result:
    """Rough membership c-means.

    With P the row-normalized delta-neighborhood graph (self included) and H
    the one-hot nearest-center assignment, memberships are R = P H.

    Args:
        delta: neighborhood radius (Euclidean, inclusive).
        max_edges: refuse graphs with more directed edges than this.
    """
    delta = check_float(delta, "delta", 0.0)
    X, mean, V = prepare(X, k, init, seed)
    P = _neighborhood(X, delta, check_int(max_edges, "max_edges", 1))
    n, rows = len(X), np.arange(len(X) + 1)

    def assign(D, _, t):
        H = sparse.csr_matrix((np.ones(n), D.argmin(axis=1), rows), shape=(n, len(V)))
        return (P @ H).toarray()

    V, U, n_iter, converged, history = alternate(
        X,
        V,
        assign,
        update=lambda U, V, t: weighted_mean(X, U, V),
        max_iter=check_int(max_iter, "max_iter", 1),
        tol=0.0,
    )
    return Result(V + mean, U.argmax(axis=1), U, n_iter, converged, history)


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
