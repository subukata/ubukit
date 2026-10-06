"""Textbook NumPy implementations of every method: the baseline of run.py --baseline.

Not part of the package. Each is the form a user would write from the
equations in docs/algorithms.md, without UbuKit's shortcuts: distances one
prototype at a time, the plain membership formulas, dense neighborhood and
adjacency matrices, full expectation sums and ranks by sorting. They take
UbuKit's arguments and must give its results (run.py marks a difference).
They hold dense (N, N) arrays, so run them at sizes up to m.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = [
    "ami",
    "ari",
    "batch_som",
    "continuity",
    "efcm",
    "fcm",
    "kmeans",
    "rcm",
    "rmcm",
    "som",
    "som_olp",
    "trustworthiness",
]


def _sqd(X, C):
    """Squared distances (N, K), one prototype at a time."""
    return np.stack([((X - c) ** 2).sum(axis=1) for c in C], axis=1)


def _mean(W, X, V):
    """Means of X weighted by the columns of W; columns without mass keep V."""
    mass = W.sum(axis=0)
    out = V.copy()
    out[mass > 0] = (W.T @ X)[mass > 0] / mass[mass > 0, None]
    return out


def _fit(V, labels, n_iter):
    return {"labels": labels.tolist(), "centers": V.tolist(), "n_iter": n_iter}


def _clusters(X, init, step, max_iter, tol):
    """Repeat V, labels = step(V) until no center moves more than tol * RMS radius."""
    X, V = np.asarray(X, float), np.asarray(init, float)
    limit = tol * np.sqrt(((X - X.mean(axis=0)) ** 2).sum(axis=1).mean())
    for n_iter in range(1, max_iter + 1):  # noqa: B007 (the count is the result)
        new, labels = step(X, V)
        done = np.abs(new - V).max() <= limit
        V = new
        if done:
            break
    return _fit(V, labels, n_iter)


def kmeans(X, k, *, init, max_iter=300):
    def step(X, V):
        labels = _sqd(X, V).argmin(axis=1)
        return _mean(np.eye(len(V))[labels], X, V), labels

    return _clusters(X, init, step, max_iter, 0.0)


def fcm(X, k, *, init, m=2.0, max_iter=300, tol=1e-6):
    def step(X, V):
        W = np.fmax(_sqd(X, V), np.finfo(float).tiny) ** (-1.0 / (m - 1.0))
        U = W / W.sum(axis=1, keepdims=True)
        return _mean(U**m, X, V), U.argmax(axis=1)

    return _clusters(X, init, step, max_iter, tol)


def efcm(X, k, *, init, tau=1.0, max_iter=300, tol=1e-6):
    def step(X, V):
        E = np.exp(-_sqd(X, V) / tau)
        U = E / E.sum(axis=1, keepdims=True)
        return _mean(U, X, V), U.argmax(axis=1)

    return _clusters(X, init, step, max_iter, tol)


def rcm(X, k, *, init, alpha=1.1, beta=0.0, p=1.0, max_iter=300):
    def step(X, V):
        d = np.sqrt(_sqd(X, V))
        A = d <= ((alpha * d.min(axis=1, keepdims=True)) ** p + beta**p) ** (1.0 / p)
        U = A / A.sum(axis=1, keepdims=True)
        return _mean(U, X, V), U.argmax(axis=1)

    return _clusters(X, init, step, max_iter, 0.0)


def rmcm(X, k, delta, *, init, max_iter=300):
    X = np.asarray(X, float)
    A = sum((X[:, f, None] - X[None, :, f]) ** 2 for f in range(X.shape[1])) <= delta**2
    P = A / A.sum(axis=1, keepdims=True)  # dense (N, N)
    del A

    def step(X, V):
        R = P @ np.eye(len(V))[_sqd(X, V).argmin(axis=1)]
        return _mean(R, X, V), R.argmax(axis=1)

    return _clusters(X, init, step, max_iter, 0.0)


def _map_start(X, grid, scale=2.0):
    """Centered X, unit coordinates and prototypes on the leading principal plane."""
    X = np.asarray(X, float)
    mean = X.mean(axis=0)
    X = X - mean
    rows, cols = grid
    R = np.column_stack(np.divmod(np.arange(rows * cols), cols)).astype(float)
    values, vectors = np.linalg.eigh(X.T @ X / len(X))
    q = min(2, X.shape[1])
    values, axes = values[::-1][:q], vectors[:, ::-1][:, :q].T.copy()
    for a in axes:  # the first clearly nonzero component positive
        a *= np.sign(a[np.abs(a) > 1e-6 * np.abs(a).max()][0])
    G = R[:, :q] - R[:, :q].mean(axis=0)
    G /= np.maximum(np.abs(G).max(axis=0), 1e-12)
    W = (G * scale * np.sqrt(np.maximum(values, 0.0))) @ axes
    return X, mean, R, W, np.ptp(R, axis=0).max() / 2 or 1.0


def _map(X, W, mean, n_iter):
    return _fit(W + mean, _sqd(X, W).argmin(axis=1), n_iter)


def som(
    X,
    grid=(10, 10),
    *,
    epochs=10,
    sigma=None,
    sigma_end=0.5,
    lr=0.5,
    lr_end=0.01,
    shuffle=True,
    seed=None,
    engine=None,
):
    X, mean, R, W, s0 = _map_start(X, grid)
    s0, s1, n = sigma or s0, sigma_end, len(X)
    rng, steps = np.random.default_rng(seed), epochs * n
    for e in range(epochs):
        for j, i in enumerate(rng.permutation(n) if shuffle else range(n)):
            f = (e * n + j) / (steps - 1) if steps > 1 else 0.0
            s, eta = s0 * (s1 / s0) ** f, lr * (lr_end / lr) ** f
            bmu = ((W - X[i]) ** 2).sum(axis=1).argmin()
            h = np.exp(-((R - R[bmu]) ** 2).sum(axis=1) / (2 * s * s))
            W += eta * h[:, None] * (X[i] - W)
    return _map(X, W, mean, epochs)


def batch_som(X, grid=(10, 10), *, epochs=50, sigma=None, sigma_end=0.5):
    X, mean, R, W, s0 = _map_start(X, grid)
    s0, s1 = sigma or s0, sigma_end
    for t in range(epochs):
        s = s0 * (s1 / s0) ** (t / (epochs - 1)) if epochs > 1 else s0
        H = np.exp(-_sqd(R[_sqd(X, W).argmin(axis=1)], R) / (2 * s * s))  # dense (N, K)
        W = _mean(H, X, W)
    return _map(X, W, mean, epochs)


def som_olp(X, grid=(10, 10), *, lam, gamma, pca_scale=2.0, max_iter=100, tol=1e-6):
    X, mean, R, W, _ = _map_start(X, grid, pca_scale)
    limit = tol * np.sqrt((X**2).sum(axis=1).mean())
    P = None
    for n_iter in range(1, max_iter + 1):  # noqa: B007 (the count is the result)
        cost = _sqd(X, W) if P is None else _sqd(X, W) + gamma * _sqd(P @ R, R)
        E = np.exp(-cost / lam)
        P = E / E.sum(axis=1, keepdims=True)
        new = _mean(P, X, W)
        done = np.abs(new - W).max() <= limit
        W = new
        if done:
            break
    return _fit(W + mean, P.argmax(axis=1), n_iter)


def _contingency(a, b):
    ca, ia = np.unique(a, return_inverse=True)
    cb, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ca), len(cb)))
    np.add.at(C, (ia, ib), 1)
    return C


def ari(a, b):
    C, n = _contingency(a, b), len(a)

    def pairs(x):
        return (x * (x - 1) / 2).sum()

    index, sa, sb = pairs(C), pairs(C.sum(axis=1)), pairs(C.sum(axis=0))
    expected = sa * sb / (n * (n - 1) / 2)
    return float((index - expected) / ((sa + sb) / 2 - expected))


def ami(a, b, *, engine=None):
    """Arithmetic-mean AMI with the full hypergeometric expectation, pair by pair."""
    C, n = _contingency(a, b), len(a)
    ra, rb = C.sum(axis=1), C.sum(axis=0)
    nz = C > 0
    mi = (C[nz] / n * np.log(n * C[nz] / np.outer(ra, rb)[nz])).sum()
    emi = 0.0
    for x in ra:
        for y in rb:
            nij = np.arange(max(1, x + y - n), min(x, y) + 1)
            log_p = (
                gammaln(x + 1)
                + gammaln(y + 1)
                + gammaln(n - x + 1)
                + gammaln(n - y + 1)
                - gammaln(n + 1)
                - gammaln(nij + 1)
                - gammaln(x - nij + 1)
                - gammaln(y - nij + 1)
                - gammaln(n - x - y + nij + 1)
            )
            emi += (nij / n * np.log(n * nij / (x * y)) * np.exp(log_p)).sum()

    def entropy(r):
        return -(r / n * np.log(r / n)).sum()

    return float((mi - emi) / ((entropy(ra) + entropy(rb)) / 2 - emi))


def trustworthiness(X, Y, k=5, *, engine=None):
    """Ranks from full distance matrices sorted by (distance, index)."""
    X, Y, n = np.asarray(X, float), np.asarray(Y, float), len(X)
    dx = sum((X[:, f, None] - X[None, :, f]) ** 2 for f in range(X.shape[1]))
    dy = sum((Y[:, f, None] - Y[None, :, f]) ** 2 for f in range(Y.shape[1]))
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    rows = np.arange(n)[:, None]
    ranks = np.empty((n, n), np.int64)
    ranks[rows, np.argsort(dx, axis=1, kind="stable")] = np.arange(1, n + 1)
    r = ranks[rows, np.argsort(dy, axis=1, kind="stable")[:, :k]] - k
    return float(1.0 - 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)) * r[r > 0].sum())


def continuity(X, Y, k=5, *, engine=None):
    return trustworthiness(Y, X, k)
