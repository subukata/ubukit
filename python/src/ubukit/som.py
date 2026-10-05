"""Self-organizing maps: online SOM, batch SOM and SOM-OLP.

Units sit on a rectangular grid; unit j = row * cols + col has coordinates
(row, col). Prototypes start on the leading principal plane unless given.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.special import xlogy

from ._core import (
    TINY,
    Result,
    alternate,
    as_matrix,
    check_float,
    check_int,
    label_sums,
    softmax_rows,
    sq_norms,
    sqdist,
    weighted_mean,
)


def som(
    X,
    grid=(10, 10),
    *,
    epochs=10,
    sigma=None,
    sigma_end=0.5,
    lr=0.5,
    lr_end=0.01,
    init="pca",
    shuffle=True,
    seed=None,
) -> Result:
    """Online (sequential) SOM.

    Each sample moves every prototype by lr_t * h_t(j, bmu) * (x - w_j) with a
    Gaussian neighborhood h_t. sigma and lr decay geometrically over all
    epochs * N updates. ``grid`` is (rows, cols) or a (K, Q) array of unit
    coordinates. ``batch_som`` is much faster for large data.
    """
    X, mean, R, W = _setup(X, grid, init)
    epochs = check_int(epochs, "epochs", 1)
    s0, s1 = _sigmas(sigma, sigma_end, R)
    lr = check_float(lr, "lr", 0.0, strict=True)
    lr_end = check_float(lr_end, "lr_end", 0.0, strict=True)
    if max(lr, lr_end) > 1:
        raise ValueError("lr and lr_end must be at most 1")
    rng = np.random.default_rng(seed)
    n = len(X)
    steps = epochs * n
    t = 0
    for _ in range(epochs):
        for i in rng.permutation(n) if shuffle else range(n):
            f = t / (steps - 1) if steps > 1 else 0.0
            s, eta = s0 * (s1 / s0) ** f, lr * (lr_end / lr) ** f
            diff = X[i] - W
            bmu = np.einsum("ij,ij->i", diff, diff).argmin()
            g = R - R[bmu]
            h = np.exp(np.einsum("ij,ij->i", g, g) * (-0.5 / (s * s)))
            W += (eta * h)[:, None] * diff
            t += 1
    return _finish(X, mean, R, W, epochs)


def batch_som(X, grid=(10, 10), *, epochs=50, sigma=None, sigma_end=0.5, init="pca") -> Result:
    """Batch SOM: w_j = sum_i h(bmu_i, j) x_i / sum_i h(bmu_i, j).

    ``grid`` is (rows, cols). The Gaussian neighborhood is separable on the
    grid, so an epoch costs O(N K D + K (rows + cols) D) with no K x K kernel.
    """
    rows, cols = _shape(grid)
    X, mean, R, W = _setup(X, grid, init)
    epochs = check_int(epochs, "epochs", 1)
    s0, s1 = _sigmas(sigma, sigma_end, R)
    gr, gc = np.arange(rows), np.arange(cols)

    def update(labels, W, t):
        s = s0 * (s1 / s0) ** (t / (epochs - 1)) if epochs > 1 else s0
        Kr = np.exp(-((gr[:, None] - gr) ** 2) / (2 * s * s))
        Kc = np.exp(-((gc[:, None] - gc) ** 2) / (2 * s * s))
        sums, counts = label_sums(X, labels, len(W))
        num = Kc @ (Kr @ sums.reshape(rows, -1)).reshape(rows, cols, -1)
        den = (Kr @ counts.reshape(rows, cols) @ Kc.T).ravel()
        out = W.copy()
        ok = den > 0
        out[ok] = num.reshape(len(W), -1)[ok] / den[ok, None]
        return out

    W, *_ = alternate(X, W, lambda D, _, t: D.argmin(axis=1), update, max_iter=epochs, tol=None)
    return _finish(X, mean, R, W, epochs)


def som_olp(
    X, grid=(10, 10), *, lam, gamma, init="pca", pca_scale=2.0, max_iter=100, tol=1e-6
) -> Result:
    """SOM with optimized latent positions (SOM-OLP, Ubukata).

    Minimizes sum p_ij (||x_i - w_j||^2 + gamma ||v_i - r_j||^2) + lam sum p log p
    with latent positions v_i = sum_j p_ij r_j. ``embedding`` holds V.

    Args:
        grid: (rows, cols) for a rectangular grid, or a (K, Q) array of unit coordinates.
        lam: softmax temperature (> 0).
        gamma: weight of the latent-position term (>= 0).
        pca_scale: spread of the PCA initialization in standard deviations.
    """
    lam = check_float(lam, "lam", 0.0, strict=True)
    gamma = check_float(gamma, "gamma", 0.0)
    X, mean, R, W = _setup(X, grid, init, check_float(pca_scale, "pca_scale", 0.0, strict=True))
    rr = sq_norms(R)
    cost = None

    def assign(D, P, t):
        nonlocal cost
        cost = D if P is None else D + gamma * sqdist(P @ R, R, cc=rr)
        return softmax_rows(cost * (-1.0 / lam))

    W, P, n_iter, converged, history = alternate(
        X,
        W,
        assign,
        update=lambda P, W, t: weighted_mean(X, P, W),
        objective=lambda D, P: float(np.sum(P * cost) + lam * np.sum(xlogy(P, P))),
        max_iter=check_int(max_iter, "max_iter", 1),
        tol=check_float(tol, "tol", 0.0),
    )
    return Result(W + mean, P.argmax(axis=1), P, n_iter, converged, history, P @ R)


def _shape(grid) -> tuple[int, int]:
    if np.ndim(grid) != 1 or len(grid) != 2:
        raise ValueError("grid must be (rows, cols)")
    return check_int(grid[0], "rows", 1), check_int(grid[1], "cols", 1)


def _grid(grid) -> np.ndarray:
    """Unit coordinates: (rows, cols) -> rectangular grid, or an explicit (K, Q) array."""
    if np.ndim(grid) == 2:
        return as_matrix(grid, "grid")
    rows, cols = _shape(grid)
    r, c = np.divmod(np.arange(rows * cols), cols)
    return np.column_stack((r, c)).astype(np.float64)


def _setup(X, grid, init, pca_scale=2.0):
    X = as_matrix(X)
    mean = X.mean(axis=0)
    X = X - mean
    R = _grid(grid)
    if isinstance(init, str):
        if init != "pca":
            raise ValueError("init must be 'pca' or a (n_units, n_features) array")
        W = _pca_init(X, R, pca_scale)
    else:
        W = as_matrix(init, "init")
        if W.shape != (len(R), X.shape[1]):
            raise ValueError(f"init must have shape ({len(R)}, {X.shape[1]})")
        W = W - mean
    return X, mean, R, W


def _pca_init(X: np.ndarray, R: np.ndarray, scale: float) -> np.ndarray:
    """Spread the normalized grid over the leading principal axes of centered X.

    The axes come from the smaller of X^T X and X X^T (whose eigenvectors u
    give the axes X^T u), so the cost is O(N D min(N, D)).
    """
    n, d = X.shape
    q = min(R.shape[1], d, n)
    dual = d > n
    S = (X @ X.T if dual else X.T @ X) / n
    eigval, vec = eigh(S, subset_by_index=[len(S) - q, len(S) - 1])
    eigval, vec = eigval[::-1], vec[:, ::-1]
    if dual:
        vec = X.T @ vec / np.sqrt(np.maximum(n * eigval, TINY))
    axes = vec.T
    # Make the first clearly nonzero component of each axis positive. (The
    # largest component would be ambiguous: standardized 2-D data have axes
    # (1, +-1)/sqrt(2), and rounding would decide the orientation of the map.)
    mags = np.abs(axes)
    lead = (mags > 1e-6 * mags.max(axis=1, keepdims=True)).argmax(axis=1)
    axes *= np.sign(axes[np.arange(q), lead])[:, None]
    G = R[:, :q] - R[:, :q].mean(axis=0)
    G /= np.maximum(np.abs(G).max(axis=0), 1e-12)
    return (G * (scale * np.sqrt(np.maximum(eigval, 0.0)))) @ axes


def _sigmas(sigma, sigma_end, R):
    """Neighborhood width schedule endpoints; default start is half the grid extent."""
    if sigma is None:
        sigma = float(np.ptp(R, axis=0).max()) / 2 or 1.0
    return (
        check_float(sigma, "sigma", 0.0, strict=True),
        check_float(sigma_end, "sigma_end", 0.0, strict=True),
    )


def _finish(X, mean, R, W, epochs) -> Result:
    labels = sqdist(X, W).argmin(axis=1)
    return Result(W + mean, labels, None, epochs, True, np.empty(0), R[labels])
