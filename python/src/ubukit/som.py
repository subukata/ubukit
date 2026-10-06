"""Self-organizing maps: online SOM, batch SOM and SOM-OLP.

Units sit on a rectangular grid; unit j = row * cols + col has coordinates
(row, col). Prototypes start on the leading principal plane unless given.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh

from ._core import (
    TINY,
    Engine,
    Grid,
    MapInit,
    Result,
    Steps,
    as_matrix,
    check_float,
    check_int,
    iterate,
    label_sums,
    lloyd,
    nearest,
    numba_kernels,
    softmax_rows,
    sq_norms,
    sqdist,
    stepwise,
    weighted_mean,
)


@stepwise
def som(
    X: ArrayLike,
    grid: Grid = (10, 10),
    *,
    epochs: int = 10,
    sigma: float | None = None,
    sigma_end: float = 0.5,
    lr: float = 0.5,
    lr_end: float = 0.01,
    init: MapInit = "pca",
    shuffle: bool = True,
    seed: int | None = None,
    engine: Engine = "numpy",
) -> Steps:
    """Online (sequential) SOM.

    Each sample moves every prototype by lr_t * h_t(j, bmu) * (x - w_j) with a
    Gaussian neighborhood h_t. sigma and lr decay geometrically over all
    epochs * N updates. ``grid`` is (rows, cols) or a (K, Q) array of unit
    coordinates. ``batch_som`` is much faster for large data; so is
    ``engine="numba"``, which runs each epoch as a compiled loop with the same
    result (``pip install 'ubukit[numba]'``).
    """
    kernels = numba_kernels(engine)
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

    def epoch(W, _, e):
        order = rng.permutation(n) if shuffle else np.arange(n)
        if kernels:
            return kernels.som_epoch(X, W, R, order, e * n, steps, s0, s1, lr, lr_end), None, None
        W = W.copy()
        for j, i in enumerate(order):
            f = (e * n + j) / (steps - 1) if steps > 1 else 0.0
            s, eta = s0 * (s1 / s0) ** f, lr * (lr_end / lr) ** f
            diff = X[i] - W
            bmu = np.einsum("ij,ij->i", diff, diff).argmin()
            g = R - R[bmu]
            h = np.exp(np.einsum("ij,ij->i", g, g) * (-0.5 / (s * s)))
            W += (eta * h)[:, None] * diff
        return W, None, None

    return (
        yield from iterate(X, W, epoch, max_iter=epochs, tol=None, view=_map(X, mean, R, epochs))
    )


@stepwise
def batch_som(
    X: ArrayLike,
    grid: tuple[int, int] = (10, 10),
    *,
    epochs: int = 50,
    sigma: float | None = None,
    sigma_end: float = 0.5,
    init: MapInit = "pca",
) -> Steps:
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

    xx = sq_norms(X)

    def step(W, _, t):
        labels = nearest(X, W, xx)
        return update(labels, W, t), labels, None

    return (
        yield from iterate(X, W, step, max_iter=epochs, tol=None, view=_map(X, mean, R, epochs))
    )


@stepwise
def som_olp(
    X: ArrayLike,
    grid: Grid = (10, 10),
    *,
    lam: float,
    gamma: float,
    init: MapInit = "pca",
    pca_scale: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Steps:
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

    def objective(D, P):
        # At the memberships of these costs, sum_j p_ij cost_ij + lam p_ij log p_ij
        # equals cost_min + lam log p_max for each point: N logarithms, not N K.
        return float(np.sum(cost.min(axis=1) + lam * np.log(P.max(axis=1))))

    step = lloyd(X, assign, update=lambda P, W, t: weighted_mean(X, P, W), objective=objective)

    def view(loop):
        P = loop.state.copy()  # the next step reads it
        return Result(loop.V + mean, P.argmax(axis=1), P, *loop[2:], P @ R)

    max_iter, tol = check_int(max_iter, "max_iter", 1), check_float(tol, "tol", 0.0)
    return (yield from iterate(X, W, step, max_iter=max_iter, tol=tol, view=view))


def _shape(grid: Any) -> tuple[int, int]:
    if np.ndim(grid) != 1 or len(grid) != 2:
        raise ValueError("grid must be (rows, cols)")
    return check_int(grid[0], "rows", 1), check_int(grid[1], "cols", 1)


def _grid(grid: Grid) -> np.ndarray:
    """Unit coordinates: (rows, cols) -> rectangular grid, or an explicit (K, Q) array."""
    if np.ndim(grid) == 2:
        return as_matrix(grid, "grid")
    rows, cols = _shape(grid)
    r, c = np.divmod(np.arange(rows * cols), cols)
    return np.column_stack((r, c)).astype(np.float64)


def _setup(
    X: ArrayLike, grid: Grid, init: MapInit, pca_scale: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _map(X, mean, R, epochs):
    """The view of a map: best-matching units of the prototypes reached; the schedule
    is complete after ``epochs``."""

    def view(loop):
        labels = nearest(X, loop.V)
        done = loop.n_iter == epochs
        return Result(loop.V + mean, labels, None, loop.n_iter, done, np.empty(0), R[labels])

    return view
