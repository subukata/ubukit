"""Self-organizing maps: online SOM, batch SOM and SOM-OLP.

Units sit on a rectangular grid; unit j = row * cols + col has coordinates
(row, col). Prototypes start on the leading principal plane unless given.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh

from ._core import (
    MAX_SQ_NORM,
    TINY,
    Data,
    Engine,
    Grid,
    Loop,
    MapInit,
    Result,
    Schedule,
    Steady,
    Steps,
    Tolerance,
    as_data,
    as_matrix,
    check_float,
    check_int,
    check_max_iter,
    check_seed,
    fitted,
    iterate,
    label_sums,
    nearest,
    numba_kernels,
    softmin,
    sq_norms,
    sqdist,
    start_fixed,
    weighted_mean,
)


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
    R = _grid(grid)
    epochs = check_int(epochs, "epochs", 1)
    s0, s1 = _sigmas(sigma, sigma_end, R)
    lr = check_float(lr, "lr", 0.0, strict=True)
    lr_end = check_float(lr_end, "lr_end", 0.0, strict=True)
    if max(lr, lr_end) > 1:
        raise ValueError("lr and lr_end must be at most 1")
    rng = np.random.default_rng(check_seed(seed))
    data, W = _setup(X, R, init)

    def epoch(data, W, _, e):
        X, n = data.X, len(data.X)
        steps = epochs * n
        order = rng.permutation(n) if shuffle else np.arange(n)
        if kernels:
            return kernels.som_epoch(X, W, R, order, e * n, steps, s0, s1, lr, lr_end), None, None
        W = W.copy()
        # The grid differences are divided by s before squaring, as in
        # batch_som: a tiny width keeps h = 1 for the winner and sends the
        # others to 0 through an overflow to inf, as in the limit.
        with np.errstate(over="ignore"):
            for j, i in enumerate(order):
                f = (e * n + j) / (steps - 1) if steps > 1 else 0.0
                s, eta = _geometric(s0, s1, f), _geometric(lr, lr_end, f)
                diff = X[i] - W
                bmu = np.einsum("ij,ij->i", diff, diff).argmin()
                g = (R - R[bmu]) / s
                h = np.exp(-0.5 * np.einsum("ij,ij->i", g, g))
                W += (eta * h)[:, None] * diff
        return W, None, None

    return (yield from iterate(data, W, epoch, stopping=Schedule(epochs), view=_map(R)))


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
    R = _grid(grid)
    epochs = check_int(epochs, "epochs", 1)
    s0, s1 = _sigmas(sigma, sigma_end, R)
    data, W = _setup(X, R, init)
    gr, gc = np.arange(rows), np.arange(cols)

    def step(data, W, _, t):
        labels = nearest(data.X, W, data.xx)
        s = _geometric(s0, s1, t / (epochs - 1)) if epochs > 1 else s0
        # exp(-(d / s)^2 / 2): finite for any width, the identity in its limit.
        with np.errstate(over="ignore"):
            Kr = np.exp(-0.5 * ((gr[:, None] - gr) / s) ** 2)
            Kc = np.exp(-0.5 * ((gc[:, None] - gc) / s) ** 2)
        sums, counts = label_sums(data.X, labels, len(W))
        num = Kc @ (Kr @ sums.reshape(rows, -1)).reshape(rows, cols, -1)
        den = (Kr @ counts.reshape(rows, cols) @ Kc.T).ravel()
        out = W.copy()
        ok = den > 0
        out[ok] = num.reshape(len(W), -1)[ok] / den[ok, None]
        return out, labels, None

    return (yield from iterate(data, W, step, stopping=Schedule(epochs), view=_map(R)))


def som_olp(
    X: ArrayLike,
    grid: Grid = (10, 10),
    *,
    lam: float,
    gamma: float,
    init: MapInit = "pca",
    pca_scale: float = 2.0,
    max_iter: int | None = 100,
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
    R = _grid(grid)
    assign = _olp(R, lam, gamma)
    pca_scale = check_float(pca_scale, "pca_scale", 0.0, strict=True)
    max_iter, tol = check_max_iter(max_iter), check_float(tol, "tol", 0.0)
    data, W = _setup(X, R, init, pca_scale)

    def step(data, W, P, t):
        P, value = assign(data, W, P)
        return weighted_mean(data.X, P, W), P, value

    stopping = Tolerance(tol, max_iter)
    return (
        yield from iterate(data, W, step, stopping=stopping, view=_olp_view(R), keep=_same_points)
    )


def apply_som(X: ArrayLike, centers: ArrayLike, grid: Grid) -> Steps:
    """The best-matching units of X among the fixed prototypes ``centers`` of the
    units at ``grid`` (``ubukit.apply.som`` and ``ubukit.apply.batch_som``).

    ``grid`` is (rows, cols) or a (K, Q) array, as the map's
    ``unit_coordinates``; ``embedding`` holds the coordinates of each
    point's unit.
    """
    R = _grid(grid)
    data, W, C = start_fixed(X, centers, len(R))

    def step(data, W, state, t):
        return W, None, None  # the view finds the best-matching units

    return (yield from iterate(data, W, step, stopping=Schedule(1), view=_map(R), fixed=C))


apply_batch_som = apply_som


def apply_som_olp(
    X: ArrayLike,
    centers: ArrayLike,
    grid: Grid,
    *,
    lam: float,
    gamma: float,
    max_iter: int | None = 1000,
    tol: float = 1e-6,
) -> Steps:
    """The memberships and latent positions of X on a fixed SOM-OLP map
    (``ubukit.apply.som_olp``).

    ``centers`` are the prototypes of the units at ``grid``; give the lam and
    gamma they were trained with. The memberships start from the distances
    alone, as a training run's first iteration does, and are iterated with
    the prototypes fixed until no membership changes by more than ``tol``,
    or ``max_iter`` iterations (more than training's default: they settle
    slowly for some settings). Each point's update depends on it alone, but
    the rule stops all points together, so a point's result can depend on
    the points applied with it, mostly for one midway between two units.
    The latent positions of earlier memberships pull later ones, so the
    iteration can settle elsewhere than training left the same points.
    """
    R = _grid(grid)
    assign = _olp(R, lam, gamma)
    max_iter, tol = check_max_iter(max_iter), check_float(tol, "tol", 0.0)
    data, W, C = start_fixed(X, centers, len(R))

    def step(data, W, P, t):
        P, value = assign(data, W, P)
        return W, P, value

    return (
        yield from iterate(
            data,
            W,
            step,
            stopping=Steady(tol, max_iter),
            view=_olp_view(R),
            keep=_same_points,
            fixed=C,
        )
    )


def _olp(R: np.ndarray, lam: float, gamma: float) -> Callable[..., tuple[np.ndarray, float]]:
    """SOM-OLP's assignment for the units at R: ``assign(data, W, P)`` gives the
    memberships, from the latent positions P R of the previous ones (none at
    first), with the objective at them."""
    lam = check_float(lam, "lam", 0.0, strict=True)
    gamma = check_float(gamma, "gamma", 0.0)
    # The latent costs gamma ||v - r_j||^2 come from centered unit coordinates,
    # as distances come from centered data: rounding then follows the extent
    # of the units, not their distance from the origin. They must stay finite
    # beside the data's costs, which the scale contract keeps below about
    # 4e300. A latent position v lies among the units, so ||v - r_j||^2 is at
    # most their squared extent, bounded by four times their largest squared
    # norm once centered.
    Rc = R - R.mean(axis=0)
    rr = sq_norms(Rc)
    extent = 4.0 * float(rr.max())
    if gamma * extent > MAX_SQ_NORM:
        raise ValueError(
            "gamma is too large for the unit coordinates: gamma times their squared extent "
            f"({extent:g}) must be at most {MAX_SQ_NORM:g}"
        )

    def assign(data, W, P):
        cost = sqdist(data.X, W, data.xx)
        if P is not None:
            cost += gamma * sqdist(P @ Rc, Rc, cc=rr)
        return softmin(cost, lam, "lam")  # the objective: the sum of the soft minima

    return assign


def _same_points(P: np.ndarray, data: Data) -> np.ndarray | None:
    """SOM-OLP's memberships give each point's latent position for the next
    iteration. Points are known only by their rows, so they carry over while
    the number of rows stays the same, row i being the same point."""
    return P if len(P) == len(data.X) else None


def _olp_view(R: np.ndarray) -> Callable[[Loop], Result]:
    """The view of SOM-OLP: the memberships, and the latent positions they give."""

    def view(loop: Loop) -> Result:
        P = loop.state.copy()  # the next step reads it
        return fitted(loop, P.argmax(axis=1), P, P @ R, R.copy())

    return view


def _shape(grid: Any) -> tuple[int, int]:
    if np.ndim(grid) != 1 or len(grid) != 2:
        raise ValueError("grid must be (rows, cols)")
    return check_int(grid[0], "rows", 1), check_int(grid[1], "cols", 1)


def _grid(grid: Grid) -> np.ndarray:
    """Unit coordinates: (rows, cols) -> rectangular grid, or an explicit (K, Q) array.

    The run owns them: a given array is copied, so changing it later changes
    neither the run nor its Results.
    """
    if np.ndim(grid) == 2:
        return as_matrix(grid, "grid").copy()
    rows, cols = _shape(grid)
    r, c = np.divmod(np.arange(rows * cols), cols)
    return np.column_stack((r, c)).astype(np.float64)


def _setup(
    X: ArrayLike, R: np.ndarray, init: MapInit, pca_scale: float = 2.0
) -> tuple[Data, np.ndarray]:
    """The data and the initial prototypes of the units at R (centered like the data)."""
    if isinstance(init, str) and init != "pca":
        raise ValueError("init must be 'pca' or a (n_units, n_features) array")
    data = as_data(X)
    if isinstance(init, str):
        W = _pca_init(data.X, R, pca_scale)
        # A start made here keeps the scale contract that a given one must.
        if not sq_norms(W).max() < MAX_SQ_NORM:
            raise ValueError("pca_scale is too large for X: the start is beyond float64 distances")
        return data, W
    W = as_matrix(init, "init")
    if W.shape != (len(R), data.X.shape[1]):
        raise ValueError(f"init must have shape ({len(R)}, {data.X.shape[1]})")
    return data, W - data.mean


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


def _geometric(a: float, b: float, f: float) -> float:
    """The point f of the geometric schedule from a to b, as a^(1-f) b^f.

    No ratio b / a is formed, which overflows or underflows for far-apart
    ends; the factors stay between 1 and the ends, and f = 0, 1 give a, b.
    """
    return a ** (1.0 - f) * b**f


def _sigmas(sigma: float | None, sigma_end: float, R: np.ndarray) -> tuple[float, float]:
    """Neighborhood width schedule endpoints; default start is half the grid extent."""
    if sigma is None:
        sigma = float(np.ptp(R, axis=0).max()) / 2 or 1.0
    return (
        check_float(sigma, "sigma", 0.0, strict=True),
        check_float(sigma_end, "sigma_end", 0.0, strict=True),
    )


def _map(R: np.ndarray) -> Callable[[Loop], Result]:
    """The view of a map: the best-matching units of the prototypes reached."""

    def view(loop: Loop) -> Result:
        labels = nearest(loop.data.X, loop.V, loop.data.xx)
        return fitted(loop, labels, embedding=R[labels], unit_coordinates=R.copy())

    return view
