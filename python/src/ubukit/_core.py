"""Shared numerics: validation, distances, initialization and the engine.

Every iterative method in UbuKit runs in one loop, ``iterate``, which repeats
a step until the prototypes stop moving. Nearly all of them use the standard
step ``lloyd``,

    D = ||x_i - v_c||^2  ->  U = assign(D)  ->  V = update(U),

and differ only in ``assign`` (and, for SOMs, ``update``); a method whose
iteration has another shape (the online SOM) supplies its own step. The data
are an input of every step (``Data``: centered rows, which keep the
Gram-identity distances accurate), so a run can go on with new data.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import wraps
from numbers import Integral, Real
from types import ModuleType
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

# Public argument types.
type Init = Literal["k-means++"] | ArrayLike  # k-means++ seeding or (k, D) centers
type MapInit = Literal["pca"] | ArrayLike  # principal-plane start or (K, D) prototypes
type Grid = tuple[int, int] | ArrayLike  # (rows, cols) or (K, Q) unit coordinates
type Engine = Literal["numpy", "numba"]  # NumPy reference or its compiled Numba kernel

TINY = np.finfo(np.float64).tiny
# "Ordinary scale": row norms below 1e150 keep every squared distance between
# data, prototypes and grid points (~1e301 at most) far from float64 overflow,
# and some feature spanning at least 1e-150 (unless all rows are equal) keeps
# the squared distances of the data from underflowing to zero.
MAX_SQ_NORM = 1e300
MIN_SPAN = 1e-150
MAX_INT = 2**53 - 1  # JavaScript's Number.MAX_SAFE_INTEGER


@dataclass(frozen=True, slots=True, eq=False)
class Result:
    """A fitted clustering or map.

    For clusterings and SOM-OLP, ``labels`` and ``membership`` come from the
    last assignment step, i.e. for the prototypes before the final update;
    they coincide at a fixed point. For ``som`` and ``batch_som``, ``labels``
    are the best-matching units of the final prototypes.

    Attributes:
        centers: (K, D) prototypes in input coordinates.
        labels: (N,) index of the strongest membership (nearest prototype for k-means,
            best-matching unit for ``som`` and ``batch_som``).
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


def as_matrix(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Return X as a finite, non-empty, C-contiguous float64 matrix of ordinary scale."""
    A = np.ascontiguousarray(X, dtype=np.float64)
    if A.ndim != 2 or 0 in A.shape:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if not np.isfinite(A).all():
        raise ValueError(f"{name} must contain only finite values")
    if not sq_norms(A).max() < MAX_SQ_NORM:
        raise ValueError(f"{name} is too large in scale for float64 distances; standardize it")
    span = float((A.max(axis=0) - A.min(axis=0)).max())
    if 0 < span < MIN_SPAN:
        raise ValueError(f"{name} is too small in scale for float64 distances; standardize it")
    return A


def check_int(value: Any, name: str, low: int, high: int | None = None) -> int:
    """Validate an integer in [low, high]; like JavaScript, never beyond +-(2**53 - 1)."""
    v = int(value) if isinstance(value, Integral) and not isinstance(value, bool) else None
    top = MAX_INT if high is None else min(high, MAX_INT)
    if v is None or not max(low, -MAX_INT) <= v <= top:
        bound = f">= {low}" if high is None else f"in [{low}, {high}]"
        raise ValueError(f"{name} must be an integer {bound}")
    return v


def check_float(value: Any, name: str, low: float = -np.inf, *, strict: bool = False) -> float:
    """Validate a finite real number >= low (> low when strict)."""
    v = float(value) if isinstance(value, Real) and not isinstance(value, bool) else math.nan
    if not (math.isfinite(v) and (v > low if strict else v >= low)):
        raise ValueError(f"{name} must be a finite number {'>' if strict else '>='} {low}")
    return v


def numba_kernels(engine: Engine) -> ModuleType | None:
    """None for ``engine="numpy"``; for ``"numba"``, the kernels, importing Numba on first use."""
    if engine == "numpy":
        return None
    if engine != "numba":
        raise ValueError("engine must be 'numpy' or 'numba'")
    try:
        from . import _numba
    except ImportError as error:
        raise ImportError("engine='numba' needs Numba: pip install 'ubukit[numba]'") from error
    return _numba


def sq_norms(X: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", X, X)


def sqdist(X: np.ndarray, C: np.ndarray, xx=None, cc=None) -> np.ndarray:
    """Squared Euclidean distances (N, K) via the Gram identity, clipped at zero."""
    D = X @ (-2.0 * C).T  # scaling C rather than the (N, K) result; doubling is exact
    D += (sq_norms(X) if xx is None else xx)[:, None]
    D += sq_norms(C) if cc is None else cc
    return np.maximum(D, 0.0, out=D)


def nearest(X: np.ndarray, C: np.ndarray, xx=None) -> np.ndarray:
    """Index of the nearest row of C for each row of X, ties to the lowest index.

    The same arithmetic as ``sqdist(X, C).argmin(axis=1)``, in blocks of rows
    small enough to stay in cache, which is up to twice as fast.
    """
    xx = sq_norms(X) if xx is None else xx
    cc = sq_norms(C)
    block = max(1, 2**15 // len(C))
    out = np.empty(len(X), np.intp)
    for start in range(0, len(X), block):
        rows = slice(start, start + block)
        out[rows] = sqdist(X[rows], C, xx[rows], cc).argmin(axis=1)
    return out


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
    """Greedy k-means++ seeding (Arthur & Vassilvitskii, 2007).

    Each new seed is the best, by the resulting sum of squared distances to the
    nearest seed, of 2 + floor(ln k) candidates drawn in proportion to that
    squared distance; a single draw too often puts two seeds in one cluster.
    """
    n = len(X)
    xx = sq_norms(X)
    trials = 2 + int(np.log(k))
    chosen = [int(rng.integers(n))]
    closest = sqdist(X, X[chosen], xx)[:, 0]
    for _ in range(1, k):
        total = closest.sum()
        if total > 0:
            draws = np.searchsorted(np.cumsum(closest), rng.random(trials) * total, side="right")
            candidates = np.minimum(draws, n - 1)
        else:
            candidates = rng.integers(n, size=trials)
        D = np.minimum(closest[:, None], sqdist(X, X[candidates], xx))
        best = int(D.sum(axis=0).argmin())
        chosen.append(int(candidates[best]))
        closest = D[:, best]
    return X[chosen].copy()


@dataclass(frozen=True, slots=True, eq=False)
class Data:
    """The input of an iteration: the rows centered on their mean, with what steps reuse.

    Attributes:
        X: (N, D) centered rows (distances are accurate on centered data).
        mean: (D,) the mean subtracted from them.
        xx: (N,) squared norms of the rows of X.
        radius: RMS norm of the rows of X, the scale of ``tol``.
    """

    X: np.ndarray
    mean: np.ndarray
    xx: np.ndarray
    radius: float


def as_data(X: ArrayLike, n_features: int | None = None) -> Data:
    """Validate and center X as the input of an iteration."""
    X = as_matrix(X)
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X must have {n_features} features, like the data the run started on")
    mean = X.mean(axis=0)
    X = X - mean
    xx = sq_norms(X)
    return Data(X, mean, xx, float(np.sqrt(xx.mean())))


def start(
    X: ArrayLike,
    k: int,
    init: Init,
    seed: int | None,
    prepare: Callable[..., Data] = as_data,
) -> tuple[Data, np.ndarray]:
    """The data and the initial centers (centered like the data) of a clustering."""
    data = prepare(X)
    k = check_int(k, "k", 1, len(data.X))
    if isinstance(init, str):
        if init != "k-means++":
            raise ValueError("init must be 'k-means++' or a (k, n_features) array")
        V = kmeans_plus_plus(data.X, k, np.random.default_rng(seed))
        # Start from the seeds' cell means: a center on a data point gives that
        # point full weight, and for large m stalls there (docs/algorithms.md).
        return data, label_mean(data.X, nearest(data.X, V, data.xx), V)
    V = as_matrix(init, "init")
    if V.shape != (k, data.X.shape[1]):
        raise ValueError(f"init must have shape ({k}, {data.X.shape[1]})")
    return data, V - data.mean


def check_max_iter(value: Any) -> int | None:
    """An iteration limit >= 1, or None for none (a run fed new data until it converges)."""
    return None if value is None else check_int(value, "max_iter", 1)


# A step maps (data, prototypes, state, t) to (new prototypes, new state,
# objective value or None); the state starts as None and is whatever the
# method carries. Steps read the data from their argument, never keep it.
Step = Callable[[Data, np.ndarray, Any, int], tuple[np.ndarray, Any, float | None]]
Assign = Callable[[Data, np.ndarray, Any, int], np.ndarray]
Update = Callable[[Data, np.ndarray, np.ndarray, int], np.ndarray]
Objective = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True, slots=True)
class Progress:
    """Yielded after every iteration (epoch for maps) by the generators in ``ubukit.steps``.

    ``result()`` returns the Result the run would return had it stopped at
    this iteration; it costs nothing unless called, stays valid as the run
    goes on, and shares no arrays with the run, so changing it leaves the
    iterations that follow unchanged. Sending new data into the generator
    (``run.send(X)``) runs the following iterations on them.
    """

    iteration: int
    result: Callable[[], Result]


class Loop(NamedTuple):
    """What the loop has reached; a method's ``view`` turns it into its Result."""

    data: Data
    V: np.ndarray
    state: Any
    n_iter: int
    converged: bool
    history: np.ndarray


# A fitting generator: it yields a Progress per iteration and returns the Result.
Steps = Generator[Progress, None, Result]
# The generators behind the public fitting functions, by name (ubukit.steps).
STEPS: dict[str, Callable[..., Steps]] = {}


def stepwise[**P](generator: Callable[P, Steps]) -> Callable[P, Result]:
    """Make a fitting generator a function that runs it to the end, and list it in STEPS."""
    STEPS[generator.__name__] = generator

    @wraps(generator)
    def run(*args: P.args, **kwargs: P.kwargs) -> Result:
        steps = generator(*args, **kwargs)
        while True:
            try:
                next(steps)
            except StopIteration as end:
                return end.value

    # help() shows the generator's parameters with the Result it returns.
    run.__signature__ = inspect.signature(generator).replace(  # type: ignore[attr-defined]
        return_annotation=Result
    )
    return run


def iterate(
    data: Data,
    V: np.ndarray,
    step: Step,
    *,
    max_iter: int | None,
    tol: float | None,
    view: Callable[[Loop], Result],
    prepare: Callable[..., Data] = as_data,
    keep: Callable[[Any, Data], Any] | None = None,
) -> Steps:
    """The one loop: repeat ``step`` until the prototypes stop moving, or max_iter.

    The data are an input of every iteration: rows sent into the generator
    are prepared by ``prepare`` and used from the next iteration on. The
    prototypes carry over (shifted to the new data's centering), and so does
    the state that ``keep(state, data)`` returns; without ``keep`` the state
    starts again, as caches of the old data must. The schedule (``t``) goes on.

    The run ends when no prototype coordinate moves more than ``tol`` times
    the RMS radius of the data and no new data arrive (``tol=0``: an exact
    fixed point; ``tol=None``: a fixed schedule), or after ``max_iter``
    iterations (``None``: no limit). It yields a Progress after every step
    and returns ``view`` of the final Loop, the method's Result; a Progress
    builds its Result with the same ``view``, which is why steps never modify
    a state they have returned and views copy the state they put in a Result.
    """
    state = None
    history: list[float] = []
    t = 0
    while True:
        V_prev = V
        V, state, value = step(data, V, state, t)
        t += 1
        if value is not None:
            history.append(value)
        limit = None if tol is None else tol * data.radius
        converged = limit is not None and float(np.max(np.abs(V - V_prev))) <= limit
        result = _result(view, Loop(data, V, state, t, converged, np.empty(0)), history)
        rows = yield Progress(t, result)
        if t == max_iter:
            return result()
        if rows is not None:
            new = prepare(rows, V.shape[1])
            V = V + (data.mean - new.mean)
            state = keep(state, new) if keep else None
            data = new
        elif converged:
            return result()


def _result(view, loop: Loop, history: list[float]) -> Callable[[], Result]:
    """The Result of ``loop`` with the history so far, built on demand."""
    n = len(history)
    return lambda: view(loop._replace(history=np.asarray(history[:n], dtype=float)))


def lloyd(assign: Assign, update: Update, objective: Objective | None = None) -> Step:
    """The standard step: D = ||x - v||^2, U = assign(data, D, U_prev, t), V = update(...).

    ``update(data, U, V, t)`` gives the new prototypes. ``assign`` returns
    integer labels or float memberships, which are the step's state; the
    optional ``objective(D, U)`` gives one history value.
    """

    def step(data, V, U, t):
        D = sqdist(data.X, V, data.xx)
        U = assign(data, D, U, t)
        return update(data, U, V, t), U, None if objective is None else objective(D, U)

    return step
