"""Shared numerics: validation, distances, initialization and the engine.

Every iterative method in UbuKit runs in one loop, ``iterate``, which repeats
the method's step until the prototypes stop moving. The fuzzy and rough
c-means use the standard step ``lloyd``,

    D = ||x_i - v_c||^2  ->  U = assign(D)  ->  V = update(U),

and differ only in ``assign`` and ``update``; k-means (with Hamerly's bounds)
and the maps, whose iterations have other shapes, supply their own steps. The
data are an input of every step (``Data``: centered rows, which keep the
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
type StopReason = Literal["tol", "schedule", "limit"]  # the stopping rule that holds

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
        stop_reason: the stopping rule that holds at this iteration, under the run's
            own settings: ``"tol"`` (no prototype coordinate moved more than ``tol``
            times the RMS radius of the data; with ``tol=0``, none moved at all; for
            ``apply.som_olp``, no membership changed by more than ``tol``),
            ``"schedule"`` (``som`` and ``batch_som``: every epoch has run; the
            other ``apply`` functions: their one assignment has run) or
            ``"limit"`` (``max_iter`` reached without ``"tol"``); ``None`` in a
            Progress result where none holds. It says nothing about the quality of
            the result.
        history: objective value per iteration; empty for ``rcm``, ``rmcm``, ``som``
            and ``batch_som``, which have none.
        embedding: (N, Q) map coordinates for SOMs, otherwise ``None``:
            ``unit_coordinates[labels]`` for ``som`` and ``batch_som``,
            ``membership @ unit_coordinates`` for ``som_olp``.
        unit_coordinates: (K, Q) coordinates of a map's units as the run used them,
            not normalized: (row, col) on a rectangular grid, unit j = row * cols + col,
            or the coordinates given; ``centers[j]`` is the prototype of unit j.
            ``None`` for clusterings.
    """

    centers: np.ndarray
    labels: np.ndarray
    membership: np.ndarray | None
    n_iter: int
    stop_reason: StopReason | None
    history: np.ndarray
    embedding: np.ndarray | None = None
    unit_coordinates: np.ndarray | None = None


def as_matrix(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Return X as a finite, non-empty, C-contiguous float64 matrix of ordinary scale.

    Its values must be numbers (integers or floats): NumPy would convert
    booleans and numeric strings, and ``None`` would become a missing value,
    so they are refused, as in JavaScript.
    """
    A = np.asarray(X)
    if A.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold numbers, not {A.dtype}")
    A = np.ascontiguousarray(A, dtype=np.float64)
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
        bound = "" if low == -np.inf else f" {'>' if strict else '>='} {low}"
        raise ValueError(f"{name} must be a finite number{bound}")
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


def sqdist(
    X: np.ndarray, C: np.ndarray, xx: np.ndarray | None = None, cc: np.ndarray | None = None
) -> np.ndarray:
    """Squared Euclidean distances (N, K) via the Gram identity, clipped at zero."""
    D = X @ (-2.0 * C).T  # scaling C rather than the (N, K) result; doubling is exact
    D += (sq_norms(X) if xx is None else xx)[:, None]
    D += sq_norms(C) if cc is None else cc
    return np.maximum(D, 0.0, out=D)


def nearest(X: np.ndarray, C: np.ndarray, xx: np.ndarray | None = None) -> np.ndarray:
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


def softmin(C: np.ndarray, temperature: float) -> tuple[np.ndarray, float]:
    """Memberships p_ij proportional to exp(-c_ij / T), and the sum of the rows' soft minima.

    The soft minimum c_min - T log sum_j exp(-(c_ij - c_min) / T) equals
    sum_j p_ij c_ij + T p_ij log p_ij at these memberships, the
    entropy-regularized objective; it comes from the row minima and sums that
    the normalization computes. Each row's minimum is subtracted before the
    division, so every exponent lies in [-inf, 0] and any T > 0, however
    small against the costs, gives finite memberships: the hard ones of its
    limit.
    """
    low = C.min(axis=1, keepdims=True)
    P = C - low
    with np.errstate(over="ignore"):  # exponents of -inf for a tiny T, as in the limit
        P /= -temperature
    np.exp(P, out=P)
    total = P.sum(axis=1, keepdims=True)
    P /= total
    return P, float(np.sum(low) - temperature * np.sum(np.log(total)))


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
    A = as_matrix(X)
    if n_features is not None and A.shape[1] != n_features:
        raise ValueError(f"X must have {n_features} features, like the data the run started on")
    mean = A.mean(axis=0)
    A = A - mean
    xx = sq_norms(A)
    return Data(A, mean, xx, float(np.sqrt(xx.mean())))


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


def start_fixed(
    X: ArrayLike,
    centers: ArrayLike,
    units: int | None = None,
    prepare: Callable[..., Data] = as_data,
) -> tuple[Data, np.ndarray, np.ndarray]:
    """The data, centered prototypes and fixed prototypes of an application.

    The prototypes are copied, so the run owns them, and kept as given: the
    loop centers them afresh for new data and puts a copy in every Result,
    so they never drift. ``units``, for a map, is the number of its units.
    """
    C = as_matrix(centers, "centers").copy()
    if units is not None and len(C) != units:
        raise ValueError(f"centers must have {units} rows, one per unit")
    data = prepare(X)
    if C.shape[1] != data.X.shape[1]:
        raise ValueError(f"centers must have {data.X.shape[1]} columns, like X")
    return data, C - data.mean, C


def check_max_iter(value: Any) -> int | None:
    """An iteration limit >= 1, or None for none (a run fed new data until it converges)."""
    return None if value is None else check_int(value, "max_iter", 1)


# A step maps (data, prototypes, state, t) to (new prototypes, new state,
# objective value or None); the state starts as None and is whatever the
# method carries. Steps read the data from their argument, never keep it.
Step = Callable[[Data, np.ndarray, Any, int], tuple[np.ndarray, Any, float | None]]
Assign = Callable[[Data, np.ndarray], tuple[np.ndarray, float | None]]
Update = Callable[[Data, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class Progress:
    """Yielded after every iteration (epoch for maps) by the generators in ``ubukit.steps``.

    ``result()`` returns the Result of the run at this iteration: the one it
    would return had it stopped here, with the ``stop_reason`` that holds
    here under the run's own settings (``None`` while none does and the run
    goes on). It costs nothing unless called, stays valid as the run goes
    on, and shares no arrays with the run, so changing it leaves the
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
    stop_reason: StopReason | None
    history: np.ndarray
    fixed: np.ndarray | None = None  # the prototypes of an application, as given


def fitted(
    loop: Loop,
    labels: np.ndarray,
    membership: np.ndarray | None = None,
    embedding: np.ndarray | None = None,
    unit_coordinates: np.ndarray | None = None,
) -> Result:
    """The Result of ``loop`` with the labels, memberships, embedding and unit
    coordinates of its method (arrays the view has made or copied for it)."""
    centers = loop.V + loop.data.mean if loop.fixed is None else loop.fixed.copy()
    return Result(
        centers,
        labels,
        membership,
        loop.n_iter,
        loop.stop_reason,
        loop.history,
        embedding,
        unit_coordinates,
    )


@dataclass(frozen=True, slots=True)
class Tolerance:
    """A run that ends when no prototype coordinate moves more than ``tol``
    times the RMS radius of the data, or after ``max_iter`` iterations
    (``None``: no limit)."""

    tol: float
    max_iter: int | None


@dataclass(frozen=True, slots=True)
class Steady:
    """A run that ends when no membership changes by more than ``tol`` from one
    iteration to the next (never at the first, which has nothing to compare),
    or after ``max_iter`` iterations: prototypes held fixed while memberships
    still move (SOM-OLP applied)."""

    tol: float
    max_iter: int | None


@dataclass(frozen=True, slots=True)
class Schedule:
    """A run of a fixed schedule of ``length`` iterations (the maps' epochs,
    or the one assignment of an application)."""

    length: int


# How a method's runs end; the loop applies it.
type Stopping = Tolerance | Steady | Schedule


def reason_at(
    stopping: Stopping,
    t: int,
    V: np.ndarray,
    V_prev: np.ndarray,
    state: Any,
    state_prev: Any,
    radius: float,
) -> StopReason | None:
    """The stopping rule that holds after iteration t ("tol" before "limit")."""
    if isinstance(stopping, Schedule):
        return "schedule" if t == stopping.length else None
    if isinstance(stopping, Steady):
        held = state_prev is not None and np.max(np.abs(state - state_prev)) <= stopping.tol
    else:
        held = np.max(np.abs(V - V_prev)) <= stopping.tol * radius
    if held:
        return "tol"
    return "limit" if t == stopping.max_iter else None


def is_last(stopping: Stopping, t: int) -> bool:
    """Whether iteration t ends the run whatever the data: the schedule's end or the limit."""
    return t == (stopping.length if isinstance(stopping, Schedule) else stopping.max_iter)


# A fitting generator (ubukit.steps): it yields a Progress per iteration,
# takes new rows or None by send, and returns the Result.
Steps = Generator[Progress, ArrayLike | None, Result]


def stepwise[**P](
    generator: Callable[P, Steps], name: str | None = None, module: str = "ubukit"
) -> Callable[P, Result]:
    """The public function of a generator, ``module.name``: it runs the generator to the end."""

    @wraps(generator)
    def run(*args: P.args, **kwargs: P.kwargs) -> Result:
        steps = generator(*args, **kwargs)
        while True:
            try:
                next(steps)
            except StopIteration as end:
                return end.value

    # help() shows the generator's parameters with the Result it returns, and
    # pickle finds the function under its public name, module.name.
    run.__signature__ = inspect.signature(generator).replace(  # type: ignore[attr-defined]
        return_annotation=Result
    )
    run.__module__ = module
    run.__name__ = run.__qualname__ = name or generator.__name__
    return run


def iterate(
    data: Data,
    V: np.ndarray,
    step: Step,
    *,
    stopping: Stopping,
    view: Callable[[Loop], Result],
    prepare: Callable[..., Data] = as_data,
    keep: Callable[[Any, Data], Any] | None = None,
    fixed: np.ndarray | None = None,
) -> Steps:
    """The one loop: repeat ``step`` until the method's stopping rule ends the run.

    The data are an input of every iteration: rows sent into the generator
    are prepared by ``prepare`` and used from the next iteration on. The
    prototypes carry over (shifted to the new data's centering), and so does
    the state that ``keep(state, data)`` returns; without ``keep`` the state
    starts again, as caches of the old data must. The schedule (``t``) goes on.
    An application passes its prototypes as given (``fixed``), which its steps
    never move: they are centered afresh for new data and copied into every
    Result, so they never drift.

    ``stopping`` is a ``Tolerance``, whose run ends when no prototype
    coordinate moves more than ``tol`` times the RMS radius of the data and
    no new data arrive (``tol=0``: the prototypes did not move at all), or
    after ``max_iter`` iterations; a ``Steady``, which measures the change of
    the memberships instead; or a ``Schedule``, whose run ends when its
    iterations are complete. Each Loop records the rule that holds at its
    iteration (``stop_reason``). The loop yields a Progress after every step
    and returns ``view`` of the final Loop, the method's Result; a Progress
    builds its Result with the same ``view``, which is why steps never modify
    a state they have returned and views copy the state they put in a Result.
    """
    state = None
    history: list[float] = []
    t = 0
    while True:
        V_prev, state_prev = V, state
        V, state, value = step(data, V, state, t)
        t += 1
        if value is not None:
            history.append(value)
        reason = reason_at(stopping, t, V, V_prev, state, state_prev, data.radius)
        loop = Loop(data, V, state, t, reason, np.empty(0), fixed)
        result = _result(view, loop, history)
        rows = yield Progress(t, result)
        if is_last(stopping, t):
            return result()
        if rows is not None:
            new = prepare(rows, V.shape[1])
            V = V + (data.mean - new.mean) if fixed is None else fixed - new.mean
            state = keep(state, new) if keep else None
            data = new
        elif reason == "tol":
            return result()


def _result(
    view: Callable[[Loop], Result], loop: Loop, history: list[float]
) -> Callable[[], Result]:
    """The Result of ``loop`` with the history so far, built on demand."""
    n = len(history)
    return lambda: view(loop._replace(history=np.asarray(history[:n], dtype=float)))


def means(data: Data, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """The standard update: the means of the data weighted by the memberships."""
    return weighted_mean(data.X, U, V)


def unchanged(data: Data, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """The update of an application: the prototypes stay where they are."""
    return V


def lloyd(assign: Assign, update: Update = means) -> Step:
    """The standard step: D = ||x - v||^2, (U, J) = assign(data, D), V = update(data, U, V).

    ``assign`` gives the memberships U for these distances, the step's state
    (which the method's view reads), and the method's objective J at them,
    one history value (None for a method without one). The memberships
    minimize the objective for the given prototypes, and the quantities that
    compute them give its value, so it costs almost nothing. ``update``
    defaults to the means weighted by U.
    """

    def step(data, V, _, t):
        U, value = assign(data, sqdist(data.X, V, data.xx))
        return update(data, U, V), U, value

    return step
