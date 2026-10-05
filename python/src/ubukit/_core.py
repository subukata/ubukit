"""Shared numerics: validation, distances, initialization and the engine.

Every iterative method in UbuKit runs in one loop, ``iterate``, which repeats
a step until the prototypes stop moving. Nearly all of them use the standard
step ``lloyd``,

    D = ||x_i - v_c||^2  ->  U = assign(D)  ->  V = update(U),

and differ only in ``assign`` (and, for SOMs, ``update``); a method whose
iteration has another shape (the online SOM) supplies its own step. Data are
centered once before fitting, which keeps the Gram-identity distances accurate.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import wraps
from numbers import Integral, Real
from typing import Any, NamedTuple

import numpy as np
from scipy import sparse

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
    span = float((A.max(axis=0) - A.min(axis=0)).max())
    if 0 < span < MIN_SPAN:
        raise ValueError(f"{name} is too small in scale for float64 distances; standardize it")
    return A


def check_int(value, name: str, low: int, high: int | None = None) -> int:
    """Validate an integer in [low, high]; like JavaScript, never beyond +-(2**53 - 1)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not max(low, -MAX_INT) <= value <= (MAX_INT if high is None else min(high, MAX_INT))
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
        # Start from the seeds' cell means, so a center sits exactly on a data
        # point (which then gets full weight in fuzzy updates) only when its
        # cell holds that point alone.
        V = label_mean(Xc, sqdist(Xc, V).argmin(axis=1), V)
    else:
        V = as_matrix(init, "init")
        if V.shape != (k, X.shape[1]):
            raise ValueError(f"init must have shape ({k}, {X.shape[1]})")
        V = V - mean
    return Xc, mean, V


# A step maps (prototypes, state, t) to (new prototypes, new state, objective
# value or None); the state starts as None and is whatever the method carries.
Step = Callable[[np.ndarray, Any, int], tuple[np.ndarray, Any, float | None]]
Assign = Callable[[np.ndarray, np.ndarray | None, int], np.ndarray]
Update = Callable[[np.ndarray, np.ndarray, int], np.ndarray]
Objective = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True, slots=True)
class Progress:
    """Yielded after every iteration (epoch for maps) by the generators in ``ubukit.steps``.

    ``result()`` returns the Result the run would return had it stopped at
    this iteration; it costs nothing unless called and stays valid as the run
    goes on.
    """

    iteration: int
    result: Callable[[], Result]


class Loop(NamedTuple):
    """What the loop has reached; a method's ``view`` turns it into its Result."""

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
    run.__signature__ = inspect.signature(generator).replace(return_annotation=Result)
    return run


def iterate(
    X: np.ndarray,
    V: np.ndarray,
    step: Step,
    *,
    max_iter: int,
    tol: float | None,
    view: Callable[[Loop], Result],
) -> Steps:
    """The one loop: repeat ``step`` until the prototypes stop moving, or max_iter.

    The loop stops when no prototype coordinate moves more than ``tol`` times
    the RMS radius of X; ``tol=0`` therefore means an exact fixed point and
    ``tol=None`` runs a fixed schedule. It yields a Progress after every step
    and returns ``view`` of the final Loop, the method's Result; a Progress
    builds its Result with the same ``view``, which is why steps never modify
    a state they have returned.
    """
    limit = None if tol is None else tol * float(np.sqrt(sq_norms(X).mean()))
    state = None
    history: list[float] = []
    for t in range(max_iter):
        V_prev = V
        V, state, value = step(V, state, t)
        if value is not None:
            history.append(value)
        converged = limit is not None and float(np.max(np.abs(V - V_prev))) <= limit
        result = _result(view, Loop(V, state, t + 1, converged, np.empty(0)), history)
        yield Progress(t + 1, result)
        if converged or t + 1 == max_iter:
            return result()
    raise ValueError("max_iter must be at least 1")


def _result(view, loop: Loop, history: list[float]) -> Callable[[], Result]:
    """The Result of ``loop`` with the history so far, built on demand."""
    n = len(history)
    return lambda: view(loop._replace(history=np.asarray(history[:n], dtype=float)))


def lloyd(
    X: np.ndarray, assign: Assign, update: Update, objective: Objective | None = None
) -> Step:
    """The standard step, D = ||x - v||^2 -> U = assign(D, U_prev, t) -> V = update(U, V, t).

    ``assign`` returns integer labels or float memberships, which are the
    step's state; the optional ``objective(D, U)`` gives one history value.
    """
    xx = sq_norms(X)

    def step(V, U, t):
        D = sqdist(X, V, xx)
        U = assign(D, U, t)
        return update(U, V, t), U, None if objective is None else objective(D, U)

    return step
