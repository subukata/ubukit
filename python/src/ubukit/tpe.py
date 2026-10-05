"""Tree-structured Parzen Estimator for small mixed search spaces.

Multivariate TPE (Bergstra et al., 2011; Falkner et al., 2018): finished
trials are split into the best ``gamma`` fraction and the rest, each side is
modeled by a Parzen mixture whose components keep the dimensions of one trial
together, and the sampled candidate maximizing l(x) / g(x) is proposed.
Numeric dimensions live on [0, 1]; categorical ones use smoothed one-hot kernels.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

from ._core import MAX_INT, check_float, check_int


@dataclass(frozen=True, slots=True)
class _Numeric:
    low: float
    high: float
    log: bool
    integer: bool

    def _bounds(self) -> tuple[float, float]:
        lo, hi = (self.low - 0.5, self.high + 0.5) if self.integer else (self.low, self.high)
        return (math.log(lo), math.log(hi)) if self.log else (lo, hi)

    def encode(self, x) -> float:
        if (
            isinstance(x, bool)
            or not isinstance(x, Real)
            or not self.low <= x <= self.high
            or (self.integer and x != round(x))
        ):
            kind = "an integer" if self.integer else "a number"
            raise ValueError(f"{x!r} is not {kind} in [{self.low}, {self.high}]")
        lo, hi = self._bounds()
        v = math.log(x) if self.log else float(x)
        return (v - lo) / (hi - lo)

    def decode(self, u: float):
        lo, hi = self._bounds()
        x = lo + float(u) * (hi - lo)
        x = math.exp(x) if self.log else x
        if self.integer:
            return int(min(max(round(x), self.low), self.high))
        return min(max(x, self.low), self.high)


@dataclass(frozen=True, slots=True)
class _Choice:
    options: tuple

    def encode(self, x) -> float:
        if x not in self.options:
            raise ValueError(f"{x!r} is not an option")
        return float(self.options.index(x))

    def decode(self, u: float):
        return self.options[int(u)]


def uniform(low, high) -> _Numeric:
    """Real values in [low, high]."""
    return _numeric(low, high, log=False, integer=False)


def loguniform(low, high) -> _Numeric:
    """Positive real values in [low, high], searched on a log scale."""
    return _numeric(low, high, log=True, integer=False)


def integer(low, high, *, log=False) -> _Numeric:
    """Integers in [low, high], optionally searched on a log scale."""
    low = check_int(low, "low", -MAX_INT, MAX_INT)
    return _numeric(low, check_int(high, "high", low, MAX_INT), log, True)


def choice(*options) -> _Choice:
    """One of the given options (compared with ``==``)."""
    if not options:
        raise ValueError("choice needs at least one option")
    return _Choice(tuple(options))


def _numeric(low, high, log, integer) -> _Numeric:
    low = check_float(low, "low", 0.0, strict=True) if log else check_float(low, "low")
    high = check_float(high, "high", low, strict=not integer)
    if not math.isfinite(high - low):
        raise ValueError("high - low must be finite")
    return _Numeric(low, high, log, integer)


@dataclass(frozen=True, slots=True, eq=False)
class TPEResult:
    """Best trial and the full history, in evaluation order."""

    best_params: dict
    best_value: float
    params: list[dict]
    values: np.ndarray


class TPE:
    """Ask/tell TPE minimizer.

    Args:
        space: mapping of names to ``uniform``, ``loguniform``, ``integer`` or ``choice``.
        seed: random seed.
        n_startup: random trials before the model is used.
        n_candidates: candidates drawn from l(x) per proposal.
        gamma: fraction of trials treated as good.
    """

    def __init__(self, space, *, seed=None, n_startup=10, n_candidates=24, gamma=0.15):
        if not isinstance(space, dict) or not space:
            raise ValueError("space must be a non-empty dict")
        for name, dim in space.items():
            if not isinstance(dim, _Numeric | _Choice):
                raise ValueError(
                    f"space[{name!r}] must come from uniform/loguniform/integer/choice"
                )
        self._space = dict(space)
        self._rng = np.random.default_rng(seed)
        self.n_startup = check_int(n_startup, "n_startup", 2)
        self.n_candidates = check_int(n_candidates, "n_candidates", 1)
        self.gamma = check_float(gamma, "gamma", 0.0, strict=True)
        if self.gamma >= 1:
            raise ValueError("gamma must be < 1")
        self._encoded: list[np.ndarray] = []
        self._params: list[dict] = []
        self._values: list[float] = []

    def ask(self) -> dict:
        """Propose parameters to evaluate next."""
        if len(self._values) < self.n_startup:
            u = [
                self._rng.integers(len(d.options)) if isinstance(d, _Choice) else self._rng.random()
                for d in self._dims
            ]
        else:
            u = self._propose()
        return {name: dim.decode(x) for (name, dim), x in zip(self._space.items(), u, strict=True)}

    def tell(self, params: dict, value: float) -> None:
        """Record the objective value of ``params`` (lower is better)."""
        if set(params) != set(self._space):
            raise ValueError("params must have exactly the keys of the space")
        value = check_float(value, "value")
        self._encoded.append(np.array([dim.encode(params[k]) for k, dim in self._space.items()]))
        self._params.append(dict(params))
        self._values.append(value)

    def result(self) -> TPEResult:
        if not self._values:
            raise ValueError("no trials have been told")
        best = int(np.argmin(self._values))
        return TPEResult(
            dict(self._params[best]),
            self._values[best],
            [dict(p) for p in self._params],
            np.array(self._values),
        )

    @property
    def _dims(self):
        return list(self._space.values())

    def _propose(self) -> np.ndarray:
        X, y = np.array(self._encoded), np.array(self._values)
        n_good = min(len(y) - 1, max(1, math.ceil(self.gamma * len(y))))
        order = np.argsort(y, kind="stable")
        good = _Parzen(X[order[:n_good]], self._dims)
        bad = _Parzen(X[order[n_good:]], self._dims)
        candidates = good.sample(self._rng, self.n_candidates)
        return candidates[np.argmax(good.logpdf(candidates) - bad.logpdf(candidates))]


def minimize(f: Callable[[dict], float], space, n_trials=100, *, seed=None, **options) -> TPEResult:
    """Minimize ``f(params)`` over ``space`` with TPE; negate ``f`` to maximize."""
    tpe = TPE(space, seed=seed, **options)
    for _ in range(check_int(n_trials, "n_trials", 1)):
        params = tpe.ask()
        tpe.tell(params, f(params))
    return tpe.result()


class _Parzen:
    """Equal-weight mixture of one component per observation plus a uniform prior."""

    def __init__(self, obs: np.ndarray, dims: list):
        m = len(obs)
        self.num = [j for j, d in enumerate(dims) if isinstance(d, _Numeric)]
        self.cat = [(j, len(d.options)) for j, d in enumerate(dims) if isinstance(d, _Choice)]
        self.log_weight = -math.log(m + 1)
        # Numeric kernels: truncated normals on [0, 1] whose width is the larger
        # gap to the neighboring centers (Bergstra et al.); the prior is wide.
        self.mu = np.vstack([obs[:, self.num], np.full((1, len(self.num)), 0.5)])
        order = np.argsort(self.mu, axis=0)
        s = np.take_along_axis(self.mu, order, axis=0)
        gaps = np.diff(s, axis=0, prepend=0.0, append=1.0)
        width = np.clip(np.maximum(gaps[:-1], gaps[1:]), max(0.03, (m + 1) ** -2.0), 1.0)
        self.sigma = np.empty_like(self.mu)
        np.put_along_axis(self.sigma, order, width, axis=0)
        self.sigma[-1] = 1.0
        self.cdf_lo = ndtr(-self.mu / self.sigma)
        self.cdf_hi = ndtr((1 - self.mu) / self.sigma)
        self.log_norm = np.log(self.sigma * math.sqrt(2 * math.pi) * (self.cdf_hi - self.cdf_lo))
        # Categorical kernels: own option with weight 1 - eps, rest uniform.
        eps = 1.0 / (m + 1)
        self.probs = []
        for j, k in self.cat:
            P = np.full((m + 1, k), eps / k)
            P[np.arange(m), obs[:, j].astype(int)] += 1 - eps
            P[-1] = 1.0 / k
            self.probs.append(P)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        comp = rng.integers(len(self.mu), size=size)
        out = np.empty((size, len(self.num) + len(self.cat)))
        lo, hi = self.cdf_lo[comp], self.cdf_hi[comp]
        z = self.mu[comp] + self.sigma[comp] * ndtri(lo + rng.random(lo.shape) * (hi - lo))
        out[:, self.num] = np.clip(z, 0.0, 1.0)
        for (j, _), P in zip(self.cat, self.probs, strict=True):
            cum = np.cumsum(P[comp], axis=1)
            out[:, j] = np.minimum((cum < rng.random((size, 1))).sum(axis=1), P.shape[1] - 1)
        return out

    def logpdf(self, Z: np.ndarray) -> np.ndarray:
        t = (Z[:, None, self.num] - self.mu) / self.sigma
        L = np.sum(-0.5 * t * t - self.log_norm, axis=2)
        for (j, _), P in zip(self.cat, self.probs, strict=True):
            L += np.log(P[:, Z[:, j].astype(int)]).T
        return logsumexp(L + self.log_weight, axis=1)
