"""Traditional online and true batch self-organizing maps.

See SOM.md for schedules, numerical limits and the per-sample/per-epoch step
contract. These algorithms do not share SOM-OLP's objective or update rule.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from fractions import Fraction
import numpy as np
from scipy.spatial.distance import cdist

from .policy import policy_or_default
from .validation import matrix_input, matrix, positive_int

__all__ = ["som", "som_batch", "fit_som", "fit_som_batch", "initialize_som",
           "initialize_som_batch", "SOMState"]


def _number(value, name, *, maximum=math.inf):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be finite and in [0, {maximum}]")
    return value


def _shape(shape):
    try:
        if len(shape) != 2:
            raise ValueError("grid_shape must contain width and height")
    except TypeError:
        raise ValueError("grid_shape must contain width and height") from None
    return tuple(positive_int(x, "grid_shape entry") for x in shape)


def _schedule(start, end, index, count, kind):
    """Endpoints are inclusive; a singleton schedule uses its start."""
    if count <= 1 or index == 0:
        return start
    if index >= count - 1:
        return end
    fraction = index / (count - 1)
    if kind == "linear" or start == 0 or end == 0:
        return (1 - fraction) * start + fraction * end
    # Log interpolation avoids overflow in start/end for extreme endpoints.
    return math.exp((1 - fraction) * math.log(start) + fraction * math.log(end))


def _mulberry32(seed):
    state = seed
    mask = 0xffffffff
    while True:
        state = (state + 0x6D2B79F5) & mask
        t = ((state ^ (state >> 15)) * (1 | state)) & mask
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & mask)) & mask) ^ t
        yield ((t ^ (t >> 14)) & mask) / 4294967296


def _kernel(distance, sigma):
    """Gaussian kernel; sigma=0 is the explicit winner-only limit."""
    if sigma == 0:
        return np.equal(distance, 0).astype(np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        ratio = np.asarray(distance, dtype=np.float64) / sigma
        np.square(ratio, out=ratio)
        ratio *= -.5
        np.exp(ratio, out=ratio)
    return ratio


def _integer_vector(row):
    # Every finite binary64 is an integer times 2**-1074. This cold path gives
    # exact squared-distance ordering even when float64 differences overflow.
    result = []
    for value in row:
        numerator, denominator = float(value).as_integer_ratio()
        result.append(numerator << (1075 - denominator.bit_length()))
    return result


def _exact_labels(X, W):
    codes = [_integer_vector(row) for row in W]
    labels = np.empty(len(X), dtype=np.intp)
    for i, row in enumerate(X):
        x = _integer_vector(row)
        best = None
        winner = 0
        for j, w in enumerate(codes):
            cost = sum((a - b) ** 2 for a, b in zip(x, w))
            if best is None or cost < best:
                best, winner = cost, j
        labels[i] = winner
    return labels


def _initial_prototypes(X, R, shape, initial, initializer, seed, pca_scale, policy):
    m = len(R)
    if initial is not None:
        W = matrix(initial, dtype=np.float64, name="initial_prototypes")
        if W.shape != (m, X.shape[1]):
            raise ValueError("initial_prototypes must have shape (width*height, n_features)")
        return W.copy(), "explicit"
    if initializer == "sample":
        rng = _mulberry32(seed)
        indices = [int(next(rng) * len(X)) for _ in range(m)]
        return X[indices].copy(), "sample"
    if initializer != "pca":
        raise ValueError("initializer must be 'sample' or 'pca'")
    from ._som_extreme import pca_factors, _float
    with policy.activate():
        mean, _, singular, basis, exponent = pca_factors(X)
    k = min(2, len(singular))
    normalized = np.zeros((m, 2), dtype=np.float64)
    for axis, length in enumerate(shape):
        if length > 1:
            normalized[:, axis] = 2 * R[:, axis] / (length - 1) - 1
    W = np.repeat(mean, m, axis=0)
    for axis in range(k):
        direction = basis[axis].copy()
        if direction[np.argmax(np.abs(direction))] < 0:
            direction *= -1
        component = (singular[axis] / math.sqrt(len(X))) * pca_scale
        W += normalized[:, axis, None] * component * direction
    if not np.isfinite(W).all():
        raise ValueError("PCA prototypes exceed float64 range; reduce pca_scale")
    # Per-entry checked restoration, including subnormal rounding.
    for row in W:
        for j in range(len(row)):
            row[j] = _float((float(row[j]), exponent), "PCA prototype")
    return W, "pca"


class SOMState:
    """Owned resumable state. Use initialize_som/initialize_som_batch.

    step() commits exactly one sample update (online) or full frozen-BMU epoch
    (batch), then returns a detached prototype snapshot. result() computes labels
    for the current prototypes. cancel() is cooperative between complete updates.
    """

    def __init__(self, X, *, algorithm="som", grid_shape=(16, 16),
                 initial_prototypes=None, initializer="sample", random_state=0,
                 pca_scale=2., epochs=100, max_iterations=None,
                 sigma=None, sigma_end=None, schedule="geometric",
                 learning_rate=None, learning_rate_end=None, policy=None):
        if algorithm not in ("som", "som_batch"):
            raise ValueError("algorithm must be 'som' or 'som_batch'")
        self.algorithm = algorithm
        self.unit = "sample" if algorithm == "som" else "epoch"
        self.grid_shape = _shape(grid_shape)
        self.policy = policy_or_default(policy)
        self._X = matrix_input(X, dtype=np.float64)[0].copy()
        n, d = self._X.shape
        width, height = self.grid_shape
        m = width * height
        epochs = positive_int(epochs, "epochs", allow_zero=True)
        self.total_iterations = epochs * n if algorithm == "som" else epochs
        if max_iterations is not None:
            self.total_iterations = positive_int(max_iterations, "max_iterations", allow_zero=True)
        if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)) or not 0 <= random_state <= 0xffffffff:
            raise ValueError("random_state must be an unsigned 32-bit integer")
        pca_scale = _number(pca_scale, "pca_scale")
        if initializer not in ("sample", "pca"):
            raise ValueError("initializer must be 'sample' or 'pca'")
        if schedule not in ("geometric", "linear"):
            raise ValueError("schedule must be 'geometric' or 'linear'")
        self.schedule = schedule
        self._sigma_start = _number(max(width, height) / 2 if sigma is None else sigma, "sigma")
        self._sigma_end = _number(min(.5, self._sigma_start) if sigma_end is None else sigma_end, "sigma_end")
        if self._sigma_end > self._sigma_start:
            raise ValueError("sigma_end must not exceed sigma")
        if algorithm == "som_batch":
            if learning_rate is not None or learning_rate_end is not None:
                raise ValueError("Batch SOM has no learning-rate parameter")
            self._rate_start = self._rate_end = None
        else:
            self._rate_start = _number(.5 if learning_rate is None else learning_rate, "learning_rate", maximum=1.)
            self._rate_end = _number(min(.05, self._rate_start) if learning_rate_end is None else learning_rate_end, "learning_rate_end", maximum=1.)
            if self._rate_end > self._rate_start:
                raise ValueError("learning_rate_end must not exceed learning_rate")
        # Named scratch cap includes primary NumPy buffers only, excluding owned
        # inputs/prototypes/output snapshots, PCA/LAPACK and cold Python integers.
        self._fixed_scratch = 8 * (8*m*d + 8*m + 2*(width*width + height*height) + 2*n) if algorithm == "som_batch" else 8 * (6*m*d + 6*m)
        available = self.policy.max_scratch_bytes - self._fixed_scratch
        if available < 8*m:
            raise ValueError("scratch cap cannot hold SOM primary buffers and one distance row")
        self._block_rows = min(n, self.policy.block_rows, available // (8*m))
        self.primary_scratch_budgeted_bytes = self._fixed_scratch + 8*m*self._block_rows
        ids = np.arange(m)
        self._R = np.column_stack((ids % width, ids // width)).astype(np.float64)
        self._W, self.initializer = _initial_prototypes(self._X, self._R, self.grid_shape,
                                                       initial_prototypes, initializer,
                                                       int(random_state), pca_scale, self.policy)
        from ._som_extreme import needs_extreme
        self._extreme = needs_extreme(self._X, self._W)
        self.iterations = 0
        self.cancelled = False
        self._last_sigma = None
        self._last_rate = None
        self._last_sample = None
        self._last_bmu = None

    @property
    def done(self):
        return self.cancelled or self.iterations >= self.total_iterations

    @property
    def centers(self):
        return self._W.copy()

    @property
    def prototypes(self):
        return self._W.copy()

    @property
    def grid(self):
        return self._R.copy()

    def cancel(self):
        """Stop before the next complete update. No partial update is exposed."""
        self.cancelled = True

    def _labels(self, X):
        if self._extreme:
            return _exact_labels(X, self._W)
        labels = np.empty(len(X), dtype=np.intp)
        for start in range(0, len(X), self._block_rows):
            block = X[start:start+self._block_rows]
            cost = cdist(block, self._W, metric="sqeuclidean")
            if not np.isfinite(cost).all():
                labels[start:start+len(block)] = _exact_labels(block, self._W)
            else:
                labels[start:start+len(block)] = np.argmin(cost, axis=1)
        return labels

    def _online_update(self, sigma, rate):
        sample = self.iterations % len(self._X)
        x = self._X[sample]
        bmu = int(self._labels(self._X[sample:sample+1])[0])
        delta = self._R - self._R[bmu]
        distances = np.hypot(delta[:, 0], delta[:, 1])
        alpha = _kernel(distances, sigma) * rate
        if self._extreme:
            W = np.empty_like(self._W)
            for j, a in enumerate(alpha):
                fraction = Fraction(float(a))
                for f in range(W.shape[1]):
                    old = Fraction(float(self._W[j, f]))
                    W[j, f] = float(old + fraction * (Fraction(float(x[f])) - old))
        else:
            a = alpha[:, None]
            # Convex form avoids losing a small x in the subtraction x-W when
            # alpha is close to one. It is the same real-valued SOM update.
            W = (1-a) * self._W + a * x
            bound = (1-a) * np.abs(self._W) + a * np.abs(x)
            suspect = (np.abs(W) <= 1e-10 * bound) & (bound > 0)
            for j, f in zip(*np.nonzero(suspect)):
                old = Fraction(float(self._W[j, f]))
                W[j, f] = float(old + Fraction(float(alpha[j])) * (Fraction(float(x[f])) - old))
            W[alpha == 1] = x
            W[alpha == 0] = self._W[alpha == 0]
        return W, sample, bmu

    def _batch_update(self, sigma):
        labels = self._labels(self._X)  # Frozen until every new prototype exists.
        if self._extreme:
            from ._som_extreme import _weighted_mean
            W = self._W.copy()
            for j, point in enumerate(self._R):
                delta = self._R - point
                weights = _kernel(np.hypot(delta[:, 0], delta[:, 1]), sigma)[labels]
                if np.any(weights):
                    for f in range(W.shape[1]):
                        W[j, f] = _weighted_mean(self._X[:, f], weights)
            return W
        width, height = self.grid_shape
        m, d = self._W.shape
        counts = np.bincount(labels, minlength=m).astype(np.float64)
        sums = np.zeros((m, d), dtype=np.float64)
        np.add.at(sums, labels, self._X)
        # Gaussian distances on a rectangular grid separate exactly into x/y
        # kernels. Smooth grouped sufficient statistics via matrix products;
        # never allocate an N*M*D tensor or a full M*M neighborhood matrix.
        hx = _kernel(np.abs(np.arange(width)[:, None] - np.arange(width)), sigma)
        hy = _kernel(np.abs(np.arange(height)[:, None] - np.arange(height)), sigma)
        horizontal = np.matmul(hx, sums.reshape(height, width, d))
        numerator = (hy @ horizontal.reshape(height, width*d)).reshape(m, d)
        denominator = (hy @ counts.reshape(height, width) @ hx.T).reshape(m)
        W = self._W.copy()
        nonempty = denominator > 0
        W[nonempty] = numerator[nonempty] / denominator[nonempty, None]
        # Strong signed cancellation can discard a representable residual in
        # grouped reductions or BLAS. Repair only suspect output coordinates
        # from raw samples using exact binary product accumulation.
        xmax = np.array([np.max(np.abs(self._X[:, f])) for f in range(d)])
        with np.errstate(under="ignore"):
            bound = denominator[:, None] * xmax[None, :]
            suspect = nonempty[:, None] & (np.abs(numerator) <= 1e-10 * bound) & (bound > 0)
        if np.any(suspect):
            from ._som_extreme import _weighted_mean
            for j in np.flatnonzero(np.any(suspect, axis=1)):
                delta = self._R - self._R[j]
                weights = _kernel(np.hypot(delta[:, 0], delta[:, 1]), sigma)[labels]
                for f in np.flatnonzero(suspect[j]):
                    W[j, f] = _weighted_mean(self._X[:, f], weights)
        # Tiny neighborhood masses can underflow weighted numerators even for
        # ordinary X. Repair only these cold rows in raw sample coordinates.
        tiny = np.flatnonzero((denominator > 0) & (denominator < 1e-140))
        if len(tiny):
            from ._som_extreme import _weighted_mean
            for j in tiny:
                delta = self._R - self._R[j]
                weights = _kernel(np.hypot(delta[:, 0], delta[:, 1]), sigma)[labels]
                if np.any(weights):
                    for f in range(d):
                        W[j, f] = _weighted_mean(self._X[:, f], weights)
        return W

    def snapshot(self):
        """Detached lightweight snapshot; no all-sample assignment is performed."""
        result = {"algorithm": self.algorithm, "centers": self._W.copy(),
                  "grid": self._R.copy(), "grid_shape": self.grid_shape,
                  "iterations": self.iterations, "total_iterations": self.total_iterations,
                  "unit": self.unit, "epochs_completed": self.iterations // len(self._X) if self.unit == "sample" else self.iterations,
                  "samples_seen": self.iterations if self.unit == "sample" else self.iterations * len(self._X),
                  "sigma": self._last_sigma, "learning_rate": self._last_rate,
                  "sample_index": self._last_sample, "bmu": self._last_bmu,
                  "done": self.done, "cancelled": self.cancelled,
                  "initializer": self.initializer, "backend": "exact-cold" if self._extreme else "scipy-cdist-numpy-separable",
                  "primary_scratch_budgeted_bytes": self.primary_scratch_budgeted_bytes,
                  "scratch_rows": self._block_rows}
        return result

    def step(self):
        """Commit one update, returning its snapshot, or None if already done."""
        if not self._advance():
            return None
        return self.snapshot()

    def _advance(self, *, activate=True):
        if self.done:
            return False
        sigma = _schedule(self._sigma_start, self._sigma_end, self.iterations,
                          self.total_iterations, self.schedule)
        rate = None if self.unit == "epoch" else _schedule(self._rate_start, self._rate_end,
                        self.iterations, self.total_iterations, self.schedule)
        with self.policy.activate() if activate else nullcontext():
            if self.unit == "sample":
                W, sample, bmu = self._online_update(sigma, rate)
            else:
                W, sample, bmu = self._batch_update(sigma), None, None
        if not np.isfinite(W).all():
            raise ValueError("SOM update exceeds float64 range; rescale inputs")
        # Only commit after a whole successful update. Exceptions leave state
        # unchanged, including counters and schedule position.
        self._W = W
        self.iterations += 1
        self._last_sigma, self._last_rate = sigma, rate
        self._last_sample, self._last_bmu = sample, bmu
        return True

    def run(self, max_updates=None):
        """Run remaining updates (or a bounded prefix), then return final labels."""
        count = self.total_iterations - self.iterations if max_updates is None else positive_int(max_updates, "max_updates", allow_zero=True)
        # One outer scope avoids entering threadpoolctl for every online sample.
        with self.policy.activate():
            for _ in range(count):
                if not self._advance(activate=False):
                    break
        return self.result()

    def result(self):
        """Return a detached snapshot with BMU labels/embedding for current W."""
        result = self.snapshot()
        with self.policy.activate():
            labels = self._labels(self._X)
        result["labels"] = labels
        result["embedding"] = self._R[labels].copy()
        # Existing SOM-family mathematical aliases, same detached snapshot.
        result["W"] = result["centers"]
        result["R"] = result["grid"]
        result["V"] = result["embedding"]
        return result


def initialize_som(X, **kwargs):
    """Create online SOM state; one step is one sample in input order."""
    return SOMState(X, algorithm="som", **kwargs)


def initialize_som_batch(X, **kwargs):
    """Create true batch SOM state; one step is one complete frozen-BMU epoch."""
    return SOMState(X, algorithm="som_batch", **kwargs)


def fit_som(X, **kwargs):
    """Fit an online SOM. See SOM.md for options and return fields."""
    return initialize_som(X, **kwargs).run()


def fit_som_batch(X, **kwargs):
    """Fit a true batch SOM; no learning-rate parameter is used."""
    return initialize_som_batch(X, **kwargs).run()


som = fit_som
som_batch = fit_som_batch
