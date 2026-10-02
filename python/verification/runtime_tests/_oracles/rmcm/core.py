"""The user-specified delta-neighborhood Rough Membership C-means algorithm.

R = P H; centers = (R.T @ X) / (R.T @ 1).
The adjoint backend reassociates to H.T @ (P.T @ X), with NO exponent.
This is not the binary threshold / upper-region ExRCM algorithm.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numbers
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

_BACKENDS = ("numpy", "csr", "adjoint", "numba")


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _data(X):
    if np.iscomplexobj(X):
        raise ValueError("complex X is unsupported")
    try:
        X = np.array(X, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a finite nonempty real 2-D array") from exc
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a finite nonempty real 2-D array")
    # Direct differences and squared norms must remain representable.
    bound = np.sqrt(np.finfo(float).max / X.shape[1]) / 4
    if np.max(np.abs(X)) > bound:
        raise FloatingPointError("coordinate magnitudes exceed safe float64 distance range; rescale X and delta")
    X.flags.writeable = False
    return X


def _squared(diff):
    sq = np.einsum("ij,ij->i", diff, diff, optimize=False)
    if not np.isfinite(sq).all():
        raise FloatingPointError("distance overflow; rescale X and delta")
    if np.any((sq > 0) & (sq < np.finfo(float).tiny)) or np.any(diff[sq == 0] != 0):
        raise FloatingPointError("distance underflow; rescale X and delta")
    return sq


def _graph(X, delta, max_edges, block_size):
    """Preflight candidate counts, then construct symmetric self-including CSR.

    A conservative radius inflation only selects candidates; the final predicate
    is direct float64 sqrt(sum((xi-xj)**2)) <= delta. It does not inflate delta.
    """
    n, d = X.shape
    tree = cKDTree(X)
    if delta == 0:
        radius = 0.0
    else:
        radius = np.nextafter(delta * (1 + 8 * np.finfo(float).eps * max(1, d)), np.inf)
        if not np.isfinite(radius):
            radius = np.inf
    counts = tree.query_ball_point(X, radius, p=2, eps=0, workers=1, return_length=True)
    candidate_edges = int(sum(map(int, counts)))
    if candidate_edges > max_edges:
        raise MemoryError(f"delta graph has up to {candidate_edges:,} candidate directed edges, exceeding max_edges={max_edges:,}; reduce delta or explicitly raise the limit")
    # Candidates can be larger than accepted edges by a few boundary points.
    indices = np.empty(candidate_edges, dtype=np.int64)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    pos = 0
    edge_block = max(1, 1_048_576 // d)
    for lo in range(0, n, block_size):
        hi = min(n, lo + block_size)
        neighbors = tree.query_ball_point(X[lo:hi], radius, p=2, eps=0, workers=1, return_sorted=True)
        lengths = np.fromiter(map(len, neighbors), dtype=np.int64, count=hi-lo)
        ids = np.concatenate(neighbors).astype(np.int64, copy=False)
        rows = np.repeat(np.arange(lo, hi), lengths)
        keep = np.empty(len(ids), dtype=bool)
        for a in range(0, len(ids), edge_block):
            b = min(len(ids), a + edge_block)
            diff = X[ids[a:b]] - X[rows[a:b]]
            if delta == 0:
                keep[a:b] = np.all(diff == 0, axis=1)
            else:
                keep[a:b] = np.sqrt(_squared(diff)) <= delta
        accepted = ids[keep]
        degrees_block = np.bincount(rows[keep] - lo, minlength=hi-lo)
        indices[pos:pos + len(accepted)] = accepted
        indptr[lo+1:hi+1] = pos + np.cumsum(degrees_block)
        pos += len(accepted)
    indices = indices[:pos].copy()
    degrees = np.diff(indptr)
    if np.any(degrees == 0):
        raise RuntimeError("internal graph error: a self-including neighborhood cannot be empty")
    inverse = 1.0 / degrees
    values = np.repeat(inverse, degrees)
    P = sparse.csr_matrix((values, indices, indptr), shape=(n, n))
    P.sort_indices()
    for a in (P.data, P.indices, P.indptr):
        a.flags.writeable = False
    return P, degrees, candidate_edges


def _hard(X, centers, block_size):
    """Shared direct NumPy distances; argmin preserves lowest-index ties.

    Chunk size also caps the (samples, clusters, features) temporary at about
    16 MiB, unless a single sample's centers exceed that budget.
    """
    labels = np.empty(len(X), dtype=np.int64)
    chunk = min(block_size, max(1, 2_097_152 // (len(centers) * X.shape[1])))
    for lo in range(0, len(X), chunk):
        hi = min(len(X), lo + chunk)
        diff = X[lo:hi, None, :] - centers[None, :, :]
        sq = np.einsum("nkd,nkd->nk", diff, diff, optimize=False)
        if not np.isfinite(sq).all():
            raise FloatingPointError("distance overflow; rescale X and delta")
        if np.any((sq > 0) & (sq < np.finfo(float).tiny)) or np.any(diff[sq == 0] != 0):
            raise FloatingPointError("distance underflow; rescale X and delta")
        labels[lo:hi] = np.argmin(sq, axis=1)
    return labels


def _indicator(labels, k):
    n = len(labels)
    return sparse.csr_matrix((np.ones(n), labels, np.arange(n + 1)), shape=(n, k))


@dataclass
class RMCMResult:
    centers: np.ndarray
    labels: np.ndarray
    memberships: np.ndarray | None
    n_iter: int
    converged: bool
    stop_reason: str
    cycle_length: int | None
    init_indices: np.ndarray | None
    backend: str
    delta: float
    n_edges: int
    empty_cluster_updates: int
    # Includes the returned centers and the assignment that PRODUCED them.
    # There is no hidden final nearest-center assignment.


class PreparedRMCM:
    """An immutable input snapshot, fixed P, and backend-specific precomputation.

    Use prepare_rmcm; repeated .fit calls amortize graph/precompute construction.
    Graph construction and precomputation are excluded from .fit timing and
    included in fit_rmcm timing. Do not mutate private attributes.
    """
    def __init__(self, X, P, degrees, candidate_edges, delta, backend, block_size):
        self._X, self._P, self._degrees = X, P, degrees
        self.delta, self.backend, self.block_size = delta, backend, block_size
        self.n_edges, self.candidate_edges = P.nnz, candidate_edges
        self._full = P.nnz == len(X) * len(X)
        self._mean = X.mean(axis=0) if self._full else None
        self._rows = self._Y = self._s = None
        if backend == "numpy" and not self._full:
            self._rows = np.repeat(np.arange(len(X)), degrees)
        elif backend in ("adjoint", "numba") and not self._full:
            self._Y = np.asarray(P.T @ X)
            self._s = np.asarray(P.sum(axis=0)).ravel()
            if not np.isfinite(self._Y).all():
                raise FloatingPointError("adjoint accumulation overflow; rescale X and delta")
            if backend == "numba":
                try:
                    from ._numba import aggregate
                except ImportError as exc:
                    raise ImportError("backend='numba' requires pip install 'ubukit-rmcm[numba]'") from exc
                self._aggregate_numba = aggregate

    @property
    def degrees(self):
        """Independent degree array; includes each point itself."""
        return self._degrees.copy()

    def neighborhood_matrix(self):
        """Return an independent copy of row-normalized P (CSR)."""
        return self._P.copy()

    def _memberships(self, labels, k):
        if self._full:
            fractions = np.bincount(labels, minlength=k) / len(labels)
            return np.broadcast_to(fractions, (len(labels), k)).copy()
        if self.backend == "numpy":
            keys = self._rows * k + labels[self._P.indices]
            counts = np.bincount(keys, minlength=len(labels) * k).reshape(-1, k)
            return counts / self._degrees[:, None]
        return self._P @ _indicator(labels, k)

    def _step(self, centers):
        X, k = self._X, len(centers)
        labels = _hard(X, centers, self.block_size)
        if self._full:
            # Exact algebra: every occupied rough center is the global mean.
            # One common reduction prevents false tie differences across clusters.
            mass = np.bincount(labels, minlength=k)
            updated = centers.copy()
            updated[mass > 0] = self._mean
            return labels, updated, int(np.count_nonzero(mass == 0))
        if self.backend in ("adjoint", "numba"):
            if self.backend == "numba":
                numerator, mass = self._aggregate_numba(labels, self._Y, self._s, k)
            else:
                H = _indicator(labels, k)
                numerator = np.asarray(H.T @ self._Y)
                mass = np.bincount(labels, weights=self._s, minlength=k)
        else:
            R = self._memberships(labels, k)
            numerator = np.asarray(R.T @ X)
            mass = np.asarray(R.sum(axis=0)).ravel()
        updated = centers.copy()
        nonempty = mass > 0
        updated[nonempty] = numerator[nonempty] / mass[nonempty, None]
        if not np.isfinite(updated).all():
            raise FloatingPointError("centroid overflow; rescale X and delta")
        return labels, updated, int(np.count_nonzero(~nonempty))

    def fit(self, n_clusters, *, init=None, random_state=None, max_iter=100,
            cycle_window=32, return_memberships=True, threads=1):
        """Fit with exact full-state stopping, no tolerance-based convergence.

        A repeated consecutive (labels, centers) state is a fixed point.
        Other repeated complete states in cycle_window are reported as cycles.
        A finite window can miss longer cycles; max_iter remains a hard bound.
        """
        X = self._X
        k = _integer("n_clusters", n_clusters, 1)
        max_iter = _integer("max_iter", max_iter, 1)
        cycle_window = _integer("cycle_window", cycle_window, 0)
        threads = _integer("threads", threads, 1)
        if not isinstance(return_memberships, (bool, np.bool_)):
            raise ValueError("return_memberships must be bool")
        if k > len(X):
            raise ValueError("n_clusters cannot exceed n_samples (distinct-index initialization)")
        indices = None
        if init is None:
            indices = np.random.default_rng(random_state).choice(len(X), k, replace=False)
            centers = X[indices].copy()
        else:
            if np.iscomplexobj(init):
                raise ValueError("complex initial centers are unsupported")
            centers = np.array(init, dtype=np.float64, order="C", copy=True)
            if centers.shape != (k, X.shape[1]) or not np.isfinite(centers).all():
                raise ValueError("init must be finite with shape (n_clusters, n_features)")
            if np.max(np.abs(centers)) > np.sqrt(np.finfo(float).max / X.shape[1]) / 4:
                raise FloatingPointError("initial centers exceed safe distance range; rescale")
        previous_labels = previous_centers = None
        history, seen = deque(), {}
        reason, cycle, empty = "max_iter", None, 0
        with threadpool_limits(limits=threads):
            for iteration in range(1, max_iter + 1):
                labels, centers, count = self._step(centers)
                empty += count
                if previous_labels is not None and np.array_equal(labels, previous_labels) and np.array_equal(centers, previous_centers):
                    reason = "fixed_point"
                    break
                if cycle_window:
                    state = labels.tobytes() + centers.tobytes()
                    if state in seen:
                        reason, cycle = "cycle", iteration - seen[state]
                        break
                    seen[state] = iteration
                    history.append((state, iteration))
                    if len(history) > cycle_window:
                        old, old_iteration = history.popleft()
                        if seen.get(old) == old_iteration:
                            del seen[old]
                previous_labels, previous_centers = labels, centers
            R = None
            if return_memberships:
                R = self._memberships(labels, k)
                if sparse.issparse(R):
                    R = R.toarray()
        return RMCMResult(centers, labels, R, iteration, reason == "fixed_point", reason,
                          cycle, indices, self.backend, self.delta, self.n_edges, empty)


def prepare_rmcm(X, delta, *, backend="adjoint", max_edges=10_000_000, block_size=512):
    """Build the graph once and prepare its chosen exact-arithmetic backend.

    max_edges limits conservative directed candidates, including self edges.
    The dense graph still needs O(N²) space; this fails before allocating it.
    """
    X = _data(X)
    if isinstance(delta, (bool, np.bool_)) or not isinstance(delta, numbers.Real) or not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be a finite real number >= 0")
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}")
    max_edges = _integer("max_edges", max_edges, 1)
    block_size = _integer("block_size", block_size, 1)
    if len(X) > max_edges:
        raise MemoryError("max_edges is smaller than the mandatory self-edge count")
    P, degrees, count = _graph(X, float(delta), max_edges, block_size)
    return PreparedRMCM(X, P, degrees, count, float(delta), backend, block_size)


def fit_rmcm(X, n_clusters, delta, *, init=None, random_state=None, max_iter=100,
             backend="adjoint", max_edges=10_000_000, block_size=512,
             cycle_window=32, return_memberships=True, threads=1):
    """One-shot RMCM, including graph construction and precomputation.

    Returns memberships (N,K), hard provisional labels (N,), centers (K,D).
    init may explicitly supply the same centers for fair comparisons.
    """
    prepared = prepare_rmcm(X, delta, backend=backend, max_edges=max_edges, block_size=block_size)
    return prepared.fit(n_clusters, init=init, random_state=random_state, max_iter=max_iter,
                        cycle_window=cycle_window, return_memberships=return_memberships, threads=threads)


def fit_rmcm_numpy(X, n_clusters, delta, **kwargs):
    """Readable NumPy scratch reference with O(E+NK) occupancy counting."""
    if "backend" in kwargs:
        raise TypeError("fit_rmcm_numpy selects backend='numpy'; do not supply backend")
    return fit_rmcm(X, n_clusters, delta, backend="numpy", **kwargs)
