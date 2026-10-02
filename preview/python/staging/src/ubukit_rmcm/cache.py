"""Opt-in exact buffered radius-graph reuse for slowly moving data.

The cache stores a conservative *candidate* superset around a reference frame.
It never reuses edges or weights without checking the current coordinates.
"""
from __future__ import annotations
import numbers
import numpy as np
import scipy
from scipy.spatial import cKDTree
from .core import _data, _integer, _BACKENDS, _squared, PreparedRMCM, prepare_rmcm
from ._graph import _csr, _radius, _limit, candidate_pairs_blocked, _screening_coordinates, _gram_candidates

_EPS = np.finfo(float).eps
_TINY = np.finfo(float).tiny


def _original_count(X, delta):
    radius = _radius(delta, X.shape[1])
    counts = cKDTree(X).query_ball_point(X, radius, p=2, eps=0, workers=1, return_length=True)
    return int(sum(map(int, counts)))


class RMCMGraphCache:
    """Reuse exact graph candidates while displacement remains inside a skin.

    Example: cache = RMCMGraphCache(skin=0.2)
             prepared = cache.prepare(current_X, delta=1.0)
             result = prepared.fit(4, init=initial_centers)

    Each prepare snapshots and validates current X, rechecks every candidate
    with the original direct predicate, rebuilds CSR/degrees, and recomputes
    P.T@X. A shape change or exhausted displacement/radius budget rebuilds the
    reference candidates. New N and delta values are supported.

    max_cache_edges is an independent limit on stored directed candidates,
    including self. If a skin would exceed it, prepare falls back to uncached
    strict preparation rather than rejecting an otherwise valid active graph.
    The original max_edges guard is retained: a covering cache below that limit
    proves it passes; otherwise the original preflight is run. candidate_edges
    is eager and exact: audited SciPy versions use ambiguous-row correction;
    other versions run the original query. Nothing is deferred to diagnostics.
    """
    def __init__(self, skin, *, max_cache_edges=10_000_000):
        if isinstance(skin, (bool, np.bool_)) or not isinstance(skin, numbers.Real) or not np.isfinite(skin) or skin < 0:
            raise ValueError("skin must be a finite real number >= 0")
        self.skin = float(skin)
        self.max_cache_edges = _integer("max_cache_edges", max_cache_edges, 1)
        self._reference = self._pairs = None
        self._reference_ids = None
        self._incremental_updates = self._inserted_points = self._removed_points = 0
        self._cover_radius = None
        self._rebuilds = self._reuses = self._fallbacks = 0

    @property
    def stats(self):
        return dict(rebuilds=self._rebuilds, reuses=self._reuses, fallbacks=self._fallbacks,
                    candidate_edges=0 if self._pairs is None else len(self._reference) + 2 * len(self._pairs),
                    cache_bytes=0 if self._pairs is None else self._reference.nbytes + self._pairs.nbytes + (0 if self._reference_ids is None else self._reference_ids.nbytes),
                    incremental_updates=self._incremental_updates, inserted_points=self._inserted_points, removed_points=self._removed_points,
                    cover_radius=self._cover_radius)

    def clear(self):
        self._reference = self._pairs = self._cover_radius = self._reference_ids = None

    def _active_radius(self, delta, d):
        with np.errstate(over="ignore"):
            # Encloses the original inflated-radius candidate predicate,
            # including direct/reference and cKDTree roundoff differences.
            return np.nextafter(_radius(delta, d) * (1 + 64 * _EPS * max(1, d)), np.inf) if delta else 0.0

    def _can_reuse(self, X, target, reference=None):
        reference = self._reference if reference is None else reference
        if reference is None or reference.shape != X.shape:
            return False
        if np.array_equal(X, reference):
            return target <= self._cover_radius
        if not np.isfinite(target) or not np.isfinite(self._cover_radius):
            return False
        diff = X - reference
        squared = np.einsum("ij,ij->i", diff, diff, optimize=False)
        d = X.shape[1]
        # Positive-term reduction + rounded coordinate subtraction: an outward
        # envelope on the Euclidean displacement of represented input points.
        upper_sq = float(np.max(squared)) * (1 + 16 * _EPS * max(1, d)) + _TINY * d
        if not np.isfinite(upper_sq):
            return False
        displacement = np.nextafter(np.sqrt(upper_sq), np.inf)
        needed = np.nextafter(target + np.nextafter(2 * displacement, np.inf), np.inf)
        return needed <= self._cover_radius

    @staticmethod
    def _point_ids(values, n):
        if values is None:
            return None
        ids = np.asarray(values)
        if (ids.ndim != 1 or len(ids) != n or not np.issubdtype(ids.dtype, np.integer)
                or np.issubdtype(ids.dtype, np.bool_)):
            raise ValueError("point_ids must be one distinct signed-64-bit integer per row")
        if np.issubdtype(ids.dtype, np.unsignedinteger) and np.any(ids > np.iinfo(np.int64).max):
            raise ValueError("point_ids must fit signed 64-bit integers")
        ids = np.array(ids, dtype=np.int64, copy=True)
        if len(np.unique(ids)) != n:
            raise ValueError("point_ids must be distinct")
        ids.flags.writeable = False
        return ids

    def _update_point_ids(self, X, ids, target):
        """Retain surviving reference pairs and query births exactly.

        Each point keeps its own reference position. The same triangle bound
        therefore applies to mixed-age reference positions after birth/death
        or reordering. New point pairs are computed at the stored cover radius.
        """
        if self._reference_ids is None or self._reference.shape[1] != X.shape[1]:
            return False
        if np.array_equal(ids, self._reference_ids):
            return self._can_reuse(X, target)
        old_n, n, d = len(self._reference), len(X), X.shape[1]
        order = np.argsort(self._reference_ids)
        sorted_ids = self._reference_ids[order]
        locations = np.searchsorted(sorted_ids, ids)
        matched = locations < old_n
        matched &= sorted_ids[np.minimum(locations, old_n - 1)] == ids
        births = np.flatnonzero(~matched)
        # A large replacement is cheaper to handle with the ordinary bounded
        # rebuild. This affects cost only, never the resulting graph.
        if not np.any(matched) or len(births) > max(64, n // 50):
            return False
        old_rows = order[locations[matched]]
        reference = np.array(X, copy=True)
        reference[matched] = self._reference[old_rows]
        if not self._can_reuse(X, target, reference):
            return False
        mapping = np.full(old_n, -1, dtype=np.int64)
        mapping[old_rows] = np.flatnonzero(matched)
        alive = (mapping[self._pairs[:, 0]] >= 0) & (mapping[self._pairs[:, 1]] >= 0)
        retained = np.sort(mapping[self._pairs[alive]], axis=1)
        saved = [retained]
        count = n + 2 * len(retained)
        if count > self.max_cache_edges:
            return False
        with np.errstate(over="ignore"):
            radius = float(np.nextafter(self._cover_radius * (1 + 64 * _EPS * max(1, d)), np.inf))
        if not np.isfinite(radius):
            return False
        chunk = max(1, 262_144 // d)
        with np.errstate(over="ignore"):
            radius2 = radius * radius
        Z, norms, factor = _screening_coordinates(reference, radius2)
        screen = (d >= 8 and np.isfinite(norms).all() and np.max(norms) <= np.finfo(float).max / 64
                  and factor < (1e-2 if Z.dtype == np.float32 else 1e-4)
                  and np.max(norms) * factor <= max(radius2, _TINY) * .25)
        if screen:
            # Rectangular birth-only tiles avoid an all-pairs rebuild. This is
            # the same certified candidate screen and exact final predicate.
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1, user_api="blas"):
                for lo in range(0, len(births), 128):
                    rows = births[lo:lo + 128]
                    width = max(1, 1_048_576 // max(1, len(rows)))
                    for jl in range(0, n, width):
                        jh = min(n, jl + width)
                        mask = _gram_candidates(Z[rows], Z[jl:jh], norms[rows], norms[jl:jh], radius2, factor, d)
                        a, b = np.nonzero(mask);a = rows[a];b += jl
                        once = (a != b) & (matched[b] | (b > a));a, b = a[once], b[once]
                        for q in range(0, len(a), chunk):
                            ii, jj = a[q:q + chunk], b[q:q + chunk]
                            diff = reference[ii] - reference[jj]
                            sq = np.einsum("ij,ij->i", diff, diff, optimize=False)
                            keep = np.sqrt(sq) <= radius;ii, jj = ii[keep], jj[keep]
                            count += 2 * len(ii)
                            if count > self.max_cache_edges:
                                return False
                            if len(ii):
                                saved.append(np.column_stack((np.minimum(ii, jj), np.maximum(ii, jj))))
        else:
            for i in births:
                for lo in range(0, n, chunk):
                    js = np.arange(lo, min(n, lo + chunk))
                    js = js[(js != i) & (matched[js] | (js > i))]
                    diff = reference[js] - reference[i]
                    sq = np.einsum("ij,ij->i", diff, diff, optimize=False)
                    js = js[np.sqrt(sq) <= radius]
                    count += 2 * len(js)
                    if count > self.max_cache_edges:
                        return False
                    if len(js):
                        saved.append(np.column_stack((np.minimum(i, js), np.maximum(i, js))))
        pairs = np.concatenate(saved)
        if len(pairs) and np.any(pairs[1:, 0] < pairs[:-1, 0]):
            pairs = pairs[np.argsort(pairs[:, 0] * n + pairs[:, 1], kind="stable")]
        if n <= np.iinfo(np.int32).max:
            pairs = np.asarray(pairs, dtype=np.int32)
        reference.flags.writeable = pairs.flags.writeable = False
        self._reference, self._pairs, self._reference_ids = reference, pairs, ids
        self._incremental_updates += 1
        self._inserted_points += len(births)
        self._removed_points += old_n - int(np.count_nonzero(matched))
        return True

    def _build(self, X, target, block_size):
        with np.errstate(over="ignore"):
            cover = float(np.nextafter(target + self.skin, np.inf))
            query_radius = float(np.nextafter(cover * (1 + 64 * _EPS * max(1, X.shape[1])), np.inf))
        if not np.isfinite(cover) or not np.isfinite(query_radius):
            return False
        pairs = None
        if X.shape[1] >= 8 and len(X) >= 64:
            try:
                pairs = candidate_pairs_blocked(X, query_radius, self.max_cache_edges, block_size)
            except MemoryError:
                return False
        if pairs is None:
            tree = cKDTree(X)
            with np.errstate(over="ignore"):
                upper_radius = np.nextafter(query_radius * (1 + 64 * _EPS * max(1, X.shape[1])), np.inf)
            count = int(tree.count_neighbors(tree, upper_radius))
            if count > self.max_cache_edges:
                return False
            pairs = tree.query_pairs(query_radius, p=2, eps=0, output_type="ndarray")
            if len(X) + 2 * len(pairs) > self.max_cache_edges:
                return False
        if len(X) <= np.iinfo(np.int32).max:
            pairs = np.asarray(pairs, dtype=np.int32)
        pairs.flags.writeable = False
        self._reference, self._pairs, self._cover_radius = X, pairs, cover
        self._rebuilds += 1
        return True

    def prepare(self, X, delta, *, backend="adjoint", max_edges=10_000_000, block_size=512, point_ids=None):
        X = _data(X)
        if isinstance(delta, (bool, np.bool_)) or not isinstance(delta, numbers.Real) or not np.isfinite(delta) or delta < 0:
            raise ValueError("delta must be a finite real number >= 0")
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}")
        max_edges = _integer("max_edges", max_edges, 1)
        block_size = _integer("block_size", block_size, 1)
        if len(X) > max_edges:
            raise MemoryError("max_edges is smaller than the mandatory self-edge count")
        point_ids = self._point_ids(point_ids, len(X))
        if point_ids is None and self._reference_ids is not None:
            self.clear()
        delta = float(delta)
        d = X.shape[1]
        target = self._active_radius(delta, d)
        # Preserve the original unusual subnormal-candidate validation exactly;
        # its radius arithmetic is outside the normal-roundoff cache proof.
        eligible = np.isfinite(target) and delta >= 16 * np.sqrt(_TINY) and 64 * _EPS * max(1, d) < 1e-4
        if not eligible:
            self._fallbacks += 1
            return prepare_rmcm(X, delta, backend=backend, max_edges=max_edges, block_size=block_size)
        reused = self._can_reuse(X, target) if point_ids is None else self._update_point_ids(X, point_ids, target)
        if not reused:
            # This reference cannot certify the current frame and would be
            # discarded on either successful rebuild or bounded fallback.
            self.clear()
            if not self._build(X, target, block_size):
                self._fallbacks += 1
                return prepare_rmcm(X, delta, backend=backend, max_edges=max_edges, block_size=block_size)
            self._reference_ids = point_ids
        if reused:
            self._reuses += 1
        count_upper = len(X) + 2 * len(self._pairs)
        exact_count = None
        if count_upper > max_edges or scipy.__version__ not in ("1.17.0", "1.17.1"):
            exact_count = _original_count(X, delta)
            _limit(exact_count, max_edges)
        row_counts = np.ones(len(X), dtype=np.int64) if exact_count is None else None
        ambiguous = np.zeros(len(X), dtype=bool) if exact_count is None else None
        radius = _radius(delta, d)
        with np.errstate(over="ignore"):
            lower = radius * (1 - 64 * _EPS * max(1, d))
            upper = radius * (1 + 64 * _EPS * max(1, d))
        keep = np.empty(len(self._pairs), dtype=bool)
        chunk = max(1, 262_144 // d)
        for lo in range(0, len(self._pairs), chunk):
            p = self._pairs[lo:lo + chunk]
            diff = X[p[:, 0]] - X[p[:, 1]]
            distance = np.sqrt(_squared(diff))
            keep[lo:lo + chunk] = distance <= delta
            if row_counts is not None:
                inside = distance <= radius
                row_counts += np.bincount(p[inside, 0], minlength=len(X))
                row_counts += np.bincount(p[inside, 1], minlength=len(X))
                uncertain = (distance >= lower) & (distance <= upper)
                ambiguous[p[uncertain, 0]] = True
                ambiguous[p[uncertain, 1]] = True
        if exact_count is None:
            exact_count = int(row_counts.sum())
            if np.any(ambiguous):
                counts = cKDTree(X).query_ball_point(X[ambiguous], radius, p=2, eps=0, workers=1, return_length=True)
                exact_count = int(row_counts[~ambiguous].sum()) + int(sum(map(int, counts)))
            _limit(exact_count, max_edges)
        P, degrees, _ = _csr(len(X), self._pairs[keep], count_upper)
        prepared = PreparedRMCM(X, P, degrees, exact_count, delta, backend, block_size)
        prepared.graph_backend = "cache"
        prepared.cache_reused = reused
        return prepared
