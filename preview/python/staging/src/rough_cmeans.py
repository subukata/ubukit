"""The requested binary-threshold / normalized-mask ExRCM algorithm (rough c-means when p=1).

No fuzzifier, lower/upper-region interpolation, or nearest-centre tie-breaking.
Public memberships are (n_clusters, n_samples), matching U[c, i].
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numbers
import numpy as np


@dataclass
class RoughCMeansResult:
    centers: np.ndarray
    memberships: np.ndarray | None
    upper_memberships: np.ndarray | None
    n_iter: int
    converged: bool
    stop_reason: str
    cycle_length: int | None
    init_indices: np.ndarray | None
    backend: str
    p: float
    alpha: float
    beta: float
    empty_cluster_updates: int
    # True only if the final centers are the means induced by their own masks.
    fixed_point: bool


def _validate(X, n_clusters, init, alpha, beta, p, max_iter, block_size,
              cycle_window, check_sample_count=True):
    if np.iscomplexobj(X) or (init is not None and np.iscomplexobj(init)):
        raise ValueError('complex input is not supported')
    X = np.asarray(X, dtype=np.float64, order='C')
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError('X must be a finite, nonempty 2-D array')
    for name, value, lower in [('n_clusters', n_clusters, 1),
                              ('max_iter', max_iter, 1),
                              ('block_size', block_size, 1),
                              ('cycle_window', cycle_window, 0)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}')
    if check_sample_count and n_clusters > len(X):
        raise ValueError('n_clusters cannot exceed n_samples (distinct-index initialization)')
    if not np.isfinite(alpha) or alpha < 1:
        raise ValueError('alpha must be finite and >= 1')
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and >= 0')
    if not np.isfinite(p) or p <= 0:
        raise ValueError('p must be finite and > 0')
    if init is not None:
        init = np.array(init, dtype=np.float64, order='C', copy=True)
        if init.shape != (n_clusters, X.shape[1]) or not np.isfinite(init).all():
            raise ValueError('init must be finite with shape (n_clusters, n_features)')
    return X, init


def _distances_numpy(X, C):
    """Direct differences: no cancellation-prone ||x||²+||c||²-2 x.c."""
    out = np.empty((len(X), len(C)), dtype=np.float64)
    if X.shape[1] == 1:
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            np.subtract(X, C.T, out=out)
            np.square(out, out=out)
        return out
    for c, center in enumerate(C):
        delta = X - center
        # Sum each contiguous difference row in native NumPy, without a
        # Python loop over features or a second squared-difference matrix.
        out[:, c] = np.einsum('ij,ij->i', delta, delta, optimize=False)
    return out


def _row_minimum(values):
    if values.shape[1] <= 16:
        out = values[:, 0].copy()
        for col in range(1, values.shape[1]):
            np.minimum(out, values[:, col], out=out)
        return out
    return values.min(axis=1)


def _check_distance_range(X, C, sq):
    # Validate range while retaining row minima for threshold construction.
    if not np.isfinite(np.max(sq)):
        raise FloatingPointError('squared distances overflowed; rescale the input')
    minsq = _row_minimum(sq)
    if minsq.min() < np.finfo(float).tiny:
        if np.any((sq > 0) & (sq < np.finfo(float).tiny)):
            raise FloatingPointError('squared distances are subnormal; rescale the input')
        points = np.flatnonzero(minsq == 0)
        if len(points):
            rows, cols = np.nonzero(sq[points] == 0)
            if np.any(X[points[rows]] != C[cols]):
                raise FloatingPointError('nonzero distances underflowed to zero; rescale the input')
    return minsq


def _scaled_power_mask(sq, alpha, beta, p):
    """Overflow-safe canonical comparison, with log fallback at extreme scales."""
    d = np.sqrt(sq)
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        a = alpha * d.min(axis=1)
        if beta == 0:
            # This algebraic cancellation is essential when p is tiny: pow can
            # round distinct positive ratios to one even far from a boundary.
            return d <= a[:, None]
        scale = np.maximum(a, beta)
        small = np.minimum(a, beta)
        ratio = small / scale
        left_ratio = d / scale[:, None]
        if p < 0.25 or p > 64:
            # Choose log before evaluating a whole matrix of expensive powers;
            # otherwise the stable fallback would still pay overflow costs.
            M = np.zeros(sq.shape, dtype=bool)
            needs_log = np.ones(len(sq), dtype=bool)
        else:
            lhs = left_ratio ** p
            rhs = 1.0 + ratio ** p
            M = lhs <= rhs[:, None]
            needs_log = ((ratio < np.finfo(float).tiny) & (small > 0)) | np.isinf(left_ratio).any(axis=1)
        needs_log &= np.isfinite(scale) & (scale > 0) & (small > 0)
        if needs_log.any():
            ds, ss, sm = d[needs_log], scale[needs_log], small[needs_log]
            # log1p((d-s)/s) is accurate around d=s. Use log differences only
            # when forming the ratio overflows; absolute logs lose nearby ULPs.
            relative = (ds - ss[:, None]) / ss[:, None]
            log_left = np.log1p(relative)
            overflow = np.isinf(relative)
            log_difference = np.log(ds) - np.log(ss[:, None])
            log_left[overflow] = log_difference[overflow]
            small_ratio = sm / ss
            log_small = np.log(small_ratio)
            underflow = small_ratio < np.finfo(float).tiny
            log_small[underflow] = np.log(sm[underflow]) - np.log(ss[underflow])
            log_right = np.logaddexp(0.0, p * log_small)
            resolved = p * log_left <= log_right[:, None]
            # Everything no farther than max(a,b) qualifies by construction.
            resolved |= ds <= ss[:, None]
            M[needs_log] = resolved
    M[scale == 0] = sq[scale == 0] == 0
    M[np.isinf(scale)] = True
    # When a=0 or beta=0 the nonzero term alone sets the exact threshold.
    one_term = small == 0
    M[one_term] = d[one_term] <= scale[one_term, None]
    return M


def _mask_from_sq(sq, alpha, beta, p, *, optimized=True, minsq=None):
    if minsq is None:
        if not np.isfinite(np.max(sq)):
            raise FloatingPointError('squared distances overflowed; rescale the input')
        minsq = _row_minimum(sq)
    if beta == 0 and p != 2.0:
        with np.errstate(over='ignore'):
            return np.sqrt(sq) <= (alpha * np.sqrt(minsq))[:, None]
    if p == 1.0:
        with np.errstate(over='ignore'):
            threshold = alpha * np.sqrt(minsq) + beta
        return np.sqrt(sq) <= threshold[:, None]
    if p == 2.0:
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            aa, bb = np.float64(alpha) ** 2, np.float64(beta) ** 2
            threshold = aa * minsq + bb
        safe = np.isfinite(aa) and (beta == 0 or bb > 0)
        if safe:
            return sq <= threshold[:, None]
        return _scaled_power_mask(sq, alpha, beta, p)
    if optimized and p < 0.25 and sq.shape[1] >= 4:
        # Admissibility is monotone in distance. A canonical comparison of
        # the minimum and maximum certifies all clusters at once for broad
        # small-p neighborhoods, without altering any threshold arithmetic.
        extrema = np.empty((len(sq), 2))
        extrema[:, 0] = minsq
        extrema[:, 1] = sq.max(axis=1)
        all_selected = _scaled_power_mask(extrema, alpha, beta, p)[:, 1]
        result = np.ones(sq.shape, dtype=bool)
        remaining = ~all_selected
        if remaining.any():
            result[remaining] = _scaled_power_mask(sq[remaining], alpha, beta, p)
        return result
    if not optimized or p < 0.25 or p > 64:
        return _scaled_power_mask(sq, alpha, beta, p)
    # One power/root per sample instead of one power per sample-cluster pair.
    # Points near the computed threshold use the canonical powered comparison.
    # The guard is for RECOMPUTATION, never an enlargement of admissibility.
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        a = alpha * np.sqrt(minsq)
        scale = np.maximum(a, beta)
        small = np.minimum(a, beta)
        threshold = scale * np.exp(np.log1p((small / scale) ** p) / p)
        threshold[scale == 0] = 0.0
        threshold_sq = threshold * threshold
        M = sq <= threshold_sq[:, None]
        guard = 128 * np.finfo(float).eps * (1.0 + 1.0 / p)
        # The wider interval encloses the old relative-error guard without
        # materializing two full Float64 matrices. It only selects rows for
        # canonical recomputation and never changes admissibility itself.
        lower = threshold_sq * (1.0 - 2.0 * guard)
        upper = threshold_sq * (1.0 + 2.0 * guard)
        near = (sq >= lower[:, None]) & (sq <= upper[:, None])
        rows = near.any(axis=1) | ~np.isfinite(threshold_sq) | np.isinf(scale) | (scale == 0)
    if rows.any():
        exact = _scaled_power_mask(sq[rows], alpha, beta, p)
        M[rows] = exact
    return M

_BLAS_CERTIFICATE_BUILD = None


def _blas_certificate_supported(X, C):
    """Only the reviewed NumPy/OpenBLAS double-GEMM path may certify masks.

    Unknown providers/versions keep working through direct distances. The
    linked NumPy build identity is cached; thread limits/controllers are not.
    """
    if (type(X) is not np.ndarray or type(C) is not np.ndarray
            or X.dtype != np.float64 or C.dtype != np.float64
            or not X.flags.c_contiguous or not C.flags.c_contiguous
            or not X.flags.aligned or not C.flags.aligned
            or len(X) < 32 or len(C) < 4 or X.shape[1] < 128
            or len(X) * len(C) * X.shape[1] <= 1000000
            or np.may_share_memory(X, C)):
        return False
    global _BLAS_CERTIFICATE_BUILD
    if _BLAS_CERTIFICATE_BUILD is None:
        _BLAS_CERTIFICATE_BUILD = False
        try:
            from threadpoolctl import threadpool_info
            blas = np.__config__.CONFIG['Build Dependencies']['blas']
            config = blas.get('openblas configuration', '')
            if (np.__version__ == '2.3.5' and blas.get('name') == 'scipy-openblas'
                    and blas.get('version') == '0.3.30'
                    and 'USE64BITINT' in config and 'DYNAMIC_ARCH' in config):
                _BLAS_CERTIFICATE_BUILD = any(
                    info.get('internal_api') == 'openblas'
                    and info.get('version') == '0.3.30'
                    and info.get('architecture') == 'SkylakeX'
                    and info.get('threading_layer') == 'pthreads'
                    and 'numpy.libs' in info.get('filepath', '')
                    and 'libscipy_openblas64_' in info.get('filepath', '')
                    for info in threadpool_info())
        except (ImportError, AttributeError, KeyError, TypeError, ValueError, OSError):
            pass
    return _BLAS_CERTIFICATE_BUILD


def _blas_squared_distance_bounds(X, C):
    """Outward Float64 bounds covering direct rounded squared distances."""
    d = X.shape[1]
    epsd = np.finfo(float).eps * (d + 2)
    if epsd >= 1.0 / 64.0:
        return np.zeros((len(X), len(C))), np.full((len(X), len(C)), np.inf)
    # Includes norm/dot accumulation and the direct sum of rounded squared
    # differences. The factor 32 is deliberately wider than the combined
    # standard gamma_d bounds; nextafter expands endpoints outward again.
    factor = 32.0 * epsd / (1.0 - epsd)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        xn = np.einsum('ij,ij->i', X, X)
        cn = np.einsum('ij,ij->i', C, C)
        q = X @ C.T
        q *= -2.0
        q += xn[:, None]
        q += cn[None, :]
        # Absolute padding also covers underflow/flush-to-zero contributions.
        error = factor * (xn[:, None] + cn[None, :]) + 64.0 * (d + 2) * np.finfo(float).tiny
        lower = np.maximum(0.0, np.nextafter(q - error, -np.inf))
        upper = np.nextafter(q + error, np.inf)
    return lower, upper


def _certified_blas_mask(X, C, alpha, beta, p, stats=None):
    """Certify p=1/2 masks using conservative Float64 distance intervals.

    Ambiguous rows and numerical-range cases use the existing direct-distance
    path. Other p values use that path in full. Default backend selection is
    unchanged; this is an explicitly requested BLAS backend.
    """
    from scipy.spatial.distance import cdist

    def direct(rows=None):
        points = X if rows is None else X[rows]
        exact = cdist(points, C, metric='sqeuclidean')
        minimum = _check_distance_range(points, C, exact)
        return _mask_from_sq(exact, alpha, beta, p, minsq=minimum)

    if stats is not None:
        stats['total_rows'] = len(X)
        stats['certificate_enabled'] = False
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        aa, bb = np.float64(alpha) ** 2, np.float64(beta) ** 2
    if not _blas_certificate_supported(X, C):
        if stats is not None: stats['direct_rows'] = len(X)
        return direct()
    if p not in (1.0, 2.0) or (p == 2.0 and not (np.isfinite(aa) and (beta == 0.0 or bb > 0.0))):
        if stats is not None: stats['direct_rows'] = len(X)
        return direct()
    if stats is not None: stats['certificate_enabled'] = True
    lower, upper = _blas_squared_distance_bounds(X, C)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        ambiguous = (lower <= np.finfo(float).tiny).any(axis=1) | ~np.isfinite(upper).all(axis=1)
        low_min, high_min = _row_minimum(lower), _row_minimum(upper)
        if p == 1.0:
            low_threshold = alpha * np.sqrt(low_min) + beta
            high_threshold = alpha * np.sqrt(high_min) + beta
            selected = np.sqrt(upper) <= low_threshold[:, None]
            rejected = np.sqrt(lower) > high_threshold[:, None]
        else:
            low_threshold = aa * low_min + bb
            high_threshold = aa * high_min + bb
            selected = upper <= low_threshold[:, None]
            rejected = lower > high_threshold[:, None]
        ambiguous |= ~(selected | rejected).all(axis=1)
    if ambiguous.any():
        selected[ambiguous] = direct(ambiguous)
    if stats is not None:
        stats['direct_rows'] = int(ambiguous.sum())
        stats['total_rows'] = len(X)
    return selected


def assign(X, centers, *, alpha=1.1, beta=0.0, p=1.0,
           backend='numpy', block_size=4096):
    """Return (normalized U, binary upper U), both with shape (C, N).

    Equal nearest distances all qualify, including zero distance ties.
    d is Euclidean distance; beta has the same units as d for every p.
    """
    if np.iscomplexobj(X) or np.iscomplexobj(centers):
        raise ValueError('complex input is not supported')
    X = np.asarray(X, dtype=np.float64, order='C')
    centers = np.asarray(centers, dtype=np.float64, order='C')
    if centers.ndim != 2:
        raise ValueError('centers must be a 2-D array')
    X, centers = _validate(X, len(centers), centers, alpha, beta, p, 1,
                           block_size, 0, check_sample_count=False)
    backend = _resolve_backend(backend)
    if backend == 'numba':
        from _numba_kernel import fused_step
        _, _, packed, U = fused_step(X, centers, alpha, beta, p, True, False)
        M = np.unpackbits(packed, axis=1, bitorder='little', count=len(centers)).astype(bool).T
        return U.T, M
    if backend == 'naive':
        _, _, _, U, M = _step_naive(X, centers, alpha, beta, p, True)
        return U, M
    # Assignment does not request centroids or cycle-state packing. Avoid
    # computing either, particularly the unused high-dimensional GEMM.
    n, k = len(X), len(centers)
    Uall = np.empty((k, n))
    Mall = np.empty((k, n), dtype=bool)
    if backend in ('scipy', 'blas'):
        from scipy.spatial.distance import cdist
    coincident = (X.shape[1] >= 16 and n * k * X.shape[1] >= (1 << 18)
                  and np.all(centers == centers[0]))
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        xb = X[start:stop]
        if coincident:
            sq = _distances_numpy(xb, centers[:1]) if backend == 'numpy' else cdist(xb, centers[:1], metric='sqeuclidean')
            _check_distance_range(xb, centers[:1], sq)
            M = np.ones((len(xb), k), dtype=bool)
        elif backend == 'blas':
            M = _certified_blas_mask(xb, centers, alpha, beta, p)
        else:
            sq = cdist(xb, centers, metric='sqeuclidean') if backend == 'scipy' and X.shape[1] != 1 else _distances_numpy(xb, centers)
            minsq = _check_distance_range(xb, centers, sq)
            M = _mask_from_sq(sq, alpha, beta, p, minsq=minsq)
        if backend != 'blas' and not coincident:
            U = sq
            np.copyto(U, M)
        else:
            U = M.astype(np.float64)
        U /= np.einsum('ij->i', U)[:, None]
        Uall[:, start:stop] = U.T
        Mall[:, start:stop] = M.T
    return Uall, Mall


def _resolve_backend(backend):
    if backend == 'auto':
        try:
            import scipy.spatial.distance
            return 'scipy'
        except ImportError:
            return 'numpy'
    if backend not in ('naive', 'numpy', 'scipy', 'numba', 'blas'):
        raise ValueError("backend must be 'auto', 'naive', 'numpy', 'scipy', 'blas', or 'numba'")
    if backend in ('scipy', 'blas'):
        try:
            import scipy.spatial.distance
        except ImportError as e:
            raise ImportError('scipy backend requires scipy; use numpy instead') from e
    if backend == 'numba':
        try:
            import numba
        except ImportError as e:
            raise ImportError('numba backend requires numba; use numpy or scipy instead') from e
    return backend


def _step_naive(X, C, alpha, beta, p, return_memberships):
    # Independent scratch oracle: intentionally simple full broadcast tensor.
    delta = X[:, None, :] - C[None, :, :]
    sq = np.sum(delta * delta, axis=2)
    _check_distance_range(X, C, sq)
    M = _mask_from_sq(sq, alpha, beta, p, optimized=False)
    U = M.astype(np.float64) / M.sum(axis=1)[:, None]
    mass = U.sum(axis=0)
    # Shift before weighted sums to avoid needless loss from a large common offset.
    origin = X[0]
    numerator = U.T @ (X - origin)
    out = C.copy()
    nz = mass > 0
    out[nz] = origin + numerator[nz] / mass[nz, None]
    packed = np.packbits(M, axis=1, bitorder='little')
    return out, int((~nz).sum()), packed, U.T if return_memberships else None, M.T if return_memberships else None


def _step(X, C, alpha, beta, p, backend, block_size, return_memberships, workspace=None):
    if backend == 'naive':
        return _step_naive(X, C, alpha, beta, p, return_memberships)
    if backend == 'numba':
        from _numba_kernel import fused_step
        out, empty, packed, U = fused_step(X, C, alpha, beta,
                                           p, return_memberships, True)
        if not np.isfinite(out).all():
            raise FloatingPointError('distance or accumulation overflowed; rescale the input')
        M = np.unpackbits(packed, axis=1, bitorder='little', count=len(C)).astype(bool).T if return_memberships else None
        return out, empty, packed, U.T if return_memberships else None, M
    n, k = len(X), len(C)
    if workspace is None:
        mass = np.zeros(k, dtype=np.float64)
        numerator = np.zeros_like(C)
    else:
        mass, numerator = workspace['mass'], workspace['numerator']
        mass.fill(0); numerator.fill(0)
    packed = np.empty((n, (k + 7) // 8), dtype=np.uint8)
    Uall = np.empty((k, n)) if return_memberships else None
    Mall = np.empty((k, n), dtype=bool) if return_memberships else None
    origin = X[0]
    if backend in ('scipy', 'blas'):
        from scipy.spatial.distance import cdist
    coincident = (X.shape[1] >= 16 and n * k * X.shape[1] >= (1 << 18)
                  and np.all(C == C[0]))
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        xb = X[start:stop]
        if coincident:
            sq = _distances_numpy(xb, C[:1]) if backend == 'numpy' else cdist(xb, C[:1], metric='sqeuclidean')
            _check_distance_range(xb, C[:1], sq)
            M = np.ones((len(xb), k), dtype=bool)
        elif backend == 'blas':
            M = _certified_blas_mask(xb, C, alpha, beta, p)
        else:
            if workspace is None:
                sq = cdist(xb, C, metric='sqeuclidean') if backend == 'scipy' and X.shape[1] != 1 else _distances_numpy(xb, C)
            else:
                sq = workspace['distance'][:stop-start]
                if backend == 'scipy' and X.shape[1] != 1:
                    cdist(xb, C, metric='sqeuclidean', out=sq)
                else:
                    sq[:] = _distances_numpy(xb, C)
            minsq = _check_distance_range(xb, C, sq)
            M = _mask_from_sq(sq, alpha, beta, p, minsq=minsq)
        if workspace is None:
            if backend != 'blas' and not coincident:
                U = sq
                np.copyto(U, M)
            else:
                U = M.astype(np.float64)
            U /= np.einsum('ij->i', U)[:, None]
            mass += np.einsum('ij->j', U)
            numerator += U.T @ (xb - origin)
        else:
            # Distances have been consumed by the independent Boolean mask.
            # Reuse their Float64 buffer for normalized memberships.
            U = workspace['distance'][:stop-start]
            np.copyto(U, M)
            U /= np.einsum('ij->i', U)[:, None]
            mass += np.einsum('ij->j', U)
            np.matmul(U.T, xb - origin, out=workspace['moment'])
            numerator += workspace['moment']
        packed[start:stop] = np.packbits(M, axis=1, bitorder='little')
        if return_memberships:
            Uall[:, start:stop] = U.T
            Mall[:, start:stop] = M.T
    out = C.copy()
    nz = mass > 0
    out[nz] = origin + numerator[nz] / mass[nz, None]
    if not np.isfinite(out).all():
        raise FloatingPointError('center accumulation overflowed; rescale the input')
    return out, int((~nz).sum()), packed, Uall, Mall


def fit(X, n_clusters, *, alpha=1.1, beta=0.0, p=1.0,
        init=None, seed=0, max_iter=300, backend='auto', block_size=4096,
        return_memberships=True, cycle_window=16):
    """Fit exact binary-threshold normalized-membership rough c-means.

    M[c,i] = 1{d(x_i,c_c)^p <= alpha^p * min_j d(x_i,c_j)^p + beta^p}
    U[c,i] = M[c,i] / sum_j M[j,i]
    c_c = sum_i U[c,i] * x_i / sum_i U[c,i]

    Initialization samples distinct DATA INDICES without replacement using
    np.random.default_rng(seed). Duplicate rows may still give equal centers.
    Empty clusters retain the previous center. The stopping rule is identical
    consecutive masks (a fixed point), exact repeated centers, detected mask
    cycles within cycle_window, or max_iter; objective decrease is NOT assumed.
    cycle_window=0 disables cycle detection. At a cycle or max_iter, converged
    is False and returned memberships are freshly assigned to returned centers.
    Streaming backends retain only packed masks when memberships are omitted.
    """
    X, init = _validate(X, n_clusters, init, alpha, beta, p, max_iter,
                        block_size, cycle_window)
    backend = _resolve_backend(backend)
    if init is None:
        indices = np.random.default_rng(seed).choice(len(X), n_clusters, replace=False)
        C = X[indices].copy()
    else:
        indices = None
        C = init
    workspace = None
    if backend == 'scipy':
        rows = min(len(X), block_size)
        workspace = dict(mass=np.empty(n_clusters), numerator=np.empty_like(C),
                         distance=np.empty((rows, n_clusters)),
                         moment=np.empty_like(C))
    previous = None
    history = deque(maxlen=cycle_window)
    converged = False
    reason = 'max_iter'
    cycle_length = None
    empties = 0
    for iteration in range(1, max_iter + 1):
        updated, empty, packed, _, _ = _step(X, C, alpha, beta, p,
                                             backend, block_size, False, workspace)
        empties += empty
        if np.array_equal(C, updated) or (previous is not None and np.array_equal(previous, packed)):
            C = updated
            converged = True
            reason = 'fixed_point'
            break
        if cycle_window:
            for old_iteration, old_packed, old_updated in history:
                if np.array_equal(old_packed, packed) and np.array_equal(old_updated, updated):
                    cycle_length = iteration - old_iteration
                    reason = 'cycle'
                    break
            if cycle_length is not None:
                C = updated
                break
            history.append((iteration, packed, updated.copy()))
        previous = packed
        C = updated
    # Always align exported U with returned centers, even for nonconverged runs.
    final_updated, _, final_packed, U, M = _step(X, C, alpha, beta, p,
                                               backend, block_size, return_memberships, workspace)
    fixed = np.array_equal(packed, final_packed)
    if fixed:
        # Same masks induce the same update; expose that exact update.
        C = final_updated
        converged = True
        reason = 'fixed_point'
        cycle_length = None
    return RoughCMeansResult(C, U, M, iteration, converged, reason, cycle_length,
                            indices, backend, float(p), float(alpha), float(beta),
                            empties, fixed)


def fit_rcm(X, n_clusters, **kwargs):
    """RCM shorthand: exactly ExRCM with p=1."""
    if 'p' in kwargs and kwargs['p'] != 1:
        raise ValueError('fit_rcm is ExRCM p=1; use fit for other p')
    kwargs['p'] = 1.0
    return fit(X, n_clusters, **kwargs)


fit_exrcm = fit
