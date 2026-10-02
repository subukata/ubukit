"""Standard FCM, with identical contracts for reference and accelerated paths."""
from numbers import Integral

import numpy as np

_MEMBERSHIP_BLOCK_ROWS = None


def memberships_from_squared_distances(d2, m=2.0, out=None):
    """Normalize inverse powers in O(N K), with exact coincident-center handling.

    Dividing by the row minimum avoids overflow even when m is near 1.
    A point at multiple centers is split equally between precisely those centers.
    """
    if np.iscomplexobj(d2):
        raise ValueError("d2 must be real-valued")
    d2 = np.asarray(d2, dtype=np.float64)
    if d2.ndim != 2 or not d2.shape[1]:
        raise ValueError("d2 must have shape (n_samples, n_clusters)")
    if not np.isfinite(m) or m <= 1:
        raise ValueError("m must be finite and greater than 1")
    if np.any(d2 < 0) or not np.isfinite(d2).all():
        raise ValueError("squared distances must be finite and nonnegative")
    if out is not None:
        if not isinstance(out, np.ndarray) or out.shape != d2.shape or out.dtype != np.float64:
            raise ValueError("out must be a float64 ndarray matching d2.shape")
        if np.shares_memory(d2, out):
            d2 = d2.copy()
    return _memberships(d2, float(m), out)


def _row_minimum(values):
    """SIMD column passes avoid NumPy's short-row reduction overhead."""
    if values.shape[1] <= 16:
        minimum = values[:, 0].copy()
        for col in range(1, values.shape[1]):
            np.minimum(minimum, values[:, col], out=minimum)
        return minimum[:, None]
    return values.min(axis=1, keepdims=True)


def _memberships(d2, m, out=None, minimum=None, *, bounds=None, row_ones=None, denominator=None):
    if out is None:
        out = np.empty_like(d2)
    if not d2.size:
        if not out.flags.writeable:
            raise ValueError("output array is read-only")
        return out
    # Reciprocal normalization is equivalent at m=2. Restrict it to normal
    # reciprocals whose row sums cannot overflow; exceptional scales retain
    # the row-minimum formula and all exact zero-distance semantics.
    if m == 2.0 and bounds is not None:
        smallest, largest = bounds
        tiny = np.finfo(float).tiny
        if smallest > tiny * d2.shape[1] and largest < 1.0 / tiny:
            np.reciprocal(d2, out=out)
            if row_ones is None:
                out /= np.einsum("ij->i", out)[:, None]
            else:
                np.matmul(out, row_ones, out=denominator)
                out /= denominator[:, None]
            return out
    if minimum is None:
        minimum = _row_minimum(d2)
    has_zero = (minimum.min() == 0) if bounds is None else bounds[0] == 0
    # Leave zero-distance rows zero here, then resolve their exact minimizers.
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        np.divide(minimum, d2, out=out)
        if m != 2.0:
            exponent = 1.0 / (m - 1.0)
            # The ratio may underflow even when its fractional power does not.
            rows = cols = stable = None
            if has_zero or np.min(out) < np.finfo(float).tiny:
                small = (out < np.finfo(float).tiny) & (d2 > 0) & (minimum > 0)
                rows, cols = np.nonzero(small)
                stable = np.exp((np.log(minimum[rows, 0]) - np.log(d2[rows, cols])) * exponent)
            close_rows = close_cols = None
            if m - 1.0 < 1e-4:
                # Beyond this radius the weight is provably below the
                # smallest subnormal: log1p(1024*delta)/delta > 970 for
                # 0 < delta=m-1 < 1e-4. Protect every representable near-tie
                # without evaluating log1p/exp for certain-zero weights.
                close = ((d2 - minimum) <= minimum * (1024.0 * (m - 1.0))) & (minimum > 0)
                close_rows, close_cols = np.nonzero(close)
                stable_close = np.exp(-np.log1p((d2[close_rows, close_cols] - minimum[close_rows, 0])
                                                / minimum[close_rows, 0]) * exponent)
            if exponent == 0.5:
                np.sqrt(out, out=out)
            elif exponent == 2.0:
                np.square(out, out=out)
            else:
                np.power(out, exponent, out=out)
            if rows is not None:
                out[rows, cols] = stable
            if close_rows is not None:
                out[close_rows, close_cols] = stable_close
    if has_zero:
        zero_rows = minimum[:, 0] == 0
        out[zero_rows] = d2[zero_rows] == 0
    if row_ones is None:
        out /= np.einsum("ij->i", out)[:, None]
    else:
        np.matmul(out, row_ones, out=denominator)
        out /= denominator[:, None]
    return out


def _validate(X, init, n_clusters, m, max_iter, tol, threads, random_state):
    if np.iscomplexobj(X) or (init is not None and np.iscomplexobj(init)):
        raise ValueError("X and init must be real-valued")
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("X must be a nonempty finite 2D array")
    if not np.isfinite(m) or m <= 1:
        raise ValueError("m must be finite and greater than 1")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if n_clusters is not None and (isinstance(n_clusters, bool) or
            not isinstance(n_clusters, Integral) or n_clusters < 1):
        raise ValueError("n_clusters must be a positive integer")
    if init is not None:
        u = np.array(init, dtype=np.float64, order="C", copy=True)
        if u.ndim != 2 or u.shape[0] != len(x) or not u.shape[1]:
            raise ValueError("init must have shape (n_samples, n_clusters)")
        if not np.isfinite(u).all() or np.any(u < 0) or np.any(u.max(axis=1) <= 0):
            raise ValueError("init must be finite, nonnegative, with positive row sums")
        if n_clusters is not None and n_clusters != u.shape[1]:
            raise ValueError("n_clusters and init shape disagree")
        u /= u.max(axis=1, keepdims=True)
        u /= u.sum(axis=1, keepdims=True)
    else:
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, Integral) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer when init is omitted")
        u = np.random.default_rng(random_state).random((len(x), n_clusters))
        u /= u.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(x), u


def _weights(u, m, out, ones=None):
    if m == 2.0:
        np.square(u, out=out)
    elif m == 3.0:
        np.square(u, out=out)
        np.multiply(out, u, out=out)
    elif m == 1.5:
        np.sqrt(u, out=out)
        np.multiply(out, u, out=out)
    else:
        np.power(u, m, out=out)
    sums = np.einsum("ij->j", out) if ones is None else ones @ out
    # Very large m can underflow an entire nonempty cluster. Scaling each
    # cluster's weights is mathematically neutral in its weighted mean.
    weak = sums < np.finfo(float).tiny
    if np.any(weak):
        maxima = u[:, weak].max(axis=0)
        valid = maxima > 0
        values = np.zeros((len(u), len(maxima)))
        with np.errstate(under="ignore"):
            values[:, valid] = (u[:, weak][:, valid] / maxima[valid]) ** m
        out[:, weak] = values
        sums[weak] = values.sum(axis=0)
    return sums



def _objective_parts(u, d2, m):
    weights = u ** m
    terms = weights * d2
    weak = (weights < np.finfo(float).tiny) & (u > 0) & (d2 > 0)
    weak_logsum = -np.inf
    if np.any(weak):
        # Aggregate before exp: individually unrepresentable terms can have
        # a representable sum. Kept logarithmic across finalizer chunks too.
        from scipy.special import logsumexp
        weak_logsum = float(logsumexp(m * np.log(u[weak]) + np.log(d2[weak])))
        terms[weak] = 0.0
    return float(terms.sum()), weak_logsum


def _objective(u, d2, m):
    normal, weak_logsum = _objective_parts(u, d2, m)
    return normal + float(np.exp(weak_logsum))


def _distance(x, centers, backend, xnorm=None, out=None):
    if x.shape[1] == 1:
        if out is None:
            out = np.empty((len(x), len(centers)))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            np.subtract(x, centers.T, out=out)
            np.square(out, out=out)
        return out
    if backend == "numpy":
        difference = x[:, None, :] - centers[None, :, :]
        return np.sum(difference * difference, axis=2)
    if backend == "scipy":
        from scipy.spatial.distance import cdist
        return cdist(x, centers, metric="sqeuclidean", out=out)
    # Retain BOTH norms: removing ||x||² changes fuzzy memberships.
    np.matmul(x, centers.T, out=out)
    out *= -2.0
    cnorm = np.einsum("ij,ij->i", centers, centers)
    out += xnorm[:, None]
    out += cnorm[None, :]
    # Near cancellation, recompute direct differences rather than clipping
    # a small positive distance to an invented zero.
    suspect = out <= 64 * np.finfo(float).eps * (xnorm[:, None] + cnorm[None, :])
    rows, cols = np.nonzero(suspect)
    for start in range(0, len(rows), 4096):
        r, c = rows[start:start+4096], cols[start:start+4096]
        diff = x[r] - centers[c]
        out[r, c] = np.einsum("ij,ij->i", diff, diff)
    return out


def fit_fcm(X, n_clusters=None, *, init=None, m=2.0, max_iter=300,
            tol=1e-5, backend="auto", threads=1, random_state=None,
            return_history=False):
    """Fit squared-Euclidean fuzzy c-means using float64 arithmetic.

    Parameters
    ----------
    X : (N, D) array
    init : (N, K) array, optional
        Nonnegative membership rows, normalized on a copy. Shared explicit
        initialization is recommended for comparisons. No input is mutated.
    backend : 'auto', 'numpy', 'scipy', 'blas', 'numba', 'numba_parallel'
        Auto currently means SciPy: portable, deterministic, no JIT startup.
        numpy is the straightforward scratch baseline (broadcast distances).
    tol : float >= 0
        Stop when absolute Frobenius norm ||U_new-U_old|| < tol.
        tol=0 means exactly max_iter iterations (also for a stationary point).
    threads : int >= 1
        BLAS thread cap; Numba parallel workers also use this number. A serial
        Numba loop stays serial. Thread settings are restored after fitting.

    Returns
    -------
    dict
        centers (K,D), membership (N,K), labels (N,), objective, fpc,
        n_iter, converged, delta, backend, and optionally objective_history.
        Each iteration updates centers from U_old, then U from those centers.
        The returned objective is evaluated at the RETURNED (centers,U) pair,
        not the previous-membership objective reported by some libraries.
        The centers are not silently updated again after the stopping test.
        Empty clusters retain their previous center (initially the data mean).
    """
    if backend == "auto":
        backend = "scipy"
    if backend not in {"numpy", "scipy", "blas", "numba", "numba_parallel"}:
        raise ValueError("unknown FCM backend")
    x, u = _validate(X, init, n_clusters, m, max_iter, tol, threads, random_state)
    m = float(m)
    # Translation first prevents offset-dominated weighted sums and Gram norms.
    origin = x[0].copy()
    original_x = x
    x = np.ascontiguousarray(x - origin)
    if not np.isfinite(x).all():
        raise ValueError("input range overflow; rescale X")
    magnitude = np.max(np.abs(x))
    scale = magnitude if 0 < magnitude < 1e-100 else 1.0
    if scale != 1.0:
        x /= scale
    if magnitude > np.sqrt(np.finfo(float).max / (4 * x.shape[1])):
        raise ValueError("squared distance range overflow; rescale X")
    n, k = u.shape
    weights = np.empty_like(u)
    unew = np.empty_like(u)
    ones_n, ones_k = np.ones(n), np.ones(k)
    # BLAS distance products benefit from bounded blocks. SciPy's direct
    # distance kernel stays whole-array for moderate arrays to avoid call
    # overhead; large arrays cap the scratch at roughly 32 MiB.
    if _MEMBERSHIP_BLOCK_ROWS is not None:
        block_rows = min(n, _MEMBERSHIP_BLOCK_ROWS)
    elif backend == "blas":
        block_rows = min(n, 16384)
    elif backend == "numpy":
        block_rows = min(n, max(1, (1 << 20) // (k * x.shape[1])))
    else:
        block_rows = min(n, max(1, (1 << 22) // k))
    denominator = np.empty(block_rows)
    centers = np.repeat(x.mean(axis=0, keepdims=True), k, axis=0)
    newcenters = np.empty_like(centers)
    d2 = np.empty((block_rows, k)) if backend in {"scipy", "blas"} else None
    xnorm = np.einsum("ij,ij->i", x, x) if backend == "blas" else None
    history = []
    numba_state = None
    if backend.startswith("numba"):
        try:
            from . import _numba
        except ImportError as exc:
            raise ImportError("Numba backend requires pip install ubukit-fcm[numba]") from exc
        update = _numba.update_membership_parallel if backend == "numba_parallel" else _numba.update_membership_serial
        if backend == "numba_parallel":
            numba_state = _numba.set_threads(int(threads))
    # cdist is also required by the returned-pair finalizer for every backend.
    from scipy.spatial.distance import cdist
    from ._threadpools import threadpool_context
    try:
        with threadpool_context(limits=int(threads), user_api="blas"):
            for iteration in range(1, int(max_iter) + 1):
                sums = _weights(u, m, weights, ones_n)
                np.matmul(weights.T, x, out=newcenters)
                nonempty = sums > 0
                if np.all(nonempty):
                    newcenters /= sums[:, None]
                else:
                    newcenters[nonempty] /= sums[nonempty, None]
                    newcenters[~nonempty] = centers[~nonempty]
                centers, newcenters = newcenters, centers
                if backend.startswith("numba"):
                    delta2 = update(x, centers, u, unew, m)
                    objective = None
                    delta = float(np.sqrt(delta2))
                    if return_history:
                        from scipy.spatial.distance import cdist
                        objective = _objective(unew, cdist(x, centers, "sqeuclidean"), m)
                else:
                    delta2 = 0.0
                    objective_normal = 0.0
                    objective_weak = -np.inf
                    for start in range(0, n, block_rows):
                        stop = min(start + block_rows, n)
                        xb = x[start:stop]
                        target = d2[:stop-start] if d2 is not None else None
                        norms = xnorm[start:stop] if xnorm is not None else None
                        distances = _distance(xb, centers, backend, norms, target)
                        largest = np.max(distances)
                        if not np.isfinite(largest):
                            raise ValueError("squared distance overflow; rescale X")
                        minimum = None if m == 2.0 else _row_minimum(distances)
                        smallest = np.min(distances) if minimum is None else minimum.min()
                        if smallest == 0 and minimum is None:
                            minimum = _row_minimum(distances)
                        zero_points = np.flatnonzero(minimum[:, 0] == 0) if smallest == 0 else ()
                        if len(zero_points):
                            zero_rows, zero_cols = np.nonzero(distances[zero_points] == 0)
                            for offset in range(0, len(zero_rows), 4096):
                                rr = zero_points[zero_rows[offset:offset+4096]]
                                cc = zero_cols[offset:offset+4096]
                                if np.any(xb[rr] != centers[cc]):
                                    raise ValueError("squared distance underflow; rescale input dynamic range")
                        next_block = unew[start:stop]
                        _memberships(distances, m, next_block, minimum=minimum,
                                     bounds=(smallest, largest), row_ones=ones_k,
                                     denominator=denominator[:stop-start])
                        difference = weights[start:stop]
                        np.subtract(next_block, u[start:stop], out=difference)
                        flat = difference.ravel()
                        delta2 += float(np.dot(flat, flat))
                        if return_history:
                            normal, weak = _objective_parts(next_block, distances, m)
                            objective_normal += normal
                            objective_weak = float(np.logaddexp(objective_weak, weak))
                    delta = float(np.sqrt(delta2))
                    objective = objective_normal + float(np.exp(objective_weak)) if return_history else None
                u, unew = unew, u
                if return_history:
                    history.append(float(objective))
                if not np.isfinite(delta):
                    raise ValueError("numeric overflow or underflow in FCM; rescale input dynamic range")
                if delta < tol:
                    break
    finally:
        if numba_state is not None:
            _numba.set_threads(numba_state)
    # Coordinate rounding in centers+origin can matter at huge translations.
    # Evaluate the published objective at exactly the published float64 pair.
    centers *= scale
    centers += origin
    from scipy.spatial.distance import cdist
    objective = 0.0
    weak_logsum = -np.inf
    for start in range(0, n, 8192):
        stop = min(start + 8192, n)
        if scale == 1.0:
            distance = cdist(original_x[start:stop], centers, "sqeuclidean")
        else:
            distance = cdist((original_x[start:stop] - origin) / scale,
                             (centers - origin) / scale, "sqeuclidean")
        normal, weak_block = _objective_parts(u[start:stop], distance, m)
        objective += normal
        weak_logsum = float(np.logaddexp(weak_logsum, weak_block))
    objective = (objective + float(np.exp(weak_logsum))) * scale * scale
    if not np.isfinite(objective):
        raise ValueError("objective overflow; rescale X")
    result = dict(centers=centers, membership=u, labels=u.argmax(axis=1),
                  objective=objective, fpc=float(np.einsum("ij,ij->", u, u) / n),
                  n_iter=iteration, converged=bool(delta < tol), delta=delta,
                  backend=backend)
    if return_history:
        result["objective_history"] = np.asarray(history) * scale * scale
    return result


def fit_fcm_numpy(X, n_clusters=None, **kwargs):
    """Straightforward NumPy scratch reference; same public result contract."""
    if "backend" in kwargs:
        raise TypeError("fit_fcm_numpy fixes backend='numpy'")
    return fit_fcm(X, n_clusters, backend="numpy", **kwargs)
