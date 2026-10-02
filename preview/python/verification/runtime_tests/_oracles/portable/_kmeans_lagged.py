"""Lloyd k-means with portable direct-distance NumPy/SciPy and optional Numba cores."""
import numpy as np
from .validation import matrix_input, matrix, positive_int
from .policy import policy_or_default
from ._optional import numba_available


def _numpy_lloyd(X, init, max_iter, policy):
    n, d = X.shape
    k = len(init)
    b = policy.rows_for(16 * k, n)
    centers = init.copy()
    labels = np.full(n, -1, dtype=np.int64)
    scores, delta = np.empty((b, k)), np.empty((b, k))
    for iteration in range(1, max_iter + 1):
        changed = False
        for start in range(0, n, b):
            stop = min(start + b, n)
            S, T = scores[:stop-start], delta[:stop-start]
            S.fill(0.0)
            # Increasing feature order matches the strict direct-distance core.
            for j in range(d):
                np.subtract(X[start:stop, j, None], centers[None, :, j], out=T)
                np.square(T, out=T)
                S += T
            if not np.isfinite(S).all():
                raise ValueError("squared distances overflowed; rescale input")
            assigned = S.argmin(axis=1)
            changed |= not np.array_equal(assigned, labels[start:stop])
            labels[start:stop] = assigned
        counts = np.bincount(labels, minlength=k)
        active = counts > 0
        for j in range(d):
            sums = np.bincount(labels, weights=X[:, j], minlength=k)
            centers[active, j] = sums[active] / counts[active]
        if not np.isfinite(centers).all():
            raise ValueError("center sums overflowed; rescale input")
        if not changed:
            break
    return {"centers": centers, "labels": labels, "n_iter": iteration,
            "scratch_rows": b}



def _scipy_lloyd(X, init, max_iter, policy):
    """Strict direct distances with a C-compiled SciPy block and NumPy means."""
    from scipy.spatial.distance import cdist
    n, d = X.shape
    k = len(init)
    # CSR multiplication sums each cluster in original row order while reading
    # contiguous feature rows. It wins only once feature-wise bincount scans
    # become expensive. Bound its conservative construction/output workspace.
    sparse_fixed = 64 * n + 16 * k * d + 32 * (k + 1)
    use_sparse = (d >= 32 and n * d >= 32768
                  and policy.max_scratch_bytes >= sparse_fixed + 8 * k)
    if use_sparse:
        from scipy.sparse import csr_matrix
        b = min(policy.block_rows, n, (policy.max_scratch_bytes - sparse_fixed) // (8 * k))
        unit_weights = np.ones(n, dtype=np.float64)
        row_indices = np.arange(n, dtype=np.int64)
    else:
        b = policy.rows_for(8 * k, n)
    centers = init.copy()
    labels = np.full(n, -1, dtype=np.int64)
    scores = np.empty((b, k))
    for iteration in range(1, max_iter + 1):
        changed = False
        for start in range(0, n, b):
            stop = min(start + b, n)
            S = scores[:stop-start]
            cdist(X[start:stop], centers, 'sqeuclidean', out=S)
            if not np.isfinite(S).all():
                raise ValueError("squared distances overflowed; rescale input")
            assigned = S.argmin(axis=1)
            changed |= not np.array_equal(assigned, labels[start:stop])
            labels[start:stop] = assigned
        counts = np.bincount(labels, minlength=k)
        active = counts > 0
        if use_sparse:
            membership = csr_matrix((unit_weights, (labels, row_indices)), shape=(k, n))
            sums = membership @ X
            np.divide(sums, counts[:, None], out=centers, where=active[:, None])
        else:
            for j in range(d):
                sums = np.bincount(labels, weights=X[:, j], minlength=k)
                centers[active, j] = sums[active] / counts[active]
        if not np.isfinite(centers).all():
            raise ValueError("center sums overflowed; rescale input")
        if not changed:
            break
    return {"centers": centers, "labels": labels, "n_iter": iteration,
            "scratch_rows": b,
            "centroid_reducer": "scipy-csr-row-order" if use_sparse else "numpy-bincount",
            "primary_scratch_budgeted_bytes": (sparse_fixed if use_sparse else 0) + 8 * b * k}

def fit_kmeans(X, init, *, max_iter=20, backend="auto", policy=None):
    """Fit from explicit initial centers; no random seeding is hidden.

    auto chooses strict Numba SIMD when importable, otherwise NumPy. It is
    an availability default, not an optimal-backend predictor. Empty centers
    are retained, exact distance ties choose the first center, and centers are
    updated before label-convergence testing. Parallel reduction changes low
    bits with thread count; no cross-thread bit-identical claim is made.

    All built-in cores use float64. Returned labels precede the final center
    update (the lab's Lloyd contract). max_scratch_bytes bounds distance/private
    accumulation buffers, excluding output arrays and runtime workspace.
    """
    X, prepared = matrix_input(X, dtype=np.float64)
    init = matrix(init, dtype=np.float64, name="init")
    if X.shape[1] != init.shape[1]:
        raise ValueError("X and init must have the same feature dimension")
    # Finite inputs alone do not guarantee finite direct squared distances.
    # A conservative per-feature range bound covers every later centroid too.
    with np.errstate(over="ignore", invalid="ignore"):
        lower, upper = (prepared.feature_bounds() if prepared is not None else
                        (X.min(axis=0), X.max(axis=0)))
        extent = np.maximum(upper, init.max(axis=0)) - np.minimum(lower, init.min(axis=0))
        bound = np.dot(extent, extent)
    if not np.isfinite(bound):
        raise ValueError("squared distance range may overflow; rescale input")
    max_iter = positive_int(max_iter, "max_iter")
    policy = policy_or_default(policy)
    requested = backend
    if backend == "auto":
        backend = "numba" if numba_available() else "numpy"
    if backend in ("numba", "numba_blas", "numba_blas_vector"):
        if not numba_available():
            raise ImportError("Numba is unavailable or disabled; use backend='numpy'")
        try:
            if backend == "numba":
                from ._backends.kmeans_numba import _register8, _select_vector_width
            elif backend == "numba_blas":
                from ._backends.kmeans_blas_numba import _lloyd
            else:
                if np.dtype(np.intp).itemsize != 8:
                    raise RuntimeError("numba_blas_vector requires a 64-bit platform; use numba_blas")
                from ._backends.kmeans_blas_vector import _lloyd
            from numba import get_num_threads
        except ImportError:
            if requested != "auto":
                raise
            backend = "numpy"
    if backend == "numba":
        n, d = X.shape
        k = len(init)
        fixed = 8 * d * (((k + 7) // 8) * 8)
        per_block = 8 * (k * d + k + 1)
        available = policy.max_scratch_bytes - fixed
        if available < per_block:
            raise ValueError("max_scratch_bytes cannot hold register8 private scratch")
        with policy.activate(numba=True):
            blocks = min(get_num_threads(), n, available // per_block)
            vector_width = _select_vector_width(k, d)
            padded = max(((k + 7) // 8) * 8,
                         ((k + vector_width - 1) // vector_width) * vector_width)
            # Wider padding must not reduce the original reduction partition
            # count or exceed the caller's existing primary scratch allowance.
            if 8 * d * padded + blocks * per_block > policy.max_scratch_bytes:
                vector_width = 8
            centers, labels, iteration = _register8(X, init, max_iter, blocks, vector_width)
        if not np.isfinite(centers).all():
            raise ValueError("center sums overflowed; rescale input")
        result = {"centers": centers, "labels": labels, "n_iter": int(iteration),
                  "private_blocks": int(blocks)}
    elif backend in ("numba_blas", "numba_blas_vector"):
        from ._threadpools import threadpool_context
        n, d = X.shape
        k = len(init)
        with np.errstate(over="ignore", invalid="ignore"):
            xnorm = (np.einsum("ij,ij->i", X, X) if prepared is None else
                     prepared.norms(preprocessing="original", contract="squared-euclidean"))
        if not np.isfinite(xnorm).all():
            raise ValueError("squared norms overflowed; rescale input or use backend='numpy'")
        # Private sums, padded counts/stats, scores, and center norms are capped.
        per_block_fixed = 8 * (k * d + k + 16)
        available = policy.max_scratch_bytes - 8 * k
        if available < per_block_fixed + 8 * k:
            raise ValueError("max_scratch_bytes cannot hold one BLAS/Numba private block")
        with policy.activate(numba=True, blas=False):
            blocks = min(get_num_threads(), n, available // (per_block_fixed + 8 * k))
            room_per_block = available // blocks - per_block_fixed
            rows = min(policy.block_rows, (n + blocks - 1) // blocks, room_per_block // (8 * k))
            # Numba owns outer parallelism; prevent nested BLAS oversubscription.
            with threadpool_context(limits=1, user_api="blas"):
                centers, labels, iteration, fallbacks = _lloyd(X, init, xnorm, max_iter, blocks, rows)
        if not np.isfinite(centers).all():
            raise ValueError("center sums overflowed; rescale input")
        result = {"centers": centers, "labels": labels, "n_iter": int(iteration),
                  "private_blocks": int(blocks), "scratch_rows": int(rows),
                  "guard_fallbacks": int(fallbacks),
                  "numerical_contract": "float64-blas-near-tie-direct-heuristic-first-tie",
                  "score_selector": "llvm-vector8" if backend == "numba_blas_vector" else "scalar"}
    elif backend == "scipy":
        with policy.activate():
            result = _scipy_lloyd(X, init, max_iter, policy)
    elif backend == "numpy":
        with policy.activate():
            result = _numpy_lloyd(X, init, max_iter, policy)
    else:
        from .registry import get_backend
        callback, contract = get_backend("kmeans", backend)
        with policy.activate():
            result = callback(X=X, init=init, max_iter=max_iter, policy=policy,
                              prepared=prepared)
        result = dict(result)
        result["numerical_contract"] = contract
    result["backend"] = backend
    result.setdefault("numerical_contract", "float64-direct-first-tie-label-convergence")
    result["prepared_input"] = prepared is not None
    return result
