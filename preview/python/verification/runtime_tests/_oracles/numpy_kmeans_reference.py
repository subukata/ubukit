"""Readable NumPy-only Lloyd reference, including final labels and inertia.

No sklearn, SciPy, Numba or library kernel is used. Distances accumulate features
in increasing order; empty centers are retained and the first center wins ties.
Final labels are evaluated at the returned centers. The float64 inertia sum can
differ in low bits from sklearn's sqrt-then-square or JavaScript's scalar sum.
"""
import numpy as np


def fit_kmeans(X, init, *, max_iter=10, block_rows=128):
    X = np.ascontiguousarray(X, dtype=np.float64)
    init = np.ascontiguousarray(init, dtype=np.float64)
    if X.ndim != 2 or init.ndim != 2 or min(*X.shape, *init.shape) < 1 or X.shape[1] != init.shape[1]:
        raise ValueError('matching nonempty two-dimensional matrices required')
    if not np.isfinite(X).all() or not np.isfinite(init).all():
        raise ValueError('finite matrices required')
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError('positive integer max_iter required')
    with np.errstate(over='ignore', invalid='ignore'):
        extent = np.maximum(X.max(axis=0), init.max(axis=0)) - np.minimum(X.min(axis=0), init.min(axis=0))
        if not np.isfinite(np.dot(extent, extent)):
            raise ValueError('squared distance range may overflow; rescale input')
    n, d = X.shape
    k = len(init)
    rows = min(block_rows, n)
    centers = init.copy()
    labels = np.full(n, -1, dtype=np.int64)
    scores = np.empty((rows, k), dtype=np.float64)
    delta = np.empty_like(scores)

    def distances(start, stop):
        values = scores[:stop-start]
        temporary = delta[:stop-start]
        values.fill(0.0)
        for feature in range(d):
            np.subtract(X[start:stop, feature, None], centers[None, :, feature], out=temporary)
            np.square(temporary, out=temporary)
            values += temporary
        if not np.isfinite(values).all():
            raise ValueError('squared distances overflowed; rescale input')
        return values

    for iteration in range(1, max_iter + 1):
        changed = False
        for start in range(0, n, rows):
            stop = min(start + rows, n)
            assigned = distances(start, stop).argmin(axis=1)
            changed |= not np.array_equal(assigned, labels[start:stop])
            labels[start:stop] = assigned
        counts = np.bincount(labels, minlength=k)
        active = counts > 0
        for feature in range(d):
            sums = np.bincount(labels, weights=X[:, feature], minlength=k)
            centers[active, feature] = sums[active] / counts[active]
        if not np.isfinite(centers).all():
            raise ValueError('center sums overflowed; rescale input')
        if not changed:
            break
    core_labels = labels.copy()
    nearest_squared = np.empty(n, dtype=np.float64)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        values = distances(start, stop)
        labels[start:stop] = values.argmin(axis=1)
        nearest_squared[start:stop] = values[np.arange(stop-start), labels[start:stop]]
    inertia = float(np.sum(nearest_squared, dtype=np.float64))
    if not np.isfinite(inertia):
        raise ValueError('inertia overflowed; rescale input')
    return {'centers': centers, 'labels': labels, 'core_labels': core_labels,
            'inertia': inertia, 'n_iter': iteration}
