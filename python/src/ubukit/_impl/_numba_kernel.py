"""Optional serial fused kernel, no fastmath and no architecture-specific build."""
import numpy as np
from numba import njit


@njit(cache=True, inline="always", fastmath=False)
def _distance_row_wide(X, C, dist, i):
    k, dimensions = C.shape
    c = 0
    while c + 3 < k:
        a = 0.0
        b = 0.0
        e = 0.0
        f = 0.0
        for d in range(dimensions):
            value = X[i, d]
            d0 = value - C[c, d]
            d1 = value - C[c + 1, d]
            d2 = value - C[c + 2, d]
            d3 = value - C[c + 3, d]
            a += d0 * d0
            b += d1 * d1
            e += d2 * d2
            f += d3 * d3
        dist[c] = a
        dist[c + 1] = b
        dist[c + 2] = e
        dist[c + 3] = f
        c += 4
    while c < k:
        value = 0.0
        for d in range(dimensions):
            delta = X[i, d] - C[c, d]
            value += delta * delta
        dist[c] = value
        c += 1
    nearest_sq = np.inf
    for c in range(k):
        total = dist[c]
        if not np.isfinite(total):
            raise FloatingPointError('squared distances overflowed; rescale the input')
        if total > 0.0 and total < np.finfo(np.float64).tiny:
            raise FloatingPointError('squared distances underflowed; rescale the input')
        if total == 0.0:
            for d in range(dimensions):
                if X[i, d] != C[c, d]:
                    raise FloatingPointError('squared distances underflowed; rescale the input')
        nearest_sq = min(nearest_sq, total)
    return nearest_sq


@njit(cache=True, fastmath=False)
def fused_step(X, C, alpha, beta, p, memberships, update_centers=True):
    n, dimensions = X.shape
    k = C.shape[0]
    packed = np.zeros((n, (k + 7) // 8), dtype=np.uint8)
    U = np.zeros((n, k), dtype=np.float64) if memberships else np.empty((0, 0), dtype=np.float64)
    sums = np.zeros((k, dimensions), dtype=np.float64) if update_centers else np.empty((0, 0), dtype=np.float64)
    mass = np.zeros(k, dtype=np.float64) if update_centers else np.empty(0, dtype=np.float64)
    dist = np.empty(k, dtype=np.float64)
    selected = np.empty(k, dtype=np.bool_)
    for i in range(n):
        if dimensions >= 16 and k >= 4:
            nearest_sq = _distance_row_wide(X, C, dist, i)
        else:
            nearest_sq = np.inf
            for c in range(k):
                total = 0.0
                for d in range(dimensions):
                    delta = X[i, d] - C[c, d]
                    total += delta * delta
                if not np.isfinite(total):
                    raise FloatingPointError('squared distances overflowed; rescale the input')
                if total > 0.0 and total < np.finfo(np.float64).tiny:
                    raise FloatingPointError('squared distances underflowed; rescale the input')
                if total == 0.0:
                    for d in range(dimensions):
                        if X[i, d] != C[c, d]:
                            raise FloatingPointError('squared distances underflowed; rescale the input')
                dist[c] = total
                nearest_sq = min(nearest_sq, total)
        a = alpha * np.sqrt(nearest_sq)
        aa, bb = alpha * alpha, beta * beta
        scale = max(a, beta)
        if beta == 0.0 and p != 2.0:
            for c in range(k):
                selected[c] = np.sqrt(dist[c]) <= a
        elif p == 1.0:
            threshold = a + beta
            for c in range(k):
                selected[c] = np.sqrt(dist[c]) <= threshold
        elif p == 2.0 and np.isfinite(aa) and (beta == 0.0 or bb > 0.0):
            threshold = aa * nearest_sq + bb
            for c in range(k):
                selected[c] = dist[c] <= threshold
        elif scale == 0.0:
            for c in range(k):
                selected[c] = dist[c] == 0.0
        elif np.isinf(scale):
            for c in range(k):
                selected[c] = True
        else:
            small = min(a, beta)
            small_ratio = small / scale
            rhs = 1.0 + small_ratio ** p
            use_log = p < 0.25 or p > 64.0 or (small_ratio < np.finfo(np.float64).tiny and small > 0.0)
            if small > 0.0:
                log_small = np.log(small_ratio) if small_ratio >= np.finfo(np.float64).tiny else np.log(small) - np.log(scale)
                log_right = np.log1p(np.exp(p * log_small))
            else:
                log_right = 0.0
            # Filter ordinary-p values by a per-point radius. The wide guard
            # triggers canonical recomputation; it never expands acceptance.
            radius_sq = 0.0
            if p >= 0.25 and p <= 64.0 and small_ratio >= np.finfo(np.float64).tiny:
                radius = scale * np.exp(np.log1p(small_ratio ** p) / p)
                radius_sq = radius * radius
            guard = 128.0 * np.finfo(np.float64).eps * (1.0 + 1.0 / p)
            for c in range(k):
                if radius_sq > 0.0 and np.isfinite(radius_sq) and abs(dist[c] - radius_sq) > guard * max(dist[c], radius_sq):
                    selected[c] = dist[c] <= radius_sq
                    continue
                dvalue = np.sqrt(dist[c])
                ratio = dvalue / scale
                if dvalue <= scale:
                    selected[c] = True
                elif small == 0.0:
                    selected[c] = False
                elif use_log or np.isinf(ratio):
                    relative = (dvalue - scale) / scale
                    log_left = np.log1p(relative) if np.isfinite(relative) else np.log(dvalue) - np.log(scale)
                    selected[c] = p * log_left <= log_right
                else:
                    selected[c] = ratio ** p <= rhs
        count = 0
        for c in range(k):
            if selected[c]:
                count += 1
        weight = 1.0 / count
        for c in range(k):
            if selected[c]:
                packed[i, c // 8] |= np.uint8(1 << (c % 8))
                if update_centers:
                    mass[c] += weight
                    for d in range(dimensions):
                        sums[c, d] += weight * (X[i, d] - X[0, d])
                if memberships:
                    U[i, c] = weight
    if not update_centers:
        return C, 0, packed, U
    out = C.copy()
    empty = 0
    for c in range(k):
        if mass[c] == 0.0:
            empty += 1
        else:
            for d in range(dimensions):
                out[c, d] = X[0, d] + sums[c, d] / mass[c]
    return out, empty, packed, U
