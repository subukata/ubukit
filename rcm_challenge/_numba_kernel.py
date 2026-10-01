"""Optional serial fused kernel, no fastmath and no architecture-specific build."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def fused_step(X, C, alpha, beta, p, memberships):
    n, dimensions = X.shape
    k = C.shape[0]
    packed = np.zeros((n, (k + 7) // 8), dtype=np.uint8)
    U = np.zeros((n, k), dtype=np.float64) if memberships else np.empty((0, 0), dtype=np.float64)
    sums = np.zeros((k, dimensions), dtype=np.float64)
    mass = np.zeros(k, dtype=np.float64)
    dist = np.empty(k, dtype=np.float64)
    selected = np.empty(k, dtype=np.bool_)
    for i in range(n):
        nearest_sq = np.inf
        for c in range(k):
            total = 0.0
            nonzero = False
            for d in range(dimensions):
                delta = X[i, d] - C[c, d]
                nonzero = nonzero or delta != 0.0
                total += delta * delta
            if not np.isfinite(total):
                raise FloatingPointError('squared distances overflowed; rescale the input')
            if (total == 0.0 and nonzero) or (total > 0.0 and total < np.finfo(np.float64).tiny):
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
            for c in range(k):
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
                mass[c] += weight
                for d in range(dimensions):
                    sums[c, d] += weight * (X[i, d] - X[0, d])
                if memberships:
                    U[i, c] = weight
    out = C.copy()
    empty = 0
    for c in range(k):
        if mass[c] == 0.0:
            empty += 1
        else:
            for d in range(dimensions):
                out[c, d] = X[0, d] + sums[c, d] / mass[c]
    return out, empty, packed, U
