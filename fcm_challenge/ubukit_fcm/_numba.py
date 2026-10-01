"""Optional fused direct-distance/membership kernels (no fastmath)."""
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads


def set_threads(threads):
    previous = get_num_threads()
    set_num_threads(threads)
    return previous


@njit(cache=True, inline="always")
def _point(x, centers, previous, out, m, i):
    k, d = centers.shape
    distances = np.empty(k, dtype=np.float64)
    minimum = np.inf
    zeros = 0
    for j in range(k):
        value = 0.0
        nonzero = False
        for feature in range(d):
            diff = x[i, feature] - centers[j, feature]
            value += diff * diff
            nonzero = nonzero or diff != 0.0
        if value == 0.0 and nonzero:
            for z in range(k):
                out[i, z] = np.nan
            return np.nan, np.nan
        distances[j] = value
        minimum = min(minimum, value)
        if value == 0.0:
            zeros += 1
    total = 0.0
    if zeros:
        for j in range(k):
            out[i, j] = 1.0 / zeros if distances[j] == 0.0 else 0.0
    else:
        exponent = 1.0 / (m - 1.0)
        for j in range(k):
            value = minimum / distances[j]
            if m != 2.0:
                if m - 1.0 < 1e-4 and distances[j] - minimum <= minimum:
                    value = np.exp(-np.log1p((distances[j] - minimum) / minimum) * exponent)
                elif value < np.finfo(np.float64).tiny:
                    value = np.exp((np.log(minimum) - np.log(distances[j])) * exponent)
                else:
                    value = value ** exponent
            out[i, j] = value
            total += value
        for j in range(k):
            out[i, j] /= total
    delta2 = 0.0
    objective = 0.0
    for j in range(k):
        diff = out[i, j] - previous[i, j]
        delta2 += diff * diff
        weight = out[i, j] * out[i, j] if m == 2.0 else out[i, j] ** m
        if weight < np.finfo(np.float64).tiny and out[i, j] > 0 and distances[j] > 0:
            objective += np.exp(m * np.log(out[i, j]) + np.log(distances[j]))
        else:
            objective += weight * distances[j]
    return delta2, objective


@njit(cache=True)
def update_serial(x, centers, previous, out, m):
    delta2 = 0.0
    objective = 0.0
    for i in range(len(x)):
        dd, jj = _point(x, centers, previous, out, m, i)
        delta2 += dd
        objective += jj
    return delta2, objective


@njit(cache=True, parallel=True)
def update_parallel(x, centers, previous, out, m):
    delta2 = 0.0
    objective = 0.0
    for i in prange(len(x)):
        dd, jj = _point(x, centers, previous, out, m, i)
        delta2 += dd
        objective += jj
    return delta2, objective
