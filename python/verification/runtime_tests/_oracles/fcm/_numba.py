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


@njit(cache=True, inline="always")
def _point_membership(x, centers, previous, out, m, i):
    """Use the destination row as temporary squared distances.

    Objective reduction belongs to the robust log-domain evaluator in the
    caller; calculating it here would be discarded for both history and the
    final returned-center objective.
    """
    k, d = centers.shape
    minimum = np.inf
    zeros = 0
    for j in range(k):
        value = 0.0
        for feature in range(d):
            diff = x[i, feature] - centers[j, feature]
            value += diff * diff
        if value == 0.0:
            for feature in range(d):
                if x[i, feature] != centers[j, feature]:
                    for z in range(k):
                        out[i, z] = np.nan
                    return np.nan
        out[i, j] = value
        minimum = min(minimum, value)
        if value == 0.0:
            zeros += 1
    total = 0.0
    if zeros:
        for j in range(k):
            out[i, j] = 1.0 / zeros if out[i, j] == 0.0 else 0.0
    else:
        exponent = 1.0 / (m - 1.0)
        for j in range(k):
            distance = out[i, j]
            value = minimum / distance
            if m != 2.0:
                if m - 1.0 < 1e-4 and distance - minimum <= minimum * (1024.0 * (m - 1.0)):
                    value = np.exp(-np.log1p((distance - minimum) / minimum) * exponent)
                elif value < np.finfo(np.float64).tiny:
                    value = np.exp((np.log(minimum) - np.log(distance)) * exponent)
                else:
                    value = value ** exponent
            out[i, j] = value
            total += value
        for j in range(k):
            out[i, j] /= total
    delta2 = 0.0
    for j in range(k):
        diff = out[i, j] - previous[i, j]
        delta2 += diff * diff
    return delta2


@njit(cache=True, inline="always")
def _point_membership_wide(x, centers, previous, out, m, i):
    """Four independent center sums retain each distance's feature order."""
    k, d = centers.shape
    j = 0
    while j + 3 < k:
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for feature in range(d):
            value = x[i, feature]
            a = value - centers[j, feature]
            b = value - centers[j + 1, feature]
            c = value - centers[j + 2, feature]
            e = value - centers[j + 3, feature]
            s0 += a * a
            s1 += b * b
            s2 += c * c
            s3 += e * e
        out[i, j] = s0
        out[i, j + 1] = s1
        out[i, j + 2] = s2
        out[i, j + 3] = s3
        j += 4
    while j < k:
        value = 0.0
        for feature in range(d):
            diff = x[i, feature] - centers[j, feature]
            value += diff * diff
        out[i, j] = value
        j += 1
    minimum = np.inf
    zeros = 0
    for j in range(k):
        value = out[i, j]
        minimum = min(minimum, value)
        if value == 0.0:
            for feature in range(d):
                if x[i, feature] != centers[j, feature]:
                    for z in range(k):
                        out[i, z] = np.nan
                    return np.nan
            zeros += 1
    total = 0.0
    if zeros:
        for j in range(k):
            out[i, j] = 1.0 / zeros if out[i, j] == 0.0 else 0.0
    else:
        exponent = 1.0 / (m - 1.0)
        for j in range(k):
            distance = out[i, j]
            value = minimum / distance
            if m != 2.0:
                if m - 1.0 < 1e-4 and distance - minimum <= minimum * (1024.0 * (m - 1.0)):
                    value = np.exp(-np.log1p((distance - minimum) / minimum) * exponent)
                elif value < np.finfo(np.float64).tiny:
                    value = np.exp((np.log(minimum) - np.log(distance)) * exponent)
                else:
                    value = value ** exponent
            out[i, j] = value
            total += value
        for j in range(k):
            out[i, j] /= total
    delta2 = 0.0
    for j in range(k):
        diff = out[i, j] - previous[i, j]
        delta2 += diff * diff
    return delta2


@njit(cache=True)
def update_membership_serial(x, centers, previous, out, m):
    delta2 = 0.0
    if centers.shape[1] >= 16 and centers.shape[0] >= 4:
        for i in range(len(x)):
            delta2 += _point_membership_wide(x, centers, previous, out, m, i)
    else:
        for i in range(len(x)):
            delta2 += _point_membership(x, centers, previous, out, m, i)
    return delta2


@njit(cache=True, parallel=True)
def update_membership_parallel(x, centers, previous, out, m):
    delta2 = 0.0
    if centers.shape[1] >= 16 and centers.shape[0] >= 4:
        for i in prange(len(x)):
            delta2 += _point_membership_wide(x, centers, previous, out, m, i)
    else:
        for i in prange(len(x)):
            delta2 += _point_membership(x, centers, previous, out, m, i)
    return delta2
