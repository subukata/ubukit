"""Optional serial aggregation only. No fastmath and no import-time JIT."""
import numpy as np
from numba import njit


@njit(cache=True)
def aggregate(labels, Y, s, k):
    numerator = np.zeros((k, Y.shape[1]), dtype=np.float64)
    mass = np.zeros(k, dtype=np.float64)
    for i in range(len(labels)):
        c = labels[i]
        mass[c] += s[i]
        for d in range(Y.shape[1]):
            numerator[c, d] += Y[i, d]
    return numerator, mass
