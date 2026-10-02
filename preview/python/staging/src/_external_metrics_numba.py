"""Optional Numba implementation; imported only on explicit backend selection."""
import math
from numba import njit


@njit(cache=True)
def emi_numba(n, pairs):
    result = 0.0
    result_correction = 0.0
    for p in range(pairs.shape[0]):
        a, b, multiplicity = pairs[p]
        lo, hi = max(0, a + b - n), min(a, b)
        # Floating division avoids an int64 intermediate product overflow.
        # Correct the mode using the recurrence if rounding lands next to it.
        mode = min(hi, max(lo, int((float(a) + 1) * (float(b) + 1) / (float(n) + 2))))
        while mode < hi and ((a - mode) / (mode + 1)) * ((b - mode) / (n - a - b + mode + 1)) > 1:
            mode += 1
        while mode > lo and (mode / (a - mode + 1)) * ((n - a - b + mode) / (b - mode + 1)) > 1:
            mode -= 1
        logscale = math.log(n) - math.log(a) - math.log(b)
        total = 1.0
        value = (mode / n) * (math.log(mode) + logscale) if mode else 0.0
        weight = 1.0
        for x in range(mode + 1, hi + 1):
            weight *= ((a - x + 1) / x) * ((b - x + 1) / (n - a - b + x))
            total += weight
            value += weight * (x / n) * (math.log(x) + logscale)
        weight = 1.0
        for x in range(mode - 1, lo - 1, -1):
            weight *= ((x + 1) / (a - x)) * ((n - a - b + x + 1) / (b - x))
            total += weight
            if x:
                value += weight * (x / n) * (math.log(x) + logscale)
        term = multiplicity * value / total
        # Compensated pair accumulation is useful with imbalanced marginals.
        z = term - result_correction
        t = result + z
        result_correction = (t - result) - z
        result = t
    return result
