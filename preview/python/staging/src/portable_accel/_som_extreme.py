"""Cold float64 SOM arithmetic for extreme finite magnitudes.

Nonnegative squared costs are mantissa/exponent pairs until after the
min-before-temperature softmax. Neither lambda nor gamma is clipped or
rescaled. Objective values and the stopping rule retain the caller's units.
This extends exponent range, not significand precision; it is not an exact
arithmetic or arbitrary-precision solver.
"""
import math
import numpy as np


def needs_extreme(*arrays, gamma=1.0, pca_scale=1.0):
    """Conservative cold-path trigger; ordinary kernels stay unchanged."""
    for value in (gamma, abs(pca_scale)):
        if value and (value < 1e-140 or value > 1e140):
            return True
    for array in arrays:
        lo, hi = float(np.min(array)), float(np.max(array))
        if max(abs(lo), abs(hi)) > 1e140:
            return True
        if lo > 0:
            if lo < 1e-140:return True
            continue
        if hi < 0:
            if -hi < 1e-140:return True
            continue
        absolute = np.abs(array)
        if np.any((absolute > 0) & (absolute < 1e-140)):
            return True
    return False


def _pair(value):
    return math.frexp(float(value))


def _multiply(a, b):
    mantissa, exponent = math.frexp(a[0] * b[0])
    return mantissa, exponent + a[1] + b[1]


def _sum(values):
    def terms():
        for mantissa, exponent in values:
            if mantissa:
                numerator, denominator = float(mantissa).as_integer_ratio()
                yield numerator, int(exponent) - (denominator.bit_length() - 1)
    return _sum_integers(terms())


def _sum_products(values, weights):
    def terms():
        for value, weight in zip(values, weights):
            if value and weight:
                a, ad = float(value).as_integer_ratio()
                b, bd = float(weight).as_integer_ratio()
                yield a * b, 2 - ad.bit_length() - bd.bit_length()
    return _sum_integers(terms())


def _sum_integers(terms):
    # Exact binary accumulation prevents small residuals from disappearing
    # before very large signed terms cancel. Round the final sum only once.
    total = 0; base = 0
    for numerator, power in terms:
        if not total:
            total, base = numerator, power
        elif power < base:
            total = (total << (base - power)) + numerator
            base = power
        else:
            total += numerator << (power - base)
    if not total:
        return 0.0, 0
    sign = -1 if total < 0 else 1
    magnitude = abs(total); bits = magnitude.bit_length()
    if bits > 53:
        shift = bits - 53; top = magnitude >> shift
        remainder = magnitude - (top << shift); half = 1 << (shift - 1)
        if remainder > half or (remainder == half and top & 1):
            top += 1
        mantissa = math.ldexp(float(top), -53)
    else:
        mantissa = math.ldexp(float(magnitude), -bits)
    normalized, extra = math.frexp(sign * mantissa)
    return normalized, base + bits + extra


def _divide(a, b):
    if not a[0]:
        return 0.0, 0
    mantissa, exponent = math.frexp(a[0] / b[0])
    return mantissa, exponent + a[1] - b[1]


def _float(pair, name):
    try:
        value = math.ldexp(*pair)
    except OverflowError:
        raise ValueError(f'SOM {name} is outside the float64 range; rescale inputs or parameters') from None
    if not math.isfinite(value):
        raise ValueError(f'SOM {name} is outside the float64 range; rescale inputs or parameters')
    return value


def _difference(a, b):
    delta = float(a) - float(b)
    if math.isfinite(delta):
        return _pair(delta)
    # The finite operands can have an overflowing difference.
    mantissa, exponent = math.frexp(float(a) * .5 - float(b) * .5)
    return mantissa, exponent + 1


def _weighted_mean(values, weights=None, normalize=True):
    if weights is None:
        numerator = _sum(_pair(value) for value in values)
        denominator = _pair(len(values))
    else:
        numerator = _sum_products(values, weights)
        denominator = _sum(_pair(weight) for weight in weights)
    if normalize:
        numerator = _divide(numerator, denominator)
    return _float(numerator, 'weighted mean')


def _cost(x, w, v=None, r=None, gamma=0.0):
    def terms():
        for a, b in zip(x, w):
            delta = _difference(a, b)
            yield _multiply(delta, delta)
        if gamma:
            g = _pair(gamma)
            for a, b in zip(v, r):
                delta = _difference(a, b)
                yield _multiply(_multiply(delta, delta), g)
    return _sum(terms())


def _key(mantissa, exponent):
    return (exponent, mantissa) if mantissa else (-100000, 0.0)


def _probability_row(mantissas, exponents, lam, out):
    minimum_index = min(range(len(out)), key=lambda j: _key(mantissas[j], exponents[j]))
    minimum = float(mantissas[minimum_index]), int(exponents[minimum_index])
    temperature = _pair(lam)
    for j in range(len(out)):
        difference = _sum(((float(mantissas[j]), int(exponents[j])), (-minimum[0], minimum[1])))
        ratio = _divide(difference, temperature)
        # exp(-x) already rounds to zero far below this overflow threshold.
        value = math.inf if ratio[0] and ratio[1] > 1024 else math.ldexp(*ratio)
        out[j] = math.exp(-value)
    unit_count = int(np.count_nonzero(out == 1.0))
    tail = math.fsum(float(value) for value in out if value != 1.0)
    total = unit_count + tail
    out /= total
    # Preserve a small nonminimal tail even when 1 + tail rounds to one.
    # A huge lambda can make its free-energy contribution representable.
    log_total = math.log(unit_count) + math.log1p(tail / unit_count)
    entropy_offset = _multiply(temperature, _pair(-log_total))
    return _sum((minimum, entropy_offset))


def probabilities(X, W, lam):
    n, m = len(X), len(W)
    P = np.empty((n, m)); mantissas = np.empty(m); exponents = np.empty(m, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            mantissas[j], exponents[j] = _cost(X[i], W[j])
        _probability_row(mantissas, exponents, lam, P[i])
    return P


def pca_factors(X):
    """A full SVD of power-of-two scaled X, with its explicit unit exponent."""
    largest = max(abs(float(np.min(X))), abs(float(np.max(X))))
    exponent = math.frexp(largest)[1] if largest else 0
    scaled = np.ldexp(X, -exponent)
    mean = scaled.mean(axis=0, keepdims=True)
    centered = scaled - mean
    U, singular, basis = np.linalg.svd(centered, full_matrices=False)
    return mean, U, singular, basis, exponent


def prototypes_from_factors(mean, singular, basis, exponent, n, R, pca_scale):
    m, q = R.shape; d = mean.shape[1]; k = min(q, d)
    normalized = np.empty((m, k))
    for h in range(k):
        origin = _weighted_mean(R[:, h])
        differences = [_difference(value, origin) for value in R[:, h]]
        maximum = max(((abs(a), b) for a, b in differences), key=lambda pair: _key(*pair))
        if _key(*maximum) < _key(*_pair(1e-12)):
            maximum = _pair(1e-12)
        for j, value in enumerate(differences):
            normalized[j, h] = _float(_divide(value, maximum), 'normalized grid')
    W = np.empty((m, d))
    factors = [_multiply(_pair(pca_scale), _pair(singular[h] / math.sqrt(n))) for h in range(k)]
    for j in range(m):
        for f in range(d):
            terms = [_pair(mean[0, f])]
            terms += [_multiply(_multiply(factors[h], _pair(normalized[j, h])), _pair(basis[h, f]))
                      for h in range(k)]
            mantissa, power = _sum(terms)
            W[j, f] = _float((mantissa, power + exponent), 'PCA prototype')
    return W


def initialize(X, R, lam, pca_scale):
    mean, _, singular, basis, exponent = pca_factors(X)
    W = prototypes_from_factors(mean, singular, basis, exponent, len(X), R, pca_scale)
    return W, probabilities(X, W, lam)


def run(X, R, W0, P0, gamma, lam, max_iters, tol, *, trace=False):
    """Slow exceptional fallback, preserving old-P -> V/W -> new-P order."""
    W, P = W0.copy(), P0.copy(); V = None; history = []; trajectory = []
    n, d = X.shape; m, q = R.shape
    # Public backend validation runs first. Invalid probabilities must not
    # enter the nonnegative-cost and convex-mean arithmetic below.
    if np.any(P < 0) or not np.allclose(P.sum(axis=1), 1.0, rtol=0., atol=1e-8):
        raise ValueError('P0 must be nonnegative and row-stochastic within 1e-8')
    mantissas = np.empty(m); exponents = np.empty(m, dtype=np.int64)
    for _ in range(max_iters):
        V = np.empty((n, q))
        for i in range(n):
            for h in range(q):
                V[i, h] = _weighted_mean(R[:, h], P[i], normalize=False)
        for j in range(m):
            weights = P[:, j]
            if np.any(weights > 0):
                for f in range(d):
                    W[j, f] = _weighted_mean(X[:, f], weights)
        row_objectives = []
        for i in range(n):
            for j in range(m):
                mantissas[j], exponents[j] = _cost(X[i], W[j], V[i], R[j], gamma)
            row_objectives.append(_probability_row(mantissas, exponents, lam, P[i]))
        objective = _float(_sum(row_objectives), 'objective')
        history.append(objective)
        if trace:
            trajectory.append(dict(W=W.copy(), P=P.copy(), V=V.copy(), objective=objective))
        if len(history) > 1:
            previous = history[-2]; difference = abs(objective - previous)
            relative = (difference / max(1., abs(previous)) if math.isfinite(difference)
                        else abs(objective / max(1., abs(previous)) - previous / max(1., abs(previous))))
            if relative <= tol:
                break
    result = dict(W=W, P=P, V=V, history=np.asarray(history), n_iter=len(history),
                  variant='extreme_float64_exponent_fallback', direct_fallback_rows=n * len(history),
                  numerical_contract='float64 significands; extended cost exponents; original-unit objective/stopping')
    if trace:
        result['trajectory'] = trajectory
    return result
