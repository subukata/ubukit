"""Guarded finite-m FCM arithmetic for exceptional float64 ranges.

This module is not an m->infinity approximation. The distance update retains
relative log weights independently of rounded public memberships. It is used
only when the ordinary kernel's exponent/range guards fail.
"""
import math
import numpy as np

_TINY = np.finfo(np.float64).tiny
_MAX = np.finfo(np.float64).max
_LN2 = math.log(2.0)


def _logmeanexp_nonpositive(a):
    """Accurate even when exp(a) would round to one in every column."""
    near = np.max(-a, axis=1) <= 0.5
    out = np.empty(len(a))
    if np.any(near):
        out[near] = np.log1p(np.mean(np.expm1(a[near]), axis=1))
    if np.any(~near):
        out[~near] = np.log(np.mean(np.exp(a[~near]), axis=1))
    return out


def _product_expansion(a, b):
    product = a * b
    split = 134217729.0 * a
    ah = split - (split - a)
    al = a - ah
    split = 134217729.0 * b
    bh = split - (split - b)
    bl = b - bh
    error = ((ah * bh - product) + ah * bl + al * bh) + al * bl
    return product, error


def _initial_logweights(raw, m):
    # Represent each row sum by a two-part mantissa and an integer exponent.
    # This also retains near-tie differences in tiny supplied memberships;
    # log(raw) first would lose them before a large m amplifies them.
    raw_mant, raw_exp = np.frexp(raw)
    n, k = raw.shape
    sum_hi, sum_lo, sum_exp = np.empty(n), np.empty(n), np.empty(n, dtype=np.int64)
    for row in range(n):
        exponent = int(np.max(raw_exp[row][raw[row] > 0]))
        scaled = np.ldexp(raw[row], -exponent)
        hi = math.fsum(scaled)
        lo = math.fsum([*scaled, -hi])
        mantissa, extra = math.frexp(hi)
        sum_hi[row], sum_lo[row] = mantissa, math.ldexp(lo, -extra)
        sum_exp[row] = exponent + extra
    result = np.full_like(raw, -np.inf)
    for col in range(k):
        rows = np.flatnonzero(raw[:, col] > 0)
        if not len(rows):
            continue
        # Any positive reference is valid. This approximate score merely
        # chooses one near the maximum to keep most subsequent ratios small.
        score = (raw_exp[rows, col] - sum_exp[rows]) * _LN2 + np.log(raw_mant[rows, col]) - np.log(sum_hi[rows])
        ref = int(rows[np.argmax(score)])
        relative = np.empty(len(rows))
        for position, row in enumerate(rows):
            shift = int((raw_exp[row, col] - sum_exp[row]) - (raw_exp[ref, col] - sum_exp[ref]))
            nh, nl = _product_expansion(float(raw_mant[row, col]), float(sum_hi[ref]))
            nl += raw_mant[row, col] * sum_lo[ref]
            dh, dl = _product_expansion(float(raw_mant[ref, col]), float(sum_hi[row]))
            dl += raw_mant[ref, col] * sum_lo[row]
            if abs(shift) <= 2:
                difference = math.fsum([math.ldexp(nh, shift), math.ldexp(nl, shift), -dh, -dl])
                # The small denominator low part matters at most at second
                # order in a near tie. Include it for non-near ratios.
                relative[position] = math.log1p(difference / (dh + dl))
            else:
                relative[position] = shift * _LN2 + math.log(nh) - math.log(dh)
        with np.errstate(over='ignore'):
            result[rows, col] = m * (relative - np.max(relative))
    return result


def _scaled_squared_pair(x, center):
    """Return mantissa, exponent for q = mantissa * 2**exponent.

    The exponent is an integer, so q need never be representable in float64.
    Exact coordinate equality, and only equality, represents zero distance.
    """
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        difference = x - center
    extra = 0
    if not np.isfinite(difference).all():
        magnitude = max(float(np.max(np.abs(x))), float(np.max(np.abs(center))))
        _, extra = math.frexp(magnitude)
        difference = np.ldexp(x, -extra) - np.ldexp(center, -extra)
    scale = float(np.max(np.abs(difference)))
    if scale == 0:
        return 0.0, 0
    _, exponent = math.frexp(scale)
    scaled = np.ldexp(difference, -exponent)
    norm = math.fsum(float(v) * float(v) for v in scaled)
    mantissa, norm_exponent = math.frexp(norm)
    return mantissa, 2 * (exponent + extra) + norm_exponent


def _squared_pair_expansion(x, center):
    """Two-part scaled square sum for near-one fuzzifier distance ratios.

    TwoSum preserves coordinate-subtraction residuals; Dekker products retain
    the low part of each square. Only the exceptional near-one path uses this
    extra work, because squaring and rounding first can erase a near-tie.
    """
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        high = x - center
    extra = 0
    if not np.isfinite(high).all():
        _, extra = math.frexp(max(float(np.max(np.abs(x))), float(np.max(np.abs(center)))))
        x, center = np.ldexp(x, -extra), np.ldexp(center, -extra)
        high = x - center
    virtual = high - x
    low = (x - (high - virtual)) + (-center - virtual)
    magnitude = float(np.max(np.abs(high)))
    if magnitude == 0:
        return 0.0, 0.0, 0
    _, exponent = math.frexp(magnitude)
    high, low = np.ldexp(high, -exponent), np.ldexp(low, -exponent)
    products, errors = [], []
    for a, residual in zip(high, low):
        a, residual = float(a), float(residual)
        product = a * a
        split = 134217729.0 * a
        ah = split - (split - a)
        al = a - ah
        error = ((ah * ah - product) + 2.0 * ah * al) + al * al
        products.append(product)
        errors.extend((error, 2.0 * a * residual, residual * residual))
    qhi = math.fsum(products)
    qlo = math.fsum(products + [-qhi] + errors)
    mantissa, qe = math.frexp(qhi)
    lomantissa = math.ldexp(qlo, -qe)
    exponent = 2 * (exponent + extra) + qe
    if mantissa == 0.5 and lomantissa < 0:
        mantissa *= 2.0
        lomantissa *= 2.0
        exponent -= 1
    return mantissa, lomantissa, exponent


def _near_one_ratios(x, centers, ratio, zero_count):
    for row in np.flatnonzero(zero_count == 0):
        parts = [_squared_pair_expansion(x[row], center) for center in centers]
        nearest = min(range(len(parts)), key=lambda j: (parts[j][2], parts[j][0], parts[j][1]))
        mh, ml, me = parts[nearest]
        for col, (hi, lo, exponent) in enumerate(parts):
            shift = exponent - me
            if shift <= 1:
                difference = math.fsum([math.ldexp(hi, shift), math.ldexp(lo, shift), -mh, -ml])
                ratio[row, col] = math.log1p(difference / mh)
    return ratio


def distance_parts(x, centers, m=None):
    """Squared-distance mantissas/exponents, accurate log ratios, and logs."""
    from scipy.spatial.distance import cdist
    q = cdist(x, centers, 'sqeuclidean')
    mantissa, exponent = np.frexp(q)
    unsafe = (~np.isfinite(q)) | ((q > 0) & (q < _TINY))
    rr, cc = np.nonzero(q == 0)
    for row, col in zip(rr, cc):
        if np.any(x[row] != centers[col]):
            unsafe[row, col] = True
    rr, cc = np.nonzero(unsafe)
    for row, col in zip(rr, cc):
        mantissa[row, col], exponent[row, col] = _scaled_squared_pair(x[row], centers[col])
    zeros = mantissa == 0
    zero_count = np.sum(zeros, axis=1)
    with np.errstate(divide='ignore'):
        logq = np.log(mantissa) + exponent * _LN2
    positive = zero_count == 0
    ratio = np.full_like(q, np.inf)
    if np.any(positive):
        mm, ee = mantissa[positive], exponent[positive]
        emin = ee.min(axis=1)
        mmin = np.where(ee == emin[:, None], mm, np.inf).min(axis=1)
        edelta = ee - emin[:, None]
        r = edelta * _LN2 + np.log(mm) - np.log(mmin[:, None])
        near = edelta <= 1
        rr, cc = np.nonzero(near)
        larger = np.ldexp(mm[rr, cc], edelta[rr, cc])
        r[rr, cc] = np.log1p((larger - mmin[rr]) / mmin[rr])
        ratio[positive] = r
    ratio[zeros] = 0.0
    if m is not None and m - 1.0 < 1e-4:
        ratio = _near_one_ratios(x, centers, ratio, zero_count)
    return ratio, logq, zeros, zero_count, int(np.count_nonzero(unsafe))


def memberships_and_logweights(ratio, zeros, zero_count, m):
    n, k = ratio.shape
    positive = zero_count == 0
    u = np.zeros_like(ratio)
    residual = np.full_like(ratio, -np.inf)
    if np.any(positive):
        r = ratio[positive]
        a = -r / (m - 1.0)
        e = np.exp(a)
        u[positive] = e / e.sum(axis=1, keepdims=True)
        # Common -m*log(K) is removed ANALYTICALLY before multiplying.
        # expm1/log1p retains O(1/m) structure even at m=1e308.
        residual[positive] = -(m / (m - 1.0)) * r - m * _logmeanexp_nonpositive(a)[:, None]
    if np.any(~positive):
        u[~positive] = zeros[~positive] / zero_count[~positive, None]
    # Exact-zero rows use a distinct base, -m*log(number of zero centers).
    # Choose each column's smallest base count before forming differences;
    # no huge positive common term or inf-inf subtraction is introduced.
    counts = np.where(zeros, zero_count[:, None], k)
    base = counts.min(axis=0)
    logweights = np.full_like(ratio, -np.inf)
    with np.errstate(over='ignore', invalid='ignore'):
        if np.any(positive):
            offset = m * np.log1p((base.astype(float) - k) / k)
            logweights[positive] = residual[positive] + offset
        for col in range(k):
            rows = zeros[:, col]
            if np.any(rows):
                logweights[rows, col] = m * np.log(base[col] / zero_count[rows])
        maximum = logweights.max(axis=0)
        valid = np.isfinite(maximum)
        logweights[:, valid] -= maximum[valid]
    return u, logweights


def _weighted_product(logweight, x, divisor):
    """Recover products whose weight alone is below the subnormal range."""
    # Below -3000, multiplication by even the largest finite coordinate is
    # far below the smallest subnormal (log(max_float) is only about 710).
    # This is a guaranteed representability cutoff, not weight clipping.
    if x == 0.0 or not math.isfinite(logweight) or logweight < -3000.0:
        return 0.0
    e = math.floor(logweight / _LN2)
    remainder = logweight - e * _LN2
    mantissa, exponent = math.frexp(x)
    return math.ldexp(mantissa * math.exp(remainder) / divisor, exponent + e)


def _balanced_fsum(values):
    """Avoid fsum's intermediate overflow before opposite signs cancel."""
    positive = [float(v) for v in values if v > 0]
    negative = [float(v) for v in values if v < 0]
    ordered = []
    balance = 0.0
    while positive and negative:
        value = negative.pop() if balance > 0 else positive.pop()
        ordered.append(value)
        balance += value
    ordered.extend(positive)
    ordered.extend(negative)
    return math.fsum(ordered)


def weighted_centers(x, logweights, previous):
    """Cluster-normalized weights, compensated sums, guarded product recovery."""
    centers = previous.copy()
    with np.errstate(under='ignore'):
        weights = np.exp(logweights)
    for col in range(logweights.shape[1]):
        total = math.fsum(weights[:, col])
        if not total:
            continue
        w = weights[:, col]
        for feature in range(x.shape[1]):
            coords = x[:, feature]
            with np.errstate(under='ignore'):
                products = w * coords
            weak = (coords != 0) & np.isfinite(logweights[:, col]) & (np.abs(products) < _TINY)
            # Sum before dividing whenever possible: preserves subnormal sums
            # and exact positive/negative cancellation across large ranges.
            for row in np.flatnonzero(weak):
                products[row] = _weighted_product(float(logweights[row, col]), float(coords[row]), 1.0)
            if np.max(np.abs(products)) < _TINY and np.any(weak):
                # Accumulate below-range products at a shared binary scale.
                # Individual half-subnormal products can sum to a valid mean.
                relevant = (coords != 0) & np.isfinite(logweights[:, col])
                peak = float(np.max(logweights[relevant, col] + np.log(np.abs(coords[relevant]))))
                if peak < -3000.0:
                    center = 0.0
                else:
                    scale_exp = math.floor(peak / _LN2)
                    scaled = []
                    for row in np.flatnonzero(relevant):
                        lw = float(logweights[row, col])
                        if lw < -6000.0:
                            continue
                        we = math.floor(lw / _LN2)
                        mant, xe = math.frexp(float(coords[row]))
                        scaled.append(math.ldexp(mant * math.exp(lw - we * _LN2), xe + we - scale_exp))
                    center = math.ldexp(math.fsum(scaled) / total, scale_exp)
                centers[col, feature] = center
                continue
            try:
                center = math.fsum(products) / total
            except OverflowError:
                try:
                    center = _balanced_fsum(products) / total
                except OverflowError:
                    # The numerator itself is outside range. A power-of-two
                    # scale is exact for normal products; dividing weights by
                    # their non-binary total before cancellation is not.
                    exponent = math.ceil(math.log2(total))
                    scaled = [math.ldexp(float(v), -exponent) for v in products]
                    center = _balanced_fsum(scaled) / math.ldexp(total, -exponent)
            centers[col, feature] = center
    return centers


def objective_from_pair(x, centers, u, m):
    """The published rounded U/center pair, with explicit range diagnostics."""
    from scipy.special import logsumexp
    _, logq, _, _, recovered = distance_parts(x, centers)
    positive = (u > 0) & np.isfinite(logq)
    if not np.any(positive):
        return 0.0, -np.inf, 'exact_zero', recovered
    with np.errstate(over='ignore', under='ignore'):
        terms = m * np.log(u[positive]) + logq[positive]
        logobjective = float(logsumexp(terms))
        objective = float(np.exp(logobjective))
    status = 'overflow' if np.isinf(objective) else 'underflow' if objective == 0 else 'finite'
    return objective, logobjective, status, recovered


def fit_robust(original_x, initial_u, raw_initial, m, max_iter, tol, backend,
               return_history, reason):
    """Same lagged-center and membership-only stopping contract as core.fit."""
    n, k = initial_u.shape
    # Translation helps ordinary offsets; do not overflow the transformation.
    origin = original_x[0].copy()
    with np.errstate(over='ignore', invalid='ignore'):
        translated = original_x - origin
    if np.isfinite(translated).all() and np.array_equal(translated + origin, original_x):
        x = translated
    else:
        x = original_x
        origin = np.zeros(x.shape[1])
    raw = initial_u if raw_initial is None else np.asarray(raw_initial, dtype=np.float64)
    logweights = _initial_logweights(raw, m)
    u = initial_u.copy()
    mean_weights = np.zeros((n, 1))
    mean = weighted_centers(x, mean_weights, np.zeros((1, x.shape[1])))
    centers = np.repeat(mean, k, axis=0)
    history = []
    loghistory = []
    recovered_total = 0
    # Dimensionless center displacement relative to max absolute translated X.
    magnitude = float(np.max(np.abs(x)))
    for iteration in range(1, int(max_iter) + 1):
        newcenters = weighted_centers(x, logweights, centers)
        if magnitude:
            with np.errstate(over='ignore', invalid='ignore'):
                displacement = newcenters - centers
            finite = np.isfinite(displacement)
            displacement[finite] /= magnitude
            displacement[~finite] = (newcenters[~finite] / magnitude - centers[~finite] / magnitude)
            center_motion = float(np.max(np.abs(displacement)))
        else:
            center_motion = 0.0
        centers = newcenters
        ratio, _, zeros, count, recovered = distance_parts(x, centers, m)
        recovered_total += recovered
        unew, logweights = memberships_and_logweights(ratio, zeros, count, m)
        difference = unew - u
        delta_scale = float(np.max(np.abs(difference)))
        delta = delta_scale * float(np.linalg.norm(difference / delta_scale)) if delta_scale else 0.0
        u = unew
        if return_history:
            value, logvalue, _, recovered = objective_from_pair(x, centers, u, m)
            history.append(value)
            loghistory.append(logvalue)
            recovered_total += recovered
        if delta < tol:
            break
    working_centers = centers
    centers = working_centers + origin
    # TwoSum detects rounding in the restoration itself, including a small
    # origin lost beside a large working center (a reverse subtraction alone
    # would not detect that case).
    virtual_origin = centers - working_centers
    restoration_error = ((working_centers - (centers - virtual_origin))
                         + (origin - virtual_origin))
    restoration_rounded = bool(np.any(restoration_error != 0.0))
    objective, logobjective, objective_status, recovered = objective_from_pair(original_x, centers, u, m)
    recovered_total += recovered
    result = dict(centers=centers, membership=u, labels=u.argmax(axis=1),
                  objective=objective, fpc=float(np.einsum('ij,ij->', u, u) / n),
                  n_iter=iteration, converged=bool(delta < tol), delta=delta,
                  backend=backend)
    result['numerical_diagnostics'] = dict(
        arithmetic='scaled_log', trigger=reason, requested_backend=backend,
        effective_backend='scipy_scaled_log', recovered_distance_pairs=recovered_total,
        membership_frame='working_coordinates', published_centers_rounded=restoration_rounded,
        center_relative_delta=center_motion,
        membership_convergence_only=bool(delta < tol and center_motion >= tol),
        log_objective=logobjective, objective_status=objective_status,
        membership_resolution_lost_rows=int(np.count_nonzero(
            (count == 0) & (np.ptp(u, axis=1) == 0) & (np.ptp(ratio, axis=1) > 0))))
    if return_history:
        result['objective_history'] = np.asarray(history)
        result['numerical_diagnostics']['log_objective_history'] = np.asarray(loghistory)
    return result
