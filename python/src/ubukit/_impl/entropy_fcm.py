"""Entropy-regularized fuzzy c-means with linear membership weights.

Only scalar range-preserving arithmetic is reused from the SOM implementation;
the squared-distance objective and alternating EFCM updates are independent.
"""
import math

import numpy as np

from .fcm.core import _validate
from .portable_accel import _som_extreme as _wide


def _temperature(value):
    if isinstance(value, (bool, np.bool_, str, bytes)) or np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError("tau must be a finite positive real scalar")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("tau must be a finite positive real scalar") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("tau must be finite and positive after float64 conversion")
    return value


def _reference_centers(x, u, old):
    centers = old.copy()
    for c in range(u.shape[1]):
        weights = u[:, c]
        if np.any(weights > 0):
            for f in range(x.shape[1]):
                centers[c, f] = _wide._weighted_mean(x[:, f], weights)
    return centers


def _reference_membership(x, centers, tau):
    # Keep every binary cost bit until AFTER minimum subtraction. Rounding
    # costs such as 1e16 and 1e16+1 first would erase a meaningful tau=1 gap.
    out = np.empty((len(x), len(centers)))
    temperature = _dyadic(tau)
    for i, row in enumerate(x):
        costs = [_exact_cost(row, center) for center in centers]
        minimum = costs[0]
        for cost in costs[1:]:
            if _subtract(cost, minimum)[0] < 0:
                minimum = cost
        weights = [math.exp(-_ratio(_subtract(cost, minimum), temperature)) for cost in costs]
        total = math.fsum(weights)
        for c, weight in enumerate(weights):
            out[i, c] = weight / total
    return out


def _dyadic(value):
    numerator, denominator = float(value).as_integer_ratio()
    return numerator, 1-denominator.bit_length()


def _add(a, b):
    if not a[0]: return b
    if not b[0]: return a
    power = min(a[1], b[1])
    return (a[0] << (a[1]-power)) + (b[0] << (b[1]-power)), power


def _subtract(a, b):
    return _add(a, (-b[0], b[1]))


def _multiply(a, b):
    return a[0]*b[0], a[1]+b[1]


def _sum_exact(terms):
    total = (0, 0)
    for term in terms:
        total = _add(total, term)
    return total


def _ratio(a, b):
    shift = a[1]-b[1]
    try:
        return (a[0] << shift)/b[0] if shift >= 0 else a[0]/(b[0] << -shift)
    except OverflowError:
        return math.inf if a[0]*b[0] >= 0 else -math.inf


def _exact_cost(row, center):
    return _sum_exact(_multiply(delta, delta) for delta in
                      (_subtract(_dyadic(a), _dyadic(b)) for a, b in zip(row, center)))


def _objective_pair(x, centers, u, tau):
    temperature = _dyadic(tau)
    def terms():
        for row, memberships in zip(x, u):
            for center, member in zip(centers, memberships):
                if member > 0:
                    weight = _dyadic(member)
                    yield _multiply(weight, _exact_cost(row, center))
                    entropy = _multiply(weight, _dyadic(math.log(member)))
                    yield _multiply(temperature, entropy)
    return _sum_exact(terms())


def _objective_value(pair):
    value = _ratio(pair, (1, 0))
    status = "overflow" if not math.isfinite(value) else "underflow" if value == 0 and pair[0] else "finite"
    log_abs = math.log(abs(pair[0])) + pair[1] * math.log(2) if pair[0] else -math.inf
    return value, status, log_abs


def _ordinary_range(x, tau):
    absolute = np.abs(x)
    return (1e-140 <= tau <= 1e140 and np.max(absolute) <= 1e140
            and not np.any((absolute > 0) & (absolute < 1e-140)))


def _numpy_centers(x, u, old):
    # Column scaling preserves the mean even when an entire cluster is tiny.
    maxima = u.max(axis=0)
    active = maxima > 0
    weights = u[:, active] / maxima[active]
    sums = weights.sum(axis=0)
    numerator = weights.T @ x
    absolute_sum = weights.T @ np.abs(x)
    centers = old.copy()
    fallback = False
    centers[active] = numerator / sums[:, None]
    # Correct strong cancellation with the exact binary accumulator already
    # used by the cold path. Constant features retain their exact value.
    for row, c in enumerate(np.flatnonzero(active)):
        for f in np.flatnonzero(np.abs(numerator[row]) <= 32*np.finfo(float).eps*absolute_sum[row]):
            if absolute_sum[row, f] > 0:
                centers[c, f] = _wide._weighted_mean(x[:, f], u[:, c])
                fallback = True
    np.clip(centers, x.min(axis=0), x.max(axis=0), out=centers)
    return centers, fallback


def _numpy_memberships_objective(x, centers, tau):
    n, d = x.shape
    k = len(centers)
    u = np.empty((n, k))
    rows = max(1, min(256, (1 << 18)//(k*d)))
    objective_terms = []
    fallback = False
    for start in range(0, n, rows):
        block = x[start:start+rows]
        difference = block[:, None, :] - centers[None, :, :]
        absolute = np.abs(difference)
        costs = np.einsum('nkd,nkd->nk', difference, difference)
        # Require the conservative cost-rounding bound to be small relative
        # to temperature. Absolute distances alone cannot justify softmax.
        sensitive_gap = np.any(8*d*np.spacing(costs.max(axis=1)) > tau*1e-12)
        if sensitive_gap or np.any((absolute > 0) & (absolute < 1e-140)):
            membership = _reference_membership(block, centers, tau)
            pair = _objective_pair(block, centers, membership, tau)
            fallback = True
        else:
            with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='raise'):
                membership = np.exp(-(costs-costs.min(axis=1, keepdims=True))/tau)
                membership /= membership.sum(axis=1, keepdims=True)
                positive = membership > 0
                distortion = float(np.sum(membership*costs))
                entropy = tau*float(np.sum(membership[positive]*np.log(membership[positive])))
            objective = math.fsum((distortion, entropy))
            if abs(objective) <= 32*np.finfo(float).eps*(distortion+abs(entropy)):
                pair = _objective_pair(block, centers, membership, tau)
                fallback = True
            else:
                pair = _dyadic(objective)
        u[start:start+len(block)] = membership
        objective_terms.append(pair)
    total = _sum_exact(objective_terms)
    magnitude = math.fsum(abs(_ratio(term, (1, 0))) for term in objective_terms)
    if abs(_ratio(total, (1, 0))) <= 64*np.finfo(float).eps*magnitude:
        total = _objective_pair(x, centers, u, tau)
        fallback = True
    return u, total, fallback


def fit_entropy_fcm(X, n_clusters=None, *, init=None, tau=1.0,
                    max_iter=300, tol=1e-5, random_state=None,
                    backend="numpy", return_history=False):
    """Fit entropy-regularized fuzzy c-means on finite real samples.

    Minimize sum(u_ik * ||x_i-v_k||**2) + tau * sum(u_ik * log(u_ik)),
    with u >= 0, row sums one, tau > 0, and 0*log(0) = 0. Centers use u,
    not u**m. Each iteration computes centers from old memberships, then
    memberships with a stable exp(-distance**2/tau) normalization.

    Parameters follow fit_fcm: X is (N,D); init is an optional (N,K)
    nonnegative membership matrix with positive row sums, normalized on a copy;
    n_clusters can be inferred from init. random_state uses NumPy's generator.
    tol tests the absolute Frobenius membership change; tol=0 runs max_iter.
    backend is 'numpy' or 'reference'; both preserve the same objective.

    Return a dict with owned centers (K,D), membership (N,K), labels (N,),
    objective, fpc, n_iter, converged, delta, tau, backend, and numerical_diagnostics;
    objective_history is included when return_history=True. Empty clusters retain
    their prior center (initially the data mean). The final objective is evaluated
    at the returned pair. A membership-only stopping test does not certify a
    stationary center update. Unrepresentable signed objectives are reported as
    signed infinity or zero with status and log-absolute-value diagnostics.
    """
    tau = _temperature(tau)
    if backend not in {"numpy", "reference"}:
        raise ValueError("entropy FCM backend must be 'numpy' or 'reference'")
    x, u = _validate(X, init, n_clusters, 2.0, max_iter, tol, 1, random_state)
    x = x.copy()
    centers = np.tile([_wide._weighted_mean(x[:, f]) for f in range(x.shape[1])], (u.shape[1], 1))
    history = []
    ordinary = backend == "numpy" and _ordinary_range(x, tau)
    fallback = not ordinary
    converged = False
    for iteration in range(1, int(max_iter) + 1):
        if ordinary:
            centers, center_fallback = _numpy_centers(x, u, centers)
            next_u, pair, used_fallback = _numpy_memberships_objective(x, centers, tau)
            fallback |= used_fallback or center_fallback
        else:
            centers = _reference_centers(x, u, centers)
            next_u = _reference_membership(x, centers, tau)
            pair = _objective_pair(x, centers, next_u, tau)
        delta = 0.0
        for old, new in zip(u.flat, next_u.flat):
            delta = math.hypot(delta, float(new) - float(old))
        u = next_u
        objective, status, log_abs = _objective_value(pair)
        if return_history:
            history.append(objective)
        converged = delta < tol
        if converged:
            break
    result = dict(centers=centers, membership=u, labels=u.argmax(axis=1),
                  objective=objective, fpc=float(np.sum(u * u) / len(u)),
                  n_iter=iteration, converged=converged, delta=delta, tau=tau,
                  backend=backend, numerical_diagnostics={
                      "arithmetic": "numpy-with-scaled-fallback" if ordinary else "scaled-exponent-reference",
                      "used_reference_fallback": fallback,
                      "objective_status": status,
                      "log_abs_objective": log_abs,
                      "objective_sign": (1 if pair[0] > 0 else -1) if pair[0] else 0,
                      "stopping_rule": "membership-frobenius",
                  })
    if return_history:
        result["objective_history"] = np.array(history, dtype=np.float64)
    return result
