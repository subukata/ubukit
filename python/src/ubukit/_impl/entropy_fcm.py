"""Entropy-regularized fuzzy c-means using ordinary float64 arithmetic only."""
import math

import numpy as np

from .fcm.core import _validate


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


def _finite(value, operation):
    if not np.isfinite(value).all():
        raise ValueError(f"entropy FCM float64 {operation} overflowed; rescale X and tau")
    return value


def _numpy_centers(x, u, old, feature_bounds):
    # Scale tiny cluster columns before normalization. All arithmetic remains
    # float64; cancellation residuals and underflowed terms are not repaired.
    maxima = u.max(axis=0)
    active = maxima > 0
    weights = u[:, active] / maxima[active]
    weights /= weights.sum(axis=0)
    centers = old.copy()
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        centers[active] = weights.T @ x
    _finite(centers, "center update")
    np.clip(centers, feature_bounds[0], feature_bounds[1], out=centers)
    # A constant feature has a known convex mean, independent of summation.
    constant = feature_bounds[0] == feature_bounds[1]
    centers[:, constant] = feature_bounds[0][constant]
    return centers


def _numpy_memberships_objective(x, centers, tau):
    n, d = x.shape
    k = len(centers)
    u = np.empty((n, k))
    rows = max(1, min(256, (1 << 18)//(k*d)))
    objective = 0.0
    for start in range(0, n, rows):
        block = x[start:start+rows]
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            difference = block[:, None, :] - centers[None, :, :]
            costs = np.einsum('nkd,nkd->nk', difference, difference)
        _finite(costs, "squared distance")
        # Costs are rounded before this subtraction. A gap lost to float64
        # rounding remains a tie; there is deliberately no exact fallback.
        with np.errstate(over='ignore', under='ignore', invalid='raise'):
            membership = np.exp(-(costs-costs.min(axis=1, keepdims=True))/tau)
            membership /= membership.sum(axis=1, keepdims=True)
        positive = membership > 0
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            distortion = float(np.sum(membership*costs, dtype=np.float64))
            entropy = tau*float(np.sum(membership[positive]*np.log(membership[positive]), dtype=np.float64))
            part = distortion + entropy
            objective += part
        _finite([distortion, entropy, part, objective], "objective")
        u[start:start+len(block)] = membership
    return u, objective


def fit_entropy_fcm(X, n_clusters=None, *, init=None, tau=1.0,
                    max_iter=300, tol=1e-5, random_state=None,
                    backend="numpy", return_history=False):
    """Fit entropy-regularized fuzzy c-means with float64-only arithmetic.

    Minimize sum(u_ik * ||x_i-v_k||**2) + tau * sum(u_ik * log(u_ik)),
    with u >= 0, row sums one, tau > 0, and 0*log(0) = 0. Centers use u,
    not u**m. Each iteration computes centers from old memberships, then
    memberships using a minimum-shifted softmax of float64 squared distances.

    X is (N,D); init is an optional (N,K) nonnegative membership matrix with
    positive row sums, normalized on a copy; n_clusters can be inferred.
    random_state uses NumPy's generator. tol tests absolute Frobenius membership
    change; tol=0 runs max_iter. Only backend='numpy' is supported. The former
    exact 'reference' backend is rejected rather than silently approximated.

    Return owned centers (K,D), membership (N,K), labels (N,), objective, fpc,
    n_iter, converged, delta, tau, backend, and numerical_diagnostics; optional
    objective_history records each iteration. Empty clusters retain their prior
    center (initially the data mean). A membership-only stopping test does not
    certify a stationary center update. Nonfinite center/distance/objective
    intermediates raise ValueError. Subnormal terms can underflow, cancellation
    can erase residuals, and large common costs can erase small distance gaps.
    No arbitrary-precision or exact-reference repair is performed.
    """
    tau = _temperature(tau)
    if backend != "numpy":
        raise ValueError("entropy FCM supports only backend='numpy' (float64); reference mode was removed")
    x, u = _validate(X, init, n_clusters, 2.0, max_iter, tol, 1, random_state)
    x = x.copy()
    feature_bounds = (x.min(axis=0), x.max(axis=0))
    # Divide first to avoid an unnecessary overflowing unweighted sum.
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        mean = np.sum(x / len(x), axis=0, dtype=np.float64)
    _finite(mean, "initial mean")
    centers = np.tile(mean, (u.shape[1], 1))
    history = []
    converged = False
    for iteration in range(1, int(max_iter) + 1):
        centers = _numpy_centers(x, u, centers, feature_bounds)
        next_u, objective = _numpy_memberships_objective(x, centers, tau)
        delta = 0.0
        for old, new in zip(u.flat, next_u.flat):
            delta = math.hypot(delta, float(new) - float(old))
        u = next_u
        if return_history:
            history.append(objective)
        converged = delta < tol
        if converged:
            break
    result = dict(centers=centers, membership=u, labels=u.argmax(axis=1),
                  objective=objective, fpc=float(np.sum(u * u) / len(u)),
                  n_iter=iteration, converged=converged, delta=delta, tau=tau,
                  backend=backend, numerical_diagnostics={
                      "arithmetic": "float64",
                      "used_reference_fallback": False,
                      "objective_status": "finite",
                      "log_abs_objective": math.log(abs(objective)) if objective else -math.inf,
                      "objective_sign": (1 if objective > 0 else -1) if objective else 0,
                      "stopping_rule": "membership-frobenius",
                  })
    if return_history:
        result["objective_history"] = np.array(history, dtype=np.float64)
    return result
