"""Opt-in SOM-OLP probability-tail experiment; dev4 backends remain unchanged.

Only initial probability tails on otherwise ordinary coordinates are localized.
The original extended-exponent implementation remains the fallback and oracle.
Direct costs and ordinary reductions are float64, not bitwise cold-path replay.
"""
import math
import numpy as np
from scipy.spatial.distance import cdist
from .validation import matrix_input, matrix
from .policy import policy_or_default
from .som_olp import _options, _run_som_olp_ordinary, run_som_olp, initialize_som_olp
from ._som_extreme import (needs_extreme, _weighted_mean, _pair, _multiply,
                          _sum, _float, _cost, _probability_row)


def _max_abs_columns(array):
    # No full-sized absolute-value copy.
    return np.array([max(abs(float(array[:, f].min())),
                         abs(float(array[:, f].max()))) for f in range(array.shape[1])])


def _repair_statistics(X, R, P, den, W, V, xmax, rmax):
    """Repair unsafe output coordinates from exact binary products.

    The convex-mean bounds cover signed cancellation, including products which
    round to zero before a later cancellation. Tiny nonempty column masses are
    always repaired, as in dev4. Grid reductions are unnormalized old-P sums.
    """
    nonempty = den > 0
    suspect = (np.abs(W) <= 1e-10 * xmax) & (xmax > 0)
    suspect |= ((den > 0) & (den < 1e-140))[:, None]
    suspect &= nonempty[:, None]
    repaired_w = int(np.count_nonzero(suspect))
    for j, f in zip(*np.nonzero(suspect)):
        W[j, f] = _weighted_mean(X[:, f], P[:, j])
    # Nonnegative row-stochastic P gives |sum(P*R)| <= max(abs(R)).
    # The accepted row-sum tolerance is covered conservatively by factor two.
    suspect_v = (np.abs(V) <= 2e-10 * rmax) & (rmax > 0)
    repaired_v = int(np.count_nonzero(suspect_v))
    for i, h in zip(*np.nonzero(suspect_v)):
        V[i, h] = _weighted_mean(R[:, h], P[i], normalize=False)
    return repaired_w, repaired_v


def _run_localized(X, R, W0, P0, gamma, lam, max_iters, tol, policy):
    n, d = X.shape; m, q = R.shape
    # Outputs, owned validated inputs, exact-integer reductions, and runtime
    # workspace are excluded, consistently with the existing primary caps.
    fixed = 8 * (3*m*d + 3*m + 3*n*q + d + q)
    per_row = 10*m + 128  # float64 distance scratch, Boolean masks, row vectors
    available = policy.max_scratch_bytes - fixed
    if available < per_row:
        raise ValueError('scratch cap cannot hold localized SOM statistics and one cost row')
    block = min(n, policy.block_rows, available // per_row)
    W = W0.copy(); P = P0; Pout = np.empty_like(P0); V = np.empty((n, q))
    numerator = np.empty_like(W); scratch = np.empty((block, m))
    xmax = _max_abs_columns(X); rmax = _max_abs_columns(R)
    mantissas = np.empty(m); exponents = np.empty(m, dtype=np.int64)
    history = []; repairs_w = repairs_v = cold_rows = 0
    temperature = _pair(lam)
    # A squared grid difference may underflow before a large gamma restores
    # its contribution. Include that amplification in the unsafe-row bound.
    cost_floor = 1e-280 * max(1., float(gamma)) * max(d, q)
    for _ in range(max_iters):
        np.matmul(P, R, out=V)
        den = P.sum(axis=0)
        np.matmul(P.T, X, out=numerator)
        np.divide(numerator, den[:, None], out=W, where=den[:, None] > 0)
        rw, rv = _repair_statistics(X, R, P, den, W, V, xmax, rmax)
        repairs_w += rw; repairs_v += rv
        row_objectives = []
        for start in range(0, n, block):
            end = min(start + block, n); Pb = Pout[start:end]
            with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
                cdist(X[start:end], W, 'sqeuclidean', out=Pb)
                if gamma:
                    tmp = scratch[:end-start]
                    cdist(V[start:end], R, 'sqeuclidean', out=tmp)
                    tmp *= gamma; Pb += tmp
                minimum = Pb.min(axis=1); maximum = Pb.max(axis=1)
                # Costs too small to trust squared products, or overflowing
                # composite costs, retain the original exponent arithmetic.
                cold = (minimum < cost_floor) | ~np.isfinite(maximum)
                Pb[cold] = 0.
                minimum[cold] = 0.
                Pb -= minimum[:, None]
                np.divide(Pb, -lam, out=Pb)
                np.exp(Pb, out=Pb)
                unit = np.count_nonzero(Pb == 1., axis=1)
                # Keep representable entropy tails even if 1+tail rounds to 1.
                tail = np.sum(Pb, axis=1, where=Pb != 1.)
                total = unit + tail
                Pb /= total[:, None]
                log_total = np.log(unit) + np.log1p(tail / unit)
            for row in range(end-start):
                if cold[row]:
                    i = start + row
                    for j in range(m):
                        mantissas[j], exponents[j] = _cost(X[i], W[j], V[i], R[j], gamma)
                    row_objectives.append(_probability_row(mantissas, exponents, lam, Pb[row]))
                    cold_rows += 1
                else:
                    row_objectives.append(_sum((_pair(minimum[row]),
                        _multiply(temperature, _pair(-log_total[row])))))
        objective = _float(_sum(row_objectives), 'objective')
        P = Pout; history.append(objective)
        if len(history) > 1:
            previous = history[-2]; difference = abs(objective - previous)
            relative = (difference / max(1., abs(previous)) if math.isfinite(difference)
                        else abs(objective / max(1., abs(previous)) - previous / max(1., abs(previous))))
            if relative <= tol:
                break
    return dict(W=W, P=Pout, V=V, history=np.asarray(history), n_iter=len(history),
                variant='experimental_localized_probability_tails', backend='cdist_optimized',
                numerical_contract='float64 direct costs; exact binary guarded mean repairs; extended-exponent unsafe cost rows',
                exact_prototype_repairs=repairs_w, exact_grid_repairs=repairs_v,
                direct_fallback_rows=cold_rows, scratch_rows=block,
                primary_scratch_budgeted_bytes=fixed + block*per_row,
                scratch_scope='primary NumPy temporaries; excludes outputs, input validation, exact integers, runtime workspace')


def run_som_olp_localized(X, R, W0, P0, *, gamma, lam, max_iters=100, tol=1e-4,
                          backend='cdist_optimized', policy=None):
    """Experimental opt-in candidate; all existing entry points are unchanged.

    Direct NumPy/SciPy requests can localize initial probability tails. Explicit
    optional/GEMM/threaded backend requests keep their original implementation.
    No global bitwise agreement or nonconvex trajectory equivalence is claimed.
    """
    X, _ = matrix_input(X, dtype=np.float64)
    R = matrix(R, dtype=np.float64, name='R')
    W0 = matrix(W0, dtype=np.float64, name='W0')
    P0 = matrix(P0, dtype=np.float64, name='P0')
    _options(gamma, lam, max_iters, tol); policy = policy_or_default(policy)
    if (max_iters == 0 or backend not in ('cdist', 'numpy', 'auto', 'cdist_optimized')
            or needs_extreme(X, R, W0, gamma=gamma)):
        return run_som_olp(X, R, W0, P0, gamma=gamma, lam=lam, max_iters=max_iters,
                           tol=tol, backend=backend, policy=policy)
    if not needs_extreme(P0):
        return _run_som_olp_ordinary(X, R, W0, P0, gamma=gamma, lam=lam,
                                    max_iters=max_iters, tol=tol, backend=backend, policy=policy)
    # Preserve dev4's initial-P restriction and requested-backend validation.
    _run_som_olp_ordinary(X, R, W0, P0, gamma=gamma, lam=lam, max_iters=0,
                           tol=tol, backend=backend, policy=policy)
    if np.any(P0 < 0) or not np.allclose(P0.sum(axis=1), 1., rtol=0., atol=1e-8):
        raise ValueError('P0 must be nonnegative and row-stochastic within 1e-8')
    n, d = X.shape; m, q = R.shape
    required = 8 * (3*m*d + 3*m + 3*n*q + d + q) + 10*m + 128
    if policy.max_scratch_bytes < required:
        # A smaller cap may fit the original exponent row even when it cannot
        # fit vectorized statistics; do not reject a formerly valid request.
        return run_som_olp(X, R, W0, P0, gamma=gamma, lam=lam, max_iters=max_iters,
                           tol=tol, backend=backend, policy=policy)
    with policy.activate():
        result = _run_localized(X, R, W0, P0, gamma, lam, max_iters, tol, policy)
    result['backend_requested'] = backend
    return result


def fit_som_olp_localized(X, R, *, gamma, lam, max_iters=100, tol=1e-4,
                          pca_scale=2., backend='cdist_optimized', policy=None,
                          initializer='original'):
    W, P = initialize_som_olp(X, R, lam, pca_scale=pca_scale, policy=policy,
                              initializer=initializer)
    result = run_som_olp_localized(X, R, W, P, gamma=gamma, lam=lam,
                                  max_iters=max_iters, tol=tol,
                                  backend=backend, policy=policy)
    result['initializer_requested'] = initializer
    return result
