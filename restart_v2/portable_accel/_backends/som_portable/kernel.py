"""Dense float64 SOM-OLP with stable objective and guarded centered scores.

Exact-real equations and lagged outputs are preserved. BLAS, centered distance
identities and objective identities reorder floating arithmetic. No universal
bitwise or nonconvex-trajectory agreement is claimed. Inputs must be finite,
C-contiguous float64 arrays; P0 is intended to be nonnegative and row-stochastic.
"""
import operator
import time
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import ThreadpoolController
from .oracle import check_inputs

_controller = ThreadpoolController()


def _stop(obj, previous, tol):
    return previous is not None and abs(obj - previous) / max(1.0, abs(previous)) <= tol


def run(X, R, W0, P0, gamma, lam, max_iters=100, tol=1e-4, threads=1,
        *, distance='guarded', block_rows=None):
    """Common run API, plus explicit distance/block options.

    distance='guarded' centers feature/grid coordinates for GEMM scores, restores
    row constants to every objective, and recomputes cancellation-risk rows by
    direct squared differences. 'direct' uses cdist throughout. 'centered' is an
    explicitly unguarded algebraic comparison and is not the safe default.
    Setup/centering/allocation is included in each call, not cached externally.
    """
    max_iters = operator.index(max_iters)
    threads = operator.index(threads)
    if threads < 1:
        raise ValueError('threads must be positive')
    check_inputs(X, R, W0, P0, gamma, lam, max_iters)
    if distance not in ('guarded', 'direct', 'centered'):
        raise ValueError('distance must be guarded, direct or centered')
    n, d = X.shape
    m, k = R.shape
    block = n if block_rows is None else operator.index(block_rows)
    if block < 1:
        raise ValueError('block_rows must be positive')
    block = min(block, n)
    W = W0.copy()
    if max_iters == 0:
        return dict(W=W, P=P0.copy(), V=None, history=np.empty(0), n_iter=0,
                    variant='numpy_zero_iterations', setup_seconds=0.0,
                    direct_fallback_rows=0)
    started = time.perf_counter()
    with _controller.limit(limits=threads, user_api='blas'):
        P = P0  # borrowed until both old-P products have been consumed
        Pout = np.empty_like(P0)
        V = np.empty((n, k), dtype=np.float64)
        numerator = np.empty_like(W)
        scratch = np.empty((block, m), dtype=np.float64)
        history = np.empty(max_iters, dtype=np.float64)
        use_scores = distance != 'direct'
        translation_fallback = False
        if use_scores:
            xorigin = X.mean(axis=0)
            rorigin = R.mean(axis=0)
            Xc = np.ascontiguousarray(X - xorigin)
            Rc = np.ascontiguousarray(R - rorigin)
            xn = np.einsum('ij,ij->i', Xc, Xc)
            rn = np.einsum('ij,ij->i', Rc, Rc)
            if distance == 'guarded':
                # Conservative translation heuristic, not a formal error bound.
                limit = np.finfo(np.float64).eps ** (-0.25)
                xspread = float(np.sqrt(np.mean(xn) / d))
                rspread = float(np.sqrt(np.mean(rn) / k))
                translation_fallback = (
                    np.max(np.abs(xorigin)) > limit * max(1.0, xspread)
                    or np.max(np.abs(rorigin)) > limit * max(1.0, rspread))
                if translation_fallback:
                    use_scores = False
        setup_seconds = time.perf_counter() - started
        previous = None
        fallback_rows = 0
        actual = 0
        eps = np.finfo(np.float64).eps
        for iteration in range(max_iters):
            np.matmul(P, R, out=V)
            den = P.sum(axis=0)
            np.matmul(P.T, X, out=numerator)
            np.divide(numerator, den[:, None], out=W, where=den[:, None] > 0)
            if use_scores:
                Wc = W - xorigin
                Vc = V - rorigin
                wn = np.einsum('ij,ij->i', Wc, Wc)
                vn = np.einsum('ij,ij->i', Vc, Vc)
                node_constant = wn + gamma * rn
                row_constant = xn + gamma * vn
                if distance == 'guarded':
                    error_scale = 32.0 * eps * (
                        (d + 2) * (xn + float(wn.max()))
                        + gamma * (k + 2) * (vn + float(rn.max())))
            objective = 0.0
            for start in range(0, n, block):
                end = min(start + block, n)
                Pb = Pout[start:end]
                tmp = scratch[:end-start]
                if not use_scores:
                    cdist(X[start:end], W, 'sqeuclidean', out=Pb)
                    if gamma:
                        cdist(V[start:end], R, 'sqeuclidean', out=tmp)
                        tmp *= gamma
                        Pb += tmp
                    minimum = Pb.min(axis=1)
                    # Preserve scipy.softmax's divide-then-shift order here.
                    np.divide(Pb, -lam, out=Pb)
                    Pb -= Pb.max(axis=1, keepdims=True)
                    row_offset = 0.0
                    if translation_fallback:
                        fallback_rows += end-start
                else:
                    np.matmul(Xc[start:end], Wc.T, out=Pb)
                    Pb *= -2.0
                    if gamma:
                        np.matmul(Vc[start:end], Rc.T, out=tmp)
                        tmp *= -2.0 * gamma
                        Pb += tmp
                    Pb += node_constant
                    minimum = Pb.min(axis=1)
                    row_offset = row_constant[start:end].copy()
                    if distance == 'guarded':
                        bound = error_scale[start:end]
                        suspect = ((minimum + row_offset <= bound)
                                   | (bound > lam * 1e-8)
                                   | ~np.isfinite(minimum + row_offset))
                        if np.any(suspect):
                            indices = np.flatnonzero(suspect) + start
                            direct_cost = cdist(X[indices], W, 'sqeuclidean')
                            if gamma:
                                direct_cost += gamma * cdist(V[indices], R, 'sqeuclidean')
                            Pb[suspect] = direct_cost
                            minimum[suspect] = direct_cost.min(axis=1)
                            row_offset[suspect] = 0.0
                            fallback_rows += int(np.count_nonzero(suspect))
                    Pb -= minimum[:, None]
                    np.divide(Pb, -lam, out=Pb)
                np.exp(Pb, out=Pb)
                total = Pb.sum(axis=1)
                Pb /= total[:, None]
                objective += float(np.sum(minimum + row_offset - lam*np.log(total)))
            P = Pout
            history[iteration] = objective
            actual = iteration + 1
            if _stop(objective, previous, tol):
                break
            previous = objective
        return dict(W=W, P=Pout, V=V, history=history[:actual].copy(), n_iter=actual,
                    variant='numpy_' + distance, setup_seconds=setup_seconds,
                    direct_fallback_rows=fallback_rows)


def run_direct(*args, **kwargs):
    return run(*args, distance='direct', **kwargs)


def run_centered(*args, **kwargs):
    return run(*args, distance='centered', **kwargs)
