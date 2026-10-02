"""Float64 oracle, preserving the original repository's update/return order.

Original: https://github.com/subukata/som-olp, commit
4361175b776987d65c348d0132b31d43505e1069 (MIT). Initial W/P are separate
from the timed kernel. Returned V and W belong to the input P of the final
iteration, not the final P returned by the method.
"""
from contextlib import nullcontext

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax, xlogy
from .._threadpools import threadpool_context


def thread_context(threads):
    return threadpool_context(limits=threads, user_api="blas") if threads is not None else nullcontext()


def check_inputs(X, R, W0, P0, gamma, lam, max_iters):
    if not all(isinstance(a, np.ndarray) and a.dtype == np.float64 and a.flags.c_contiguous
               for a in (X, R, W0, P0)):
        raise ValueError("X, R, W0, P0 must be C-contiguous float64 arrays")
    if X.ndim != 2 or R.ndim != 2 or W0.shape != (R.shape[0], X.shape[1]) or P0.shape != (X.shape[0], R.shape[0]):
        raise ValueError("inconsistent input shapes")
    if min(*X.shape, *R.shape) < 1:
        raise ValueError("empty input dimensions are unsupported")
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("require finite lam > 0 and gamma >= 0")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 0:
        raise ValueError("max_iters must be a nonnegative integer")


def initialize(X, R, lam, pca_scale=2.0, threads=1):
    """Identical PCA initialization to original.py; no input mutation."""
    with thread_context(threads):
        X = np.asarray(X, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        R_c = R - R.mean(axis=0, keepdims=True)
        R_c /= np.maximum(np.abs(R_c).max(axis=0, keepdims=True), 1e-12)
        k = min(R.shape[1], X.shape[1])
        S = pca_scale * (s[:k] / np.sqrt(X.shape[0]))[None, :]
        W = mu + (R_c[:, :k] * S) @ Vt[:k]
        P = softmax(-cdist(X, W, "sqeuclidean") / lam, axis=1)
        return np.ascontiguousarray(W), np.ascontiguousarray(P)


def run(X, R, W0, P0, gamma, lam, max_iters=100, tol=1e-4, threads=1, *, trace=False):
    check_inputs(X, R, W0, P0, gamma, lam, max_iters)
    with thread_context(threads):
        W, P = W0.copy(), P0.copy()
        V = None
        history, trajectory = [], []
        prev_obj = None
        for _ in range(max_iters):
            V = P @ R
            den = P.sum(axis=0, keepdims=True).T
            np.divide(P.T @ X, den, out=W, where=den > 0)
            cost = cdist(X, W, "sqeuclidean") + gamma * cdist(V, R, "sqeuclidean")
            P = softmax(-cost / lam, axis=1)
            obj = float(np.sum(P * cost) + lam * np.sum(xlogy(P, P)))
            history.append(obj)
            if trace:
                trajectory.append({"W": W.copy(), "P": P.copy(), "V": V.copy(), "objective": obj})
            if prev_obj is not None and abs(obj - prev_obj) / max(1.0, abs(prev_obj)) <= tol:
                break
            prev_obj = obj
        result = {"W": W, "P": P, "V": V, "history": np.asarray(history), "n_iter": len(history)}
        if trace:
            result["trajectory"] = trajectory
        return result


def fit(X, R, gamma, lam, max_iters=100, tol=1e-4, threads=1, pca_scale=2.0, kernel=run, **kwargs):
    """Initialization plus kernel, measured separately from kernel-only timing."""
    W0, P0 = initialize(X, R, lam, pca_scale=pca_scale, threads=threads)
    return kernel(X, R, W0, P0, gamma, lam, max_iters=max_iters, tol=tol, threads=threads, **kwargs)
