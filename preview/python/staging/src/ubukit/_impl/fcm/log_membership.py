"""Stable log-domain FCM membership alternatives for controlled ablation.

These are independently callable research kernels; the production fit default
remains the inverse-power kernel in core.py. The ablation script changes only
that kernel in an isolated process, outside timed regions.
"""
import numpy as np
from scipy.special import logsumexp, softmax


def centered_logits(d2, m, out=None, *, check=True):
    """Compute -log(q/q_min)/(m-1), preserving near-tie information.

    log(q)-log(q_min) loses small relative changes for large/small q. For
    nearby values Sterbenz subtraction + log1p is more accurate. Far values
    use separate logs so q/q_min cannot overflow.
    """
    d2 = np.asarray(d2, dtype=np.float64)
    if check and (d2.ndim != 2 or d2.shape[1] == 0 or not np.isfinite(d2).all() or np.any(d2 < 0)):
        raise ValueError('finite nonnegative 2D squared distances required')
    if check and (not np.isfinite(m) or m <= 1):
        raise ValueError('m must be finite and greater than 1')
    if out is None:
        out = np.empty_like(d2)
    elif np.shares_memory(d2, out):
        d2 = d2.copy()
    minimum = d2.min(axis=1, keepdims=True)
    zeros = minimum[:, 0] == 0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        np.log(d2, out=out)
        out -= np.log(minimum)
        close = (d2 - minimum) <= minimum
        rows, cols = np.nonzero(close & (minimum > 0))
        out[rows, cols] = np.log1p((d2[rows, cols]-minimum[rows, 0])/minimum[rows, 0])
        out /= -(m - 1.)
    if np.any(zeros):
        out[zeros] = np.where(d2[zeros] == 0, 0., -np.inf)
    return out


def membership_logsoftmax(d2, m, out=None):
    out = centered_logits(d2, m, out, check=False)
    # max is already zero after q_min centering, including exact-zero rows.
    np.exp(out, out=out)
    out /= out.sum(axis=1, keepdims=True)
    return out


def membership_logsumexp(d2, m, out=None):
    out = centered_logits(d2, m, out, check=False)
    out -= logsumexp(out, axis=1, keepdims=True)
    np.exp(out, out=out)
    return out


def membership_scipy_softmax(d2, m, out=None):
    logits = centered_logits(d2, m, out, check=False)
    result = softmax(logits, axis=1)
    logits[:] = result
    return logits


def membership_naive_logsoftmax(d2, m, out=None):
    """Conventional direct logs: correct normally, near-tie rounding caveat."""
    if out is None:
        out = np.empty_like(d2)
    zeros = d2.min(axis=1) == 0
    with np.errstate(divide='ignore'):
        np.log(d2, out=out)
    out /= -(m-1)
    if np.any(zeros):
        out[zeros] = np.where(d2[zeros] == 0, 0., -np.inf)
    out -= out.max(axis=1, keepdims=True)
    np.exp(out, out=out)
    out /= out.sum(axis=1, keepdims=True)
    return out
