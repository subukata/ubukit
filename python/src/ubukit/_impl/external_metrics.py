"""Exact-definition adjusted Rand and mutual information scores.

The ordinary path requires NumPy only; singular/ill-conditioned AMI uses a lazy
scikit-learn compatibility fallback. Optional Numba accelerates the complete
hypergeometric support recurrence; compilation is excluded only from explicitly
warm timings. No process-global data cache is used. Labels are encoded anew for
every public call; ``adjusted_scores`` shares the contingency within that call.
"""
from __future__ import annotations

import math
import warnings
import numpy as np

__all__ = ["adjusted_rand_score", "adjusted_mutual_info_score", "adjusted_scores"]
_EPS = np.finfo(np.float64).eps
_AVERAGES = ("arithmetic", "geometric", "min", "max")


def _labels(x, name):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D: shape is {x.shape!r}")
    if x.dtype.kind in "fc":
        if x.dtype.kind == "c":
            raise ValueError("Complex labels are not supported")
        if not np.all(np.isfinite(x)):
            raise ValueError("Labels must be finite")
        if np.any(x != np.floor(x)):
            warnings.warn("Clustering metrics expects discrete values but received continuous labels", UserWarning, stacklevel=3)
    elif x.dtype.kind == "O":
        # np.unique supplies the comparison/type errors for unsupported mixed
        # object labels. Detect floating NaN/inf before it can become a cluster.
        if any(isinstance(v, (float, np.floating)) and not math.isfinite(v) for v in x):
            raise ValueError("Labels must be finite")
    return x


def _encode(x):
    """Dense small-integer fast path; arbitrary labels use NumPy factorization."""
    n = x.size
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    if x.dtype.kind in "biu":
        low, high = int(x.min()), int(x.max())
        span = high - low + 1
        if span <= max(256, 4 * n) and span <= 4_000_000:
            # Convert only after subtracting the offset in the original domain;
            # avoid uint64/int64 overflow for labels near either extreme.
            if low >= 0 and high <= np.iinfo(np.int64).max:
                codes = x.astype(np.int64, copy=False) - low
            elif x.dtype.kind == "u":
                codes = (x - np.uint64(low)).astype(np.int64)
            else:
                codes = (x - np.int64(low)).astype(np.int64, copy=False)
            counts = np.bincount(codes, minlength=span)
            used = counts != 0
            if not np.all(used):
                remap = np.cumsum(used, dtype=np.int64) - 1
                codes = remap[codes]
                counts = counts[used]
            return codes, counts
    _, codes, counts = np.unique(x, return_inverse=True, return_counts=True)
    return codes, counts


def _contingency(labels_true, labels_pred):
    x, y = _labels(labels_true, "labels_true"), _labels(labels_pred, "labels_pred")
    if x.size != y.size:
        raise ValueError("labels_true and labels_pred must have equal lengths")
    u, a = _encode(x)
    v, b = _encode(y)
    n, ka, kb = x.size, a.size, b.size
    if n == 0:
        e = np.empty(0, np.int64)
        return n, a, b, e, e, e
    # ka * kb <= n**2. Realizable arrays cannot approach int64 index overflow,
    # but retain a safe structured-key path rather than relying on that fact.
    if ka * kb <= np.iinfo(np.int64).max:
        keys = u * kb + v
        cells = ka * kb
        if cells <= max(256, 4 * n) and cells <= 4_000_000:
            counts = np.bincount(keys, minlength=cells)
            nz = np.flatnonzero(counts)
            counts = counts[nz]
        else:
            nz, counts = np.unique(keys, return_counts=True)
        rows, cols = nz // kb, nz % kb
    else:
        pairs = np.rec.fromarrays((u, v), names=("u", "v"))
        nz, counts = np.unique(pairs, return_counts=True)
        rows, cols = nz.u, nz.v
    return n, a, b, rows, cols, counts


def _squares(counts, n):
    # This bound prevents overflow in every product and the summed int64 dot.
    if n <= 3_037_000_499:
        return int(np.dot(counts, counts))
    return sum(int(c) ** 2 for c in counts)


def _ari(n, a, b, counts):
    q = _squares(counts, n)
    sa, sb = _squares(a, n), _squares(b, n)
    tp, fp, fn = q - n, sb - q, sa - q
    if fp == 0 and fn == 0:
        return 1.0
    tn = n * n - sa - sb + q
    # Python integers keep the cross-products exact even for very large N.
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def _entropy(counts, n):
    p = counts / n
    return float(-np.dot(p, np.log(p)))


def _mi(n, a, b, rows, cols, counts):
    # Sum logs instead of multiplying a*b in int64. This has no integer
    # overflow even for synthetic counts larger than an in-memory label array.
    p = counts / n
    terms = p * (np.log(counts) + math.log(n) - np.log(a[rows]) - np.log(b[cols]))
    terms[np.abs(terms) < _EPS] = 0.0
    return max(0.0, float(np.sum(terms)))


def _margin_pairs(a, b):
    av, ac = np.unique(a, return_counts=True)
    bv, bc = np.unique(b, return_counts=True)
    # E_ij depends only on the unordered pair of marginal sizes. Aggregate
    # multiplicities in integers; this is within-call compression, not a cache.
    left = np.repeat(av, len(bv))
    right = np.tile(bv, len(av))
    multiplicity = (ac[:, None] * bc[None, :]).ravel()
    low, high = np.minimum(left, right), np.maximum(left, right)
    order = np.lexsort((high, low))
    low, high, multiplicity = low[order], high[order], multiplicity[order]
    starts = np.r_[0, np.flatnonzero((low[1:] != low[:-1]) | (high[1:] != high[:-1])) + 1]
    return np.column_stack((low[starts], high[starts], np.add.reduceat(multiplicity, starts)))


def _pair_emi_numpy(n, a, b):
    """Full hypergeometric support, normalized mode-centered recurrence.

    Probability weights start at a mode, then run outwards using exact adjacent
    probability ratios. Normalizing their sum avoids log-gamma cancellation.
    This is the permutation-model expectation, without truncation/sampling.
    """
    lo, hi = max(0, a + b - n), min(a, b)
    mode = min(hi, max(lo, ((a + 1) * (b + 1)) // (n + 2)))
    x = np.arange(lo, hi + 1, dtype=np.float64)
    weights = np.empty(x.size, np.float64)
    ix = mode - lo
    weights[ix] = 1.0
    if mode < hi:
        t = x[ix + 1:]
        ratios = ((a - t + 1) / t) * ((b - t + 1) / (n - a - b + t))
        weights[ix + 1:] = np.cumprod(ratios)
    if mode > lo:
        t = x[:ix][::-1]
        ratios = ((t + 1) / (a - t)) * ((n - a - b + t + 1) / (b - t))
        weights[:ix] = np.cumprod(ratios)[::-1]
    positive = x > 0
    xp = x[positive]
    information = (xp / n) * (np.log(xp) + math.log(n) - math.log(a) - math.log(b))
    return float(np.dot(weights[positive], information) / np.sum(weights))


def _emi_numpy_reference(n, pairs):
    return math.fsum(int(mult) * _pair_emi_numpy(n, int(a), int(b)) for a, b, mult in pairs)


def _emi_numpy(n, pairs, *, presorted=False):
    """Bounded-memory batching of the same full-support recurrence."""
    # Similar support lengths are adjacent, limiting padded zero work.
    # _margin_pairs already orders (low, high) lexicographically.
    if not presorted:
        pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    result = []
    start = 0
    budget = 131_072
    while start < len(pairs):
        end = start + 1
        # a <= b, so a+1 bounds the complete support length.
        while end < len(pairs) and (end + 1 - start) * (int(pairs[end, 0]) + 1) <= budget:
            end += 1
        aa, bb, multiplicity = pairs[start:end].T
        a, b = aa.astype(float), bb.astype(float)
        lo = np.maximum(0, aa + bb - n)
        mode = np.minimum(aa, np.maximum(lo, np.floor((a + 1) * (b + 1) / (n + 2)).astype(np.int64)))
        logs = math.log(n) - np.log(a) - np.log(b)
        total = np.ones(end - start)
        value = (mode / n) * (np.log(np.maximum(mode, 1)) + logs)
        width = int(np.max(aa - mode))
        if width:
            x = mode[:, None] + np.arange(1, width + 1, dtype=float)
            ratio = ((a[:, None] - x + 1) / x) * ((b[:, None] - x + 1) / (n - a[:, None] - b[:, None] + x))
            ratio[x > a[:, None]] = 0.0
            weight = np.cumprod(ratio, axis=1)
            total += weight.sum(axis=1)
            value += (weight * (x / n) * (np.log(x) + logs[:, None])).sum(axis=1)
        width = int(np.max(mode - lo))
        if width:
            x = mode[:, None] - np.arange(1, width + 1, dtype=float)
            ratio = ((x + 1) / (a[:, None] - x)) * ((n - a[:, None] - b[:, None] + x + 1) / (b[:, None] - x))
            ratio[x < lo[:, None]] = 0.0
            weight = np.cumprod(ratio, axis=1)
            total += weight.sum(axis=1)
            # Zero and padded negative intersections have zero contribution.
            xp = np.maximum(x, 0)
            value += (weight * (xp / n) * (np.log(np.maximum(xp, 1)) + logs[:, None])).sum(axis=1)
        result.extend((multiplicity * value / total).tolist())
        start = end
    return math.fsum(result)


def _emi(n, a, b, backend):
    pairs = _margin_pairs(a, b)
    if backend == "numpy":
        return _emi_numpy(n, pairs, presorted=True)
    if backend == "numba":
        from ._external_metrics_numba import emi_numba
        return float(emi_numba(n, pairs))
    raise ValueError("backend must be 'numpy' or 'numba'")


def _ami(n, a, b, rows, cols, counts, average_method, backend):
    ka, kb = a.size, b.size
    if (ka == kb == 0) or (ka == kb == 1):
        return 1.0
    if ka == 1 or kb == 1:
        return 0.0
    # A one-to-one contingency mapping means the two partitions are identical,
    # including the all-singletons 0/0 limiting case.
    if counts.size == ka == kb:
        return 1.0
    # Singleton-vs-other has MI == EMI exactly and (for min) an undefined
    # 0/0 normalization. Preserve sklearn's actual rounded/clamped behavior
    # through the explicit compatibility fallback, including non-min roundoff.
    if ka == n or kb == n:
        return None
    # Conservative empirical compatibility screen: sklearn's ungrouped EMI
    # summation can accumulate visible rounding error for singleton-heavy/high-K
    # tables. Check this before computing EMI. This is not a universal error
    # bound; absolute parity is established on the published test panel.
    if n >= 256 and 4 * max(ka, kb) >= n:
        return None
    mi = _mi(n, a, b, rows, cols, counts)
    emi = _emi(n, a, b, backend)
    ha, hb = _entropy(a, n), _entropy(b, n)
    if average_method == "arithmetic":
        normalizer = (ha + hb) / 2
    elif average_method == "geometric":
        normalizer = math.sqrt(ha * hb)
    elif average_method == "min":
        normalizer = min(ha, hb)
    else:
        normalizer = max(ha, hb)
    denominator, numerator = normalizer - emi, mi - emi
    if abs(denominator) <= (0.01 if n >= 256 else 1e-10) * max(1.0, abs(normalizer), abs(emi)):
        return None
    denominator = min(denominator, -_EPS) if denominator < 0 else max(denominator, _EPS)
    numerator = min(numerator, -_EPS) if numerator < 0 else max(numerator, _EPS)
    return float(numerator / denominator)


def _compatible_ami(labels_true, labels_pred, data, average_method, backend):
    result = _ami(*data, average_method, backend)
    if result is not None:
        return result
    try:
        from sklearn.metrics import adjusted_mutual_info_score as reference
    except ImportError as exc:
        raise ImportError("scikit-learn is required for the singular/ill-conditioned AMI compatibility fallback") from exc
    return float(reference(labels_true, labels_pred, average_method=average_method))


def _validate_options(average_method, backend):
    if average_method not in _AVERAGES:
        raise ValueError(f"average_method must be one of {_AVERAGES}")
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be 'numpy' or 'numba'")


def adjusted_rand_score(labels_true, labels_pred):
    """Adjusted Rand index; labels may be integer, boolean, or string arrays."""
    n, a, b, rows, cols, counts = _contingency(labels_true, labels_pred)
    return _ari(n, a, b, counts)


def adjusted_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic", backend="numpy"):
    """Permutation-adjusted mutual information, arithmetic normalization default.

    ``average_method`` also accepts ``min``, ``max``, and ``geometric``.
    ``backend='numpy'`` requires no JIT; ``'numba'`` is an optional acceleration.
    Singular/ill-conditioned normalization delegates to scikit-learn so its
    rounded behavior is preserved; that fallback requires scikit-learn.
    """
    _validate_options(average_method, backend)
    return _compatible_ami(labels_true, labels_pred, _contingency(labels_true, labels_pred), average_method, backend)


def adjusted_scores(labels_true, labels_pred, *, average_method="arithmetic", backend="numpy"):
    """Return ``{'ari': ARI, 'ami': AMI}``, sharing encoding and contingency."""
    _validate_options(average_method, backend)
    n, a, b, rows, cols, counts = _contingency(labels_true, labels_pred)
    return {"ari": _ari(n, a, b, counts), "ami": _compatible_ami(labels_true, labels_pred, (n, a, b, rows, cols, counts), average_method, backend)}
