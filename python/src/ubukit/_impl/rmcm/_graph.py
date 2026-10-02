"""Exact radius graphs with bounded-memory candidate generation.

Every retained edge is tested using the original direct Float64 einsum/sqrt
predicate. Spatial and blocked Gram routes differ only in conservative candidate
screening. No N-by-N distance or adjacency matrix is allocated.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits, threadpool_info
from functools import lru_cache

_EPS = np.finfo(np.float64).eps
_TINY = np.finfo(np.float64).tiny


def _radius(delta, d):
    if delta == 0:
        return 0.0
    with np.errstate(over="ignore"):
        radius = np.nextafter(delta * (1 + 8 * _EPS * max(1, d)), np.inf)
    return float(radius) if np.isfinite(radius) else np.inf


def _limit(count, max_edges):
    if count > max_edges:
        raise MemoryError(
            f"delta graph has up to {count:,} candidate directed edges, exceeding "
            f"max_edges={max_edges:,}; reduce delta or explicitly raise the limit")


def _csr(n, pairs, candidate_edges):
    """Merge a sorted upper triangle with its transpose without COO sorting."""
    m = len(pairs)
    if m and np.any(pairs[1:, 0] < pairs[:-1, 0]):
        return _csr_coo(n, pairs, candidate_edges)
    dtype = np.int32 if max(n, n + 2 * m) < 2**31 else np.int64
    counts = np.bincount(pairs[:, 0], minlength=n)
    indptr = np.empty(n + 1, dtype=dtype)
    indptr[0] = 0
    np.cumsum(counts + 1, out=indptr[1:])
    indices = np.empty(n + m, dtype=dtype)
    indices[indptr[:-1]] = np.arange(n)
    indices[np.arange(m) + pairs[:, 0] + 1] = pairs[:, 1]
    upper = sparse.csr_matrix((np.ones(n + m), indices, indptr), shape=(n, n))
    P = upper + upper.T
    P.sort_indices()
    degrees = np.diff(P.indptr).astype(np.int64)
    P.data[:] = np.repeat(1.0 / degrees, degrees)
    for a in (P.data, P.indices, P.indptr):
        a.flags.writeable = False
    return P, degrees, candidate_edges


def _csr_coo(n, pairs, candidate_edges):
    """Create canonical CSR once; row order matches the original builder."""
    m = len(pairs)
    # scipy uses 32-bit CSR when shape/nnz permit it; do not first allocate
    # 64-bit directed row/column arrays only to have scipy copy them down.
    dtype = np.int32 if max(n, n + 2 * m) < 2**31 else np.int64
    rows = np.empty(n + 2 * m, dtype=dtype)
    cols = np.empty_like(rows)
    rows[:n] = np.arange(n)
    cols[:n] = rows[:n]
    rows[n:n + m], cols[n:n + m] = pairs[:, 0], pairs[:, 1]
    rows[n + m:], cols[n + m:] = pairs[:, 1], pairs[:, 0]
    degrees = np.bincount(rows, minlength=n)
    P = sparse.csr_matrix((np.ones(n + 2 * m), (rows, cols)), shape=(n, n))
    P.data[:] = np.repeat(1.0 / degrees, degrees)
    for a in (P.data, P.indices, P.indptr):
        a.flags.writeable = False
    return P, degrees, candidate_edges


def _bounded_tree_pairs(X, tree, radius, counts, max_edges, block_size):
    """Shrink the pair radius, then recover original-query boundary rows.

    The unchanged original point-query count bounds staging. An inner radius
    separated by 256*eps*D ensures its pair set is contained in that preflight
    relation on the audited cKDTree implementation. Missing rows are queried
    with the exact original radius and their candidate keys are deduplicated.
    """
    import scipy
    n, d = X.shape
    margin = 256 * _EPS * max(1, d)
    if (scipy.__version__ not in ("1.17.0", "1.17.1") or radius <= 0
            or not np.isfinite(radius) or margin >= 1e-4 or n * n > np.iinfo(np.int64).max
            or radius < 32 * np.sqrt(_TINY / _EPS) * np.sqrt(max(1, d))
            or radius > np.sqrt(np.finfo(float).max) / 4):
        return None
    inner = float(np.nextafter(radius * (1 - margin), 0.0))
    if inner <= 0:
        return None
    pairs = tree.query_pairs(inner, p=2, eps=0, output_type="ndarray")
    inner_counts = np.bincount(pairs.ravel(), minlength=n) + 1
    if np.any(inner_counts > counts):
        # Defensive fallback; never infer diagnostic counts from a pair query.
        return None
    changed = np.flatnonzero(inner_counts != counts)
    if not len(changed):
        return pairs
    keys = [pairs[:, 0] * n + pairs[:, 1]]
    for lo in range(0, len(changed), block_size):
        rows = changed[lo:lo + block_size]
        neighbors = tree.query_ball_point(X[rows], radius, p=2, eps=0, workers=1, return_sorted=True)
        for i, js in zip(rows, neighbors):
            ids = np.asarray(js, dtype=np.int64)
            ids = ids[ids != i]
            keys.append(np.minimum(i, ids) * n + np.maximum(i, ids))
    unique = np.unique(np.concatenate(keys))
    return np.column_stack((unique // n, unique % n))


def _complete_graph_certificate(X, delta):
    """Prove a complete graph without hiding any distance-underflow error.

    Nonzero coordinates with magnitude at least 2**-458 have representable
    adjacent spacing at least 2*sqrt(tiny64). Thus any nonzero coordinate
    difference has a normal square; all-equal points remain safe at any scale.
    """
    span = np.max(X, axis=0) - np.min(X, axis=0)
    if not np.any(span):
        return True
    squared = float(np.einsum("i,i->", span, span, optimize=False))
    with np.errstate(over="ignore"):
        upper = np.nextafter(np.sqrt(squared) * (1 + 16 * _EPS * max(1, X.shape[1])), np.inf)
    if upper > delta:
        return False
    threshold = np.ldexp(1., -458)
    flat = X.ravel()
    for lo in range(0, len(flat), 262_144):
        part = flat[lo:lo + 262_144]
        if np.any((part != 0) & (np.abs(part) < threshold)):
            return False
    return True


def graph_tree(X, delta, max_edges, block_size):
    """Dual-tree counting/pair traversal; exact direct final edge predicate."""
    from .core import _squared
    n, d = X.shape
    radius = _radius(delta, d)
    tree = cKDTree(X)
    if 0 < delta < 16 * np.sqrt(_TINY):
        from ._graph_reference import graph_reference
        return graph_reference(X, delta, max_edges, block_size)
    # Preserve the original preflight, diagnostic candidate count, and edge
    # budget decision exactly. count_neighbors is faster but can round a
    # candidate boundary differently, so it is not a drop-in default.
    counts = tree.query_ball_point(X, radius, p=2, eps=0, workers=1, return_length=True)
    count = int(sum(map(int, counts)))
    _limit(count, max_edges)
    if count == n:
        return _csr(n, np.empty((0, 2), dtype=np.int64), count)
    if np.all(X == X[0]) or (count == n * n and _complete_graph_certificate(X, delta)):
        # Pair differences are zero, or the complete-graph certificate proves
        # normal squared differences and inclusion. Materialize the same CSR.
        dtype = np.int32 if n * n < 2**31 else np.int64
        indices = np.tile(np.arange(n, dtype=dtype), n)
        indptr = np.arange(n + 1, dtype=dtype) * n
        P = sparse.csr_matrix((np.full(n * n, 1.0 / n), indices, indptr), shape=(n, n))
        for a in (P.data, P.indices, P.indptr):
            a.flags.writeable = False
        return P, np.full(n, n, dtype=np.int64), count
    pairs = None
    if n * n > max_edges:
        pairs = _bounded_tree_pairs(X, tree, radius, counts, max_edges, block_size)
        if pairs is None:
            # Uncertified versions/numeric regimes keep the original guard.
            with np.errstate(over="ignore"):
                upper_radius = np.nextafter(radius * (1 + 64 * _EPS * max(1, d)), np.inf)
            if int(tree.count_neighbors(tree, upper_radius)) > max_edges:
                from ._graph_reference import graph_reference
                return graph_reference(X, delta, max_edges, block_size)
    if pairs is None:
        pairs = tree.query_pairs(radius, p=2, eps=0, output_type="ndarray")
    keep = np.empty(len(pairs), dtype=bool)
    chunk = max(1, 1_048_576 // d)
    for lo in range(0, len(pairs), chunk):
        p = pairs[lo:lo + chunk]
        diff = X[p[:, 0]] - X[p[:, 1]]
        keep[lo:lo + chunk] = np.all(diff == 0, axis=1) if delta == 0 else np.sqrt(_squared(diff)) <= delta
    return _csr(n, pairs[keep], count)


@lru_cache(maxsize=1)
def _audited_float32_blas():
    """Keep uncertified BLAS providers on the established Float64 route."""
    providers = [x for x in threadpool_info() if x.get("user_api") == "blas"
                 and "numpy.libs" in x.get("filepath", "")]
    return np.__version__ == "2.3.5" and bool(providers) and all(x.get("internal_api") == "openblas"
                                  and x.get("version") == "0.3.30"
                                  and x.get("architecture") == "SkylakeX"
                                  for x in providers)


def _screening_coordinates(X, radius_squared):
    """Choose bounded Float32 screening only where its absolute envelope is useful.

    This precision choice affects candidates only. All final graph and original
    candidate-boundary predicates still use direct Float64 differences.
    """
    d = X.shape[1]
    Z = X - X[0]
    norms = np.einsum("ij,ij->i", Z, Z, optimize=False)
    factor = 512 * _EPS * max(1, d)
    f32 = np.finfo(np.float32)
    factor32 = float(128 * f32.eps * max(1, d))
    largest = float(np.max(norms))
    # Keep products and dot sums well inside normal Float32 range. Unhelpful
    # cancellation, extreme scales and very high D retain Float64 screening.
    if (d >= 8 and factor32 < 1e-2 and _audited_float32_blas() and np.isfinite(norms).all()
            and radius_squared >= 256 * float(f32.tiny)
            and np.isfinite(radius_squared) and radius_squared < float(f32.max) / 64
            and largest < float(f32.max) / 64
            and largest * factor32 <= radius_squared * .25):
        Z = Z.astype(np.float32)
        norms = np.einsum("ij,ij->i", Z, Z, optimize=False)
        factor = factor32
    return Z, norms, factor


def _gram_candidates(A, B, norms_a, norms_b, radius_squared, error_factor, d):
    """Conservative tile mask; never an authoritative edge predicate."""
    # Avoid uncertified Float32 GEMV/dot/SYRK dispatches on degenerate or
    # identical square tiles. The converted inputs are exact Float64 values;
    # this dot and its final cast have no larger error than the certified path.
    if A.dtype == np.float32 and (len(A) == 1 or len(B) == 1
            or (A.shape == B.shape and np.shares_memory(A, B))):
        estimate = (A.astype(np.float64) @ B.astype(np.float64).T).astype(np.float32)
    else:
        estimate = A @ B.T
    estimate *= -2
    estimate += norms_b
    estimate += norms_a[:, None]
    screen_tiny = np.finfo(A.dtype).tiny
    bound = error_factor * (norms_a[:, None] + norms_b) + 64 * screen_tiny * d
    with np.errstate(over="ignore"):
        if A.dtype == np.float32:
            eps32 = np.finfo(np.float32).eps
            outward_radius = float(np.float32(radius_squared * (1 + 2 * eps32)))
            bound += outward_radius
            bound *= 1 + 2 * eps32
            return estimate <= bound
        return estimate <= radius_squared + bound


def _graph_blocked_impl(X, delta, max_edges, block_size, strict=False):
    """Bounded Gram screening followed by the original direct predicate.

    The Gram identity is never the graph membership test. Its cancellation and
    summation error envelope deliberately overselects candidate pairs, which are
    recomputed by direct differences and einsum. Large translations can make
    screening less selective but cannot introduce graph edges. Temporary Gram
    tiles have at most 1,048,576 entries; stored accepted pairs stay O(max_edges).

    Candidate-edge accounting uses the direct inflated-radius predicate here,
    rather than scipy's internal cKDTree reduction. This may differ at the
    inflated *candidate* boundary, never at the requested graph boundary.
    """
    n, d = X.shape
    radius = _radius(delta, d)
    with np.errstate(over="ignore"):
        radius2 = radius * radius
        # Strict counting must see the ENTIRE ambiguity band, including
        # points just outside the nominal inflated candidate radius.
        screen_radius = radius * (1 + 64 * _EPS * max(1, d)) if strict else radius
        screen_radius2 = screen_radius * screen_radius
    # Subtraction of one observed point reduces cancellation on translated data.
    # The factor covers subtraction, products, dot reductions, norm reductions,
    # final additions, and the direct reference reduction, without fastmath.
    Z, norms, error_factor = _screening_coordinates(X, screen_radius2)
    # Keep every arithmetic stage far from overflow, and use the strict tree
    # implementation where the normal-roundoff proof does not apply usefully.
    if (0 < delta < 16 * np.sqrt(_TINY) or error_factor >= (1e-2 if Z.dtype == np.float32 else 1e-4)
            or not np.isfinite(norms).all()
            or np.max(norms) > np.finfo(float).max / 64):
        return graph_tree(X, delta, max_edges, block_size)
    # Screening loses value under extreme cancellation; this fallback is for
    # predictable cost as well as avoiding reliance on a very wide envelope.
    if np.max(norms) * error_factor > max(radius2, _TINY) * 0.25:
        return graph_tree(X, delta, max_edges, block_size)
    block = min(block_size, 128)
    width = max(block, min(n, 1_048_576 // block))
    edge_chunk = max(1, 1_048_576 // d)
    saved = []
    candidate_edges = n
    row_counts = np.ones(n, dtype=np.int64) if strict else None
    ambiguous_rows = np.zeros(n, dtype=bool) if strict else None
    exact_count_override = None
    underflow = False
    with np.errstate(over="ignore"):
        candidate_lower = radius * (1 - 64 * _EPS * max(1, d))
        candidate_upper = radius * (1 + 64 * _EPS * max(1, d))
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        for jl in range(lo, n, width):
            jh = min(n, jl + width)
            mask = _gram_candidates(Z[lo:hi], Z[jl:jh], norms[lo:hi], norms[jl:jh],
                                    screen_radius2, error_factor, d)
            if jl < hi:
                for i in range(hi - lo):
                    mask[i, :max(0, lo + i - jl + 1)] = False
            ai, bi = np.nonzero(mask)
            ai += lo
            bi += jl
            for a in range(0, len(ai), edge_chunk):
                rows, cols = ai[a:a + edge_chunk], bi[a:a + edge_chunk]
                diff = X[rows] - X[cols]
                if delta == 0:
                    accepted = candidates = np.all(diff == 0, axis=1)
                else:
                    sq = np.einsum("ij,ij->i", diff, diff, optimize=False)
                    distance = np.sqrt(sq)
                    candidates = distance <= radius
                    # Only true radius candidates have the original graph's
                    # underflow validation contract; broad Gram hits may not.
                    bad = np.any((sq[candidates] > 0) & (sq[candidates] < _TINY)) or np.any(diff[candidates & (sq == 0)] != 0)
                    if bad:
                        if strict:
                            underflow = True
                        else:
                            raise FloatingPointError("distance underflow; rescale X and delta")
                    accepted = distance <= delta
                    if strict:
                        ambiguous = (distance >= candidate_lower) & (distance <= candidate_upper)
                        ambiguous_rows[rows[ambiguous]] = True
                        ambiguous_rows[cols[ambiguous]] = True
                candidate_edges += 2 * int(np.count_nonzero(candidates))
                if strict:
                    row_counts += np.bincount(rows[candidates], minlength=n)
                    row_counts += np.bincount(cols[candidates], minlength=n)
                    if candidate_edges > max_edges and exact_count_override is None:
                        # Preserve global preflight-before-underflow ordering.
                        # A boundary correction can lower the direct count, so
                        # use the original query before any rejection.
                        counts = cKDTree(X).query_ball_point(X, radius, p=2, eps=0, workers=1, return_length=True)
                        exact_count_override = int(sum(map(int, counts)))
                        _limit(exact_count_override, max_edges)
                else:
                    _limit(candidate_edges, max_edges)
                if np.any(accepted):
                    saved.append(np.column_stack((rows[accepted], cols[accepted])))
    if strict:
        if exact_count_override is not None:
            candidate_edges = exact_count_override
        elif np.any(ambiguous_rows):
            counts = cKDTree(X).query_ball_point(X[ambiguous_rows], radius, p=2, eps=0, workers=1, return_length=True)
            candidate_edges = int(row_counts[~ambiguous_rows].sum()) + int(sum(map(int, counts)))
        _limit(candidate_edges, max_edges)
        if underflow:
            raise FloatingPointError("distance underflow; rescale X and delta")
    pairs = np.concatenate(saved) if saved else np.empty((0, 2), dtype=np.int64)
    return _csr(n, pairs, candidate_edges)


def graph_blocked(X, delta, max_edges, block_size):
    # Preparation had no BLAS thread knob before this optional route. Keep its
    # bounded tiles serial by default rather than spawning a machine-wide pool.
    with threadpool_limits(limits=1, user_api="blas"):
        return _graph_blocked_impl(X, delta, max_edges, block_size)


def graph_blocked_strict(X, delta, max_edges, block_size):
    """Eager exact original candidate counts via ambiguous-row correction.

    Away from the inflated-radius boundary, nonnegative Float64 reductions
    cannot change inclusion. Only rows with a pair in a 64*eps*D boundary band
    require the original query_ball_point count. The original global guard is
    resolved before any distance-underflow exception or CSR allocation.

    The bound/source audit currently covers SciPy 1.17.0/1.17.1; other versions
    retain the fully original preflight through the tree construction route.
    """
    import scipy
    if scipy.__version__ not in ("1.17.0", "1.17.1") or delta == 0 or _complete_graph_certificate(X, delta):
        return graph_tree(X, delta, max_edges, block_size)
    with threadpool_limits(limits=1, user_api="blas"):
        return _graph_blocked_impl(X, delta, max_edges, block_size, strict=True)


def _prefer_sparse_tree(X, delta):
    """Cheap, scale-aware performance hint; both routes remain exact.

    A small deterministic sample estimates local, per-coordinate spacing.
    Using nearest sampled points avoids treating distant clusters or isolated
    outliers as local scale. This is a speed heuristic, not a graph predicate.
    """
    n, d = X.shape
    if delta == 0:
        return True
    if d > 262_144:
        return False
    size = min(16, max(2, 131_072 // d), n)
    sample = X[np.linspace(0, n - 1, size, dtype=np.int64)]
    if np.all(sample == sample[0]):
        return True
    nearest = np.full(size, np.inf)
    for i in range(size - 1):
        diff = sample[i + 1:] - sample[i]
        sq = np.einsum("ij,ij->i", diff, diff, optimize=False)
        nearest[i] = min(nearest[i], float(sq.min()))
        nearest[i + 1:] = np.minimum(nearest[i + 1:], sq)
    local_scale = np.sqrt(float(np.median(nearest))) / np.sqrt(d)
    return delta < .2 * local_scale


def graph(X, delta, max_edges, block_size, graph_backend="tree"):
    if graph_backend == "auto":
        import scipy
        if (X.shape[1] >= 8 and len(X) >= 64
                and scipy.__version__ in ("1.17.0", "1.17.1")
                and not _prefer_sparse_tree(X, delta)):
            return graph_blocked_strict(X, delta, max_edges, block_size)
        return graph_tree(X, delta, max_edges, block_size)
    if graph_backend == "blocked-strict":
        return graph_blocked_strict(X, delta, max_edges, block_size)
    if graph_backend == "blocked":
        return graph_blocked(X, delta, max_edges, block_size)
    if graph_backend == "reference":
        from ._graph_reference import graph_reference
        return graph_reference(X, delta, max_edges, block_size)
    return graph_tree(X, delta, max_edges, block_size)


def candidate_pairs_blocked(X, radius, max_edges, block_size):
    """Bounded candidate-only builder for the optional moving-data cache.

    Numeric errors are deliberately left for the active graph's original guard
    and predicate. Cache storage is an independent upper-triangle superset.
    None requests the spatial fallback for unsuitable numerical regimes.
    """
    n, d = X.shape
    if n * n > max_edges and _complete_graph_certificate(X, radius):
        _limit(n * n, max_edges)
    with np.errstate(over="ignore"):
        radius2 = radius * radius
    Z, norms, error_factor = _screening_coordinates(X, radius2)
    if (error_factor >= (1e-2 if Z.dtype == np.float32 else 1e-4) or not np.isfinite(norms).all()
            or np.max(norms) > np.finfo(float).max / 64
            or np.max(norms) * error_factor > max(radius2, _TINY) * .25):
        return None
    saved = []
    count = n
    block = min(block_size, 128)
    width = max(block, min(n, 1_048_576 // block))
    chunk = max(1, 1_048_576 // d)
    with threadpool_limits(limits=1, user_api="blas"):
        for lo in range(0, n, block):
            hi = min(n, lo + block)
            for jl in range(lo, n, width):
                jh = min(n, jl + width)
                mask = _gram_candidates(Z[lo:hi], Z[jl:jh], norms[lo:hi], norms[jl:jh], radius2, error_factor, d)
                if jl < hi:
                    for i in range(hi - lo):
                        mask[i, :max(0, lo + i - jl + 1)] = False
                ai, bi = np.nonzero(mask)
                ai += lo
                bi += jl
                for a in range(0, len(ai), chunk):
                    rows, cols = ai[a:a + chunk], bi[a:a + chunk]
                    diff = X[rows] - X[cols]
                    sq = np.einsum("ij,ij->i", diff, diff, optimize=False)
                    keep = np.sqrt(sq) <= radius
                    count += 2 * int(np.count_nonzero(keep))
                    _limit(count, max_edges)
                    if np.any(keep):
                        saved.append(np.column_stack((rows[keep], cols[keep])))
    return np.concatenate(saved) if saved else np.empty((0, 2), dtype=np.int64)
