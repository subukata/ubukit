"""Frozen reference graph used for strict numeric fallback and testing."""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .core import _squared

def graph_reference(X, delta, max_edges, block_size):
    """Preflight candidate counts, then construct symmetric self-including CSR.

    A conservative radius inflation only selects candidates; the final predicate
    is direct float64 sqrt(sum((xi-xj)**2)) <= delta. It does not inflate delta.
    """
    n, d = X.shape
    tree = cKDTree(X)
    if delta == 0:
        radius = 0.0
    else:
        radius = np.nextafter(delta * (1 + 8 * np.finfo(float).eps * max(1, d)), np.inf)
        if not np.isfinite(radius):
            radius = np.inf
    counts = tree.query_ball_point(X, radius, p=2, eps=0, workers=1, return_length=True)
    candidate_edges = int(sum(map(int, counts)))
    if candidate_edges > max_edges:
        raise MemoryError(f"delta graph has up to {candidate_edges:,} candidate directed edges, exceeding max_edges={max_edges:,}; reduce delta or explicitly raise the limit")
    # Candidates can be larger than accepted edges by a few boundary points.
    indices = np.empty(candidate_edges, dtype=np.int64)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    pos = 0
    edge_block = max(1, 1_048_576 // d)
    for lo in range(0, n, block_size):
        hi = min(n, lo + block_size)
        neighbors = tree.query_ball_point(X[lo:hi], radius, p=2, eps=0, workers=1, return_sorted=True)
        lengths = np.fromiter(map(len, neighbors), dtype=np.int64, count=hi-lo)
        ids = np.concatenate(neighbors).astype(np.int64, copy=False)
        rows = np.repeat(np.arange(lo, hi), lengths)
        keep = np.empty(len(ids), dtype=bool)
        for a in range(0, len(ids), edge_block):
            b = min(len(ids), a + edge_block)
            diff = X[ids[a:b]] - X[rows[a:b]]
            if delta == 0:
                keep[a:b] = np.all(diff == 0, axis=1)
            else:
                keep[a:b] = np.sqrt(_squared(diff)) <= delta
        accepted = ids[keep]
        degrees_block = np.bincount(rows[keep] - lo, minlength=hi-lo)
        indices[pos:pos + len(accepted)] = accepted
        indptr[lo+1:hi+1] = pos + np.cumsum(degrees_block)
        pos += len(accepted)
    indices = indices[:pos].copy()
    degrees = np.diff(indptr)
    if np.any(degrees == 0):
        raise RuntimeError("internal graph error: a self-including neighborhood cannot be empty")
    inverse = 1.0 / degrees
    values = np.repeat(inverse, degrees)
    P = sparse.csr_matrix((values, indices, indptr), shape=(n, n))
    P.sort_indices()
    for a in (P.data, P.indices, P.indptr):
        a.flags.writeable = False
    return P, degrees, candidate_edges

