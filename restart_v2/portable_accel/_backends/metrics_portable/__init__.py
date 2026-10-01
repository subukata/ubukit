"""Restart-v2 strict joint trustworthiness/continuity candidates.

The default imports no Numba. Optional JIT modules load only when explicitly
selected. Every public backend retains full sklearn distance call shapes,
floating dtype, independent per-k neighbor calls, and NumPy quicksort ties.
"""
from numbers import Integral
from threadpoolctl import threadpool_limits
from ._numpy import Quality, _validate_pair, _ks

BACKENDS = ('numpy', 'numba', 'sqrt_numpy', 'sqrt_numba')


def joint_quality(X, Y, ks=5, *, backend='numpy', threads=1, block_rows=256,
                  rank_method=None, max_scratch_bytes=32 * 2**20,
                  max_distance_bytes=None):
    """Always return a list of immutable Quality records, including penalties.

    block_rows partitions rank work only, NEVER the full-distance calculation.
    max_distance_bytes is a refusal guard for the largest one-space distance
    matrix, not whole-process RSS. max_scratch_bytes limits the main NumPy
    broadcast/sort rank buffer; query/output and library arrays are additional.
    NumPy defaults to sortsearch; Numba defaults to scan. No global sys.path
    mutation or implicit ANN/distance/neighbor/tie substitution occurs.
    """
    if backend not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}')
    for name, value in [('threads', threads), ('block_rows', block_rows),
                        ('max_scratch_bytes', max_scratch_bytes)]:
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    X, Y = _validate_pair(X, Y)
    kvals = _ks(ks, len(X))
    itemsize = max(4 if str(Z.dtype) == 'float32' else 8 for Z in (X, Y))
    distance_bytes = len(X) * len(X) * itemsize
    if max_distance_bytes is not None:
        if not isinstance(max_distance_bytes, Integral) or max_distance_bytes < 1:
            raise ValueError('max_distance_bytes must be a positive integer or None')
        if distance_bytes > max_distance_bytes:
            raise MemoryError(f'One full distance matrix needs {distance_bytes} bytes; '
                              f'max_distance_bytes={max_distance_bytes}')
    method = rank_method or ('sortsearch' if backend in ('numpy', 'sqrt_numpy') else 'scan')
    effective_block = int(block_rows)
    if backend in ('numpy', 'sqrt_numpy') and method == 'sortsearch':
        row_bytes = len(X) * itemsize
        if row_bytes > max_scratch_bytes:
            raise MemoryError('One sortsearch row exceeds max_scratch_bytes')
        effective_block = min(effective_block, max_scratch_bytes // row_bytes)
    old_numba_threads = None
    if backend in ('numba', 'sqrt_numba'):
        try:
            from numba import get_num_threads, set_num_threads
        except ImportError as error:
            raise ImportError(f"backend='{backend}' requires the optional numba dependency") from error
        old_numba_threads = get_num_threads()
        set_num_threads(int(threads))
    try:
        with threadpool_limits(limits=int(threads)):
            if backend == 'numpy':
                from ._numpy import joint_sklearn_numpy
                result = joint_sklearn_numpy(X, Y, kvals, rank_method=method,
                    block_size=effective_block, max_scratch_bytes=max_scratch_bytes)
            elif backend == 'numba':
                from ._strict_numba import joint_strict_numba
                result = joint_strict_numba(X, Y, kvals, rank_method=method, block_rows=effective_block)
            elif backend == 'sqrt_numpy':
                from ._sqrt_numpy import joint_sklearn_sqrt_numpy
                result = joint_sklearn_sqrt_numpy(X, Y, kvals, rank_method=method,
                    block_size=effective_block, max_scratch_bytes=max_scratch_bytes)
            else:
                from ._strict_numba import joint_strict_numba
                result = joint_strict_numba(X, Y, kvals, rank_method=method,
                    block_rows=effective_block, sqrt_elision=True)
    finally:
        if old_numba_threads is not None:
            set_num_threads(old_numba_threads)
    return [Quality(q.k, q.trustworthiness, q.continuity,
                    q.trustworthiness_penalty, q.continuity_penalty) for q in result]


__all__ = ['Quality', 'BACKENDS', 'joint_quality']
