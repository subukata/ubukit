"""New strict wrapper with rank-only blocking and optional lazy kernels."""
from .validation import matrix_input,positive_int
from .policy import policy_or_default
from ._optional import numba_available

def joint_quality(X,Y,ks=5,*,backend='auto',policy=None,max_distance_bytes=None,return_stats=False,rank_method=None):
    X,px=matrix_input(X);Y,py=matrix_input(Y,name='Y')
    if len(X)!=len(Y):raise ValueError('row counts must match')
    policy=policy_or_default(policy)
    if backend=='auto':backend='numba' if numba_available() else 'numpy'
    if backend=='numba_sqrt':backend='sqrt_numba'
    if backend not in ('numpy','numba','sqrt_numpy','sqrt_numba'):raise ValueError('unknown metrics backend')
    use_numba=backend in ('numba','sqrt_numba')
    if use_numba and not numba_available():raise ImportError('Numba is unavailable or disabled')
    from ._backends.metrics_portable._numpy import _ks,joint_sklearn_numpy
    kvals=_ks(ks,len(X))
    # sklearn preserves float32 only; other accepted real dtypes use float64.
    # PreparedData can retain explicit float16/integer dtypes, unlike raw input.
    distance_itemsize=max(4 if Z.dtype=='float32' else 8 for Z in (X,Y))
    distance_bytes=len(X)**2*distance_itemsize
    if max_distance_bytes is not None:
        positive_int(max_distance_bytes,'max_distance_bytes')
        if distance_bytes>max_distance_bytes:raise ValueError('strict full distance matrix exceeds cap')
    method=rank_method or ('scan' if use_numba else 'sortsearch' if backend=='sqrt_numpy' else 'broadcast')
    rows=policy.block_rows
    if method in ('sortsearch','full'):
        # Bound the main sort buffer. Rank/output/library work is additional.
        rows=policy.rows_for(len(X)*distance_itemsize,len(X))
    elif use_numba:
        # Conservative endpoint/query scratch budget; full distance is separate.
        rows=policy.rows_for(128*sum(kvals),len(X))
    if backend=='sqrt_numpy':from ._backends.metrics_portable._sqrt_numpy import joint_sklearn_sqrt_numpy
    if use_numba:from ._backends.metrics_portable._strict_numba import joint_strict_numba
    with policy.activate(numba=use_numba):
        if backend=='numpy':
            result=joint_sklearn_numpy(X,Y,kvals,rank_method=method,block_size=rows,max_scratch_bytes=policy.max_scratch_bytes)
        elif backend=='sqrt_numpy':
            result=joint_sklearn_sqrt_numpy(X,Y,kvals,rank_method=method,block_size=rows,max_scratch_bytes=policy.max_scratch_bytes)
        else:
            result=joint_strict_numba(X,Y,kvals,rank_method=method,block_rows=rows,sqrt_elision=backend=='sqrt_numba')
    stats={'backend':backend,'rank_method':method,'rank_block_rows':rows,'largest_distance_bytes':distance_bytes,'strict_full_distance':True,'prepared_input':(px is not None,py is not None),'neighbor_or_distance_cache_used':False}
    return (result,stats) if return_stats else result
