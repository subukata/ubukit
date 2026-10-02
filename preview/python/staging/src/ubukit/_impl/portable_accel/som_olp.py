"""New validated API with recovered reference and verified portable candidates."""
import numpy as np
from .validation import matrix_input,matrix,positive_int
from .policy import policy_or_default

def _options(gamma,lam,max_iters,tol):
    if not np.isfinite(gamma) or gamma<0 or not np.isfinite(lam) or lam<=0:raise ValueError('require finite gamma>=0 and lam>0')
    positive_int(max_iters,'max_iters',allow_zero=True)
    if not np.isfinite(tol) or tol<0:raise ValueError('tol must be finite and nonnegative')

def initialize_som_olp(X,R,lam,*,pca_scale=2.,policy=None,initializer='original'):
    X,_=matrix_input(X,dtype=np.float64);R=matrix(R,dtype=np.float64,name='R')
    _options(0.,lam,0,0.)
    if not np.isfinite(pca_scale):raise ValueError('pca_scale must be finite')
    if min(R.shape[1],X.shape[1])>min(X.shape):raise ValueError('too few samples for PCA grid rank')
    policy=policy_or_default(policy)
    if initializer not in ('original','svd_lowrank'):raise ValueError('initializer must be original or svd_lowrank')
    if initializer=='svd_lowrank' and policy.threads is None:raise ValueError('svd_lowrank currently requires an explicit thread count')
    from ._som_extreme import needs_extreme
    if needs_extreme(X,R,pca_scale=pca_scale):
        from ._som_extreme import initialize
        with policy.activate():return initialize(X,R,lam,pca_scale)
    if initializer=='original':
        from ._backends.som_reference import initialize
        with policy.activate():return initialize(X,R,lam,pca_scale=pca_scale,threads=policy.threads)
    if initializer!='svd_lowrank':raise ValueError('initializer must be original or svd_lowrank')
    if policy.threads is None:raise ValueError('svd_lowrank currently requires an explicit thread count')
    from ._backends.som_portable import initialize
    with policy.activate():return initialize(X,R,lam,pca_scale,policy.threads,method='same_svd_lowrank')

def _run_som_olp_ordinary(X,R,W0,P0,*,gamma,lam,max_iters=100,tol=1e-4,backend='cdist',policy=None):
    if backend in ('cdist','numpy','auto'):
        from ._backends.som_reference import run
        with policy.activate():result=run(X,R,W0,P0,gamma,lam,max_iters=max_iters,tol=tol,threads=policy.threads)
        result['backend']='cdist';return result
    if backend=='numba_direct':
        from ._optional import numba_available
        if not numba_available():raise ImportError("Numba is unavailable or disabled; use backend='cdist_optimized'")
        if policy.threads is None:raise ValueError('numba_direct requires an explicit thread count')
        from ._backends.som_portable.numba_direct import run
        n,d=X.shape;m,q=R.shape
        # Transposed prototype/grid storage, numerator, denominator and a
        # conservative bound for per-row normalization temporaries. Outputs,
        # validated caller inputs and JIT/runtime workspace are not RSS-capped.
        fixed=8*(2*m*d+m*q+m)
        available=policy.max_scratch_bytes-fixed
        if available<48:raise ValueError('scratch cap cannot hold SIMD transposes and one normalization row')
        rows=min(n,policy.block_rows,available//48)
        with policy.activate():result=run(X,R,W0,P0,float(gamma),float(lam),max_iters,tol,
                                         policy.threads,block_rows=rows)
        result['backend']='numba_direct';result['scratch_rows']=rows
        result['primary_scratch_budgeted_bytes']=fixed+48*rows
        result['simd_threads']=1
        result['numerical_contract']='strict-float64-direct-increasing-feature-sums; log-sum-exp-objective'
        return result
    if backend=='threadpool':
        if policy.threads is None:raise ValueError('threadpool requires an explicit thread count')
        from ._backends.som_portable.threaded import run
        active=min(policy.threads,len(X));m=len(R);d=X.shape[1]
        # Private and merged numerator/denominator arrays plus row work buffers.
        fixed=8*(active+1)*(m*d+m)
        available=policy.max_scratch_bytes-fixed
        per_row=8*active*m
        if available<per_row:raise ValueError('scratch cap cannot hold ThreadPool private arrays and one row per worker')
        rows=min(policy.block_rows,len(X),available//per_row)
        with policy.activate():result=run(X,R,W0,P0,gamma,lam,max_iters,tol,policy.threads,block_rows=rows,distance='guarded')
        result['backend']='threadpool';result['scratch_rows']=rows
        result['primary_scratch_budgeted_bytes']=fixed+rows*per_row
        return result
    paths={'gemm_guarded':'guarded','gemm_centered':'centered','cdist_optimized':'direct'}
    if backend not in paths:raise ValueError('unknown SOM backend')
    if policy.threads is None:raise ValueError('portable SOM currently requires an explicit thread count')
    from ._backends.som_portable import run
    # Bound the primary B-by-M cost scratch, not output/SVD/fallback workspace.
    rows=policy.rows_for(8*len(R),len(X))
    with policy.activate():result=run(X,R,W0,P0,gamma,lam,max_iters,tol,policy.threads,distance=paths[backend],block_rows=rows)
    result['backend']=backend;result['scratch_rows']=rows;result['primary_scratch_bytes']=rows*len(R)*8
    return result

def run_som_olp(X,R,W0,P0,*,gamma,lam,max_iters=100,tol=1e-4,backend='cdist',policy=None):
    X,_=matrix_input(X,dtype=np.float64);R=matrix(R,dtype=np.float64,name='R');W0=matrix(W0,dtype=np.float64,name='W0');P0=matrix(P0,dtype=np.float64,name='P0')
    _options(gamma,lam,max_iters,tol);policy=policy_or_default(policy)
    from ._som_extreme import needs_extreme
    if max_iters==0 or not needs_extreme(X,R,W0,P0,gamma=gamma):
        return _run_som_olp_ordinary(X,R,W0,P0,gamma=gamma,lam=lam,max_iters=max_iters,tol=tol,backend=backend,policy=policy)
    # Keep backend/dependency/thread/shape/budget validation, including the
    # user's explicit optional-backend request, before selecting the fallback.
    checked=_run_som_olp_ordinary(X,R,W0,P0,gamma=gamma,lam=lam,max_iters=0,tol=tol,backend=backend,policy=policy)
    checked_backend=checked['backend'];del checked
    required=16*len(R)
    if policy.max_scratch_bytes<required:raise ValueError('scratch cap cannot hold an extreme SOM cost row')
    from ._som_extreme import run
    with policy.activate():result=run(X,R,W0,P0,gamma,lam,max_iters,tol)
    result['backend']=checked_backend;result['backend_requested']=backend
    result['primary_scratch_bytes']=required
    result['scratch_scope']='mantissa/exponent cost row; excludes Python reductions, outputs, validation and runtime workspace'
    return result

def fit_som_olp(X,R,*,gamma,lam,max_iters=100,tol=1e-4,pca_scale=2.,backend='cdist',policy=None,initializer='original'):
    W0,P0=initialize_som_olp(X,R,lam,pca_scale=pca_scale,policy=policy,initializer=initializer)
    result=run_som_olp(X,R,W0,P0,gamma=gamma,lam=lam,max_iters=max_iters,tol=tol,backend=backend,policy=policy)
    result['initializer_requested']=initializer
    return result
