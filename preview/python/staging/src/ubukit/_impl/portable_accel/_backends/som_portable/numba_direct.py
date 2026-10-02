"""SOM-OLP direct SIMD experiment, same old-P -> V/W -> new-P return order."""
import operator,time
import numpy as np
from ..._threadpools import threadpool_context
from .oracle import check_inputs
from ..._som_numerics import repair_tiny_mass_prototypes, normalize_costs_inplace
from .simd_costs import costs_into
from ..kmeans_numba import _runtime_vector_cap

def run(X,R,W0,P0,gamma,lam,max_iters=100,tol=1e-4,threads=1,*,block_rows=256,width=None):
    max_iters=operator.index(max_iters);threads=operator.index(threads)
    check_inputs(X,R,W0,P0,gamma,lam,max_iters)
    if threads<1 or block_rows<1:raise ValueError('positive threads and block_rows required')
    n,d=X.shape;m,q=R.shape;block_rows=min(block_rows,n)
    width=_runtime_vector_cap() if width is None else width
    W=W0.copy()
    if max_iters==0:return dict(W=W,P=P0.copy(),V=None,history=np.empty(0),n_iter=0,
        variant='numba_direct_zero_iterations',setup_seconds=0.,direct_fallback_rows=0)
    started=time.perf_counter()
    with threadpool_context(limits=threads,user_api='blas'):
        P=P0;Pout=np.empty_like(P0);V=np.empty((n,q));numerator=np.empty_like(W)
        WT=np.empty((d,m));RT=np.ascontiguousarray(R.T)
        history=np.empty(max_iters);previous=None
        setup=time.perf_counter()-started
        for iteration in range(max_iters):
            np.matmul(P,R,out=V);den=P.sum(axis=0)
            np.matmul(P.T,X,out=numerator)
            np.divide(numerator,den[:,None],out=W,where=den[:,None]>0)
            repair_tiny_mass_prototypes(X,P,den,W)
            WT[:]=W.T
            costs_into(X,WT,V,RT,float(gamma),Pout,width)
            objective=0.
            for start in range(0,n,block_rows):
                Pb=Pout[start:min(n,start+block_rows)]
                minimum,total=normalize_costs_inplace(Pb,lam)
                objective+=float(np.sum(minimum-lam*np.log(total)))
            P=Pout;history[iteration]=objective
            if previous is not None and abs(objective-previous)/max(1.,abs(previous))<=tol:break
            previous=objective
    return dict(W=W,P=Pout,V=V,history=history[:iteration+1].copy(),n_iter=iteration+1,
                variant='numba_direct',setup_seconds=setup,direct_fallback_rows=0)
