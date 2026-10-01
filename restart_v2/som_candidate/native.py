"""Explicit rebuilt native control, adapted from retained v2 source bodies.

The kernel has been rebuilt/retested for this challenge; old measurements are
not claimed. Guarded mode conservatively chooses direct distances when the raw
Gram roundoff scale is material relative to lambda. Not a bitwise guarantee.
"""
import operator
import numpy as np
from threadpoolctl import ThreadpoolController
from .oracle import check_inputs
from ._native_v2 import iterate_stream
_controller=ThreadpoolController()

def run(X,R,W0,P0,gamma,lam,max_iters=100,tol=1e-4,threads=1,*,block_rows=256,distance='guarded'):
    max_iters=operator.index(max_iters);threads=operator.index(threads);block_rows=operator.index(block_rows)
    check_inputs(X,R,W0,P0,gamma,lam,max_iters)
    if threads<1 or block_rows<1:raise ValueError('positive threads and block_rows required')
    if distance not in ('guarded','direct','aos','gemm'):
        raise ValueError('distance must be guarded, direct, aos or explicit raw gemm')
    if max(X.shape[0],X.shape[1],R.shape[0],R.shape[1])>2**31-1:
        raise ValueError('native BLAS dimensions exceed int32')
    if max_iters==0:
        return dict(W=W0.copy(),P=P0.copy(),V=None,history=np.empty(0),n_iter=0,variant='native_zero_iterations')
    selected=distance
    if distance=='guarded':
        xmax=float(np.max(np.einsum('ij,ij->i',X,X)))
        wmax=float(np.max(np.einsum('ij,ij->i',W0,W0)))
        bound=64.*np.finfo(np.float64).eps*(X.shape[1]+1)*(xmax+max(xmax,wmax))
        selected='gemm' if np.isfinite(bound) and bound<=lam*1e-8 else 'direct'
    mode={'direct':0,'aos':1,'gemm':2}[selected]
    with _controller.limit(limits=1,user_api='blas'):
        out=iterate_stream(X,R,W0,P0,gamma,lam,max_iters,tol,threads,mode,block_rows,1,15)
    out['variant']='rebuilt_native_'+selected
    out['requested_distance']=distance
    return out
