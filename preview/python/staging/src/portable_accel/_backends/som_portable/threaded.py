"""Explicit standard-dependency outer-thread SOM experiment.

One ThreadPoolExecutor per call, single-thread BLAS scoped around the entire
executor. Statistics finish before any old P is overwritten. Row assignment
writes disjoint ranges; reductions follow fixed worker order. Not the default.
"""
from concurrent.futures import ThreadPoolExecutor
import operator
import time
import numpy as np
from scipy.spatial.distance import cdist
from .oracle import check_inputs
from ..._som_numerics import repair_tiny_mass_prototypes, normalize_costs_inplace
from .kernel import _controller,run as sequential_run,_stop


def run(X,R,W0,P0,gamma,lam,max_iters=100,tol=1e-4,threads=1,*,block_rows=256,distance='guarded'):
    max_iters=operator.index(max_iters);threads=operator.index(threads)
    block_rows=operator.index(block_rows)
    check_inputs(X,R,W0,P0,gamma,lam,max_iters)
    if threads<1 or block_rows<1:raise ValueError('positive threads and block_rows required')
    if distance not in ('guarded','direct','centered'):
        raise ValueError('distance must be guarded, direct or centered')
    if max_iters==0:
        out=sequential_run(X,R,W0,P0,gamma,lam,0,tol,threads,distance=distance)
        out['variant']='threaded_zero_iterations';return out
    n,d=X.shape;m,k=R.shape;active=min(threads,n)
    # Bound the mean by the coordinate range before bypassing the original
    # translation fallback. This preserves its variant/fallback metadata too.
    ordinary_coordinates = active == 1 and (distance != 'guarded' or (
        max(abs(float(X.min())), abs(float(X.max())),
            abs(float(R.min())), abs(float(R.max())))
        <= 0.5 * np.finfo(np.float64).eps ** (-0.25)))
    if active == 1 and ordinary_coordinates:
        # One worker has exactly the sequential reduction and row-block order.
        # Avoid creating an executor and two future barriers per iteration.
        out=sequential_run(X,R,W0,P0,gamma,lam,max_iters,tol,1,
                           distance=distance,block_rows=block_rows)
        out['variant']='threaded_numpy_'+distance
        out['effective_threads']=1;out['blas_threads']=1
        return out
    begin=time.perf_counter()
    with _controller.limit(limits=1,user_api='blas'):
        use_scores=distance!='direct'
        if use_scores:
            xorigin=X.mean(0);rorigin=R.mean(0)
            Xc=np.ascontiguousarray(X-xorigin);Rc=np.ascontiguousarray(R-rorigin)
            xn=np.einsum('ij,ij->i',Xc,Xc);rn=np.einsum('ij,ij->i',Rc,Rc)
            if distance=='guarded':
                limit=np.finfo(np.float64).eps**(-.25)
                translated=(np.max(np.abs(xorigin))>limit*max(1.,float(np.sqrt(np.mean(xn)/d)))
                            or np.max(np.abs(rorigin))>limit*max(1.,float(np.sqrt(np.mean(rn)/k))))
                if translated:
                    # Parallel numerator sums can magnify translation rounding.
                    # Explicit conservative fallback, with its cost included.
                    out=sequential_run(X,R,W0,P0,gamma,lam,max_iters,tol,1,distance='direct',block_rows=block_rows)
                    out['variant']='threaded_translation_single_thread_fallback'
                    out['effective_threads']=1
                    return out
        W=W0.copy();P=P0;Pout=np.empty_like(P0);V=np.empty((n,k))
        numerators=np.empty((active,m,d));denominators=np.empty((active,m))
        buffers=[np.empty((min(block_rows,n),m)) for _ in range(active)]
        ranges=[(n*t//active,n*(t+1)//active) for t in range(active)]
        history=[];previous=None;fallback_rows=0
        numerator=np.empty((m,d));den=np.empty(m)
        eps=np.finfo(np.float64).eps

        def statistics(t):
            start,end=ranges[t]
            old=P[start:end]
            np.matmul(old,R,out=V[start:end])
            np.sum(old,axis=0,out=denominators[t])
            np.matmul(old.T,X[start:end],out=numerators[t])

        def assignment(t):
            start,end=ranges[t];objective=0.;fallback=0
            for b in range(start,end,block_rows):
                e=min(b+block_rows,end);Pb=Pout[b:e];tmp=buffers[t][:e-b]
                if not use_scores:
                    cdist(X[b:e],W,'sqeuclidean',out=Pb)
                    if gamma:
                        cdist(V[b:e],R,'sqeuclidean',out=tmp);tmp*=gamma;Pb+=tmp
                    minimum,total=normalize_costs_inplace(Pb,lam);offset=0.
                else:
                    np.matmul(Xc[b:e],Wc.T,out=Pb);Pb*=-2.
                    if gamma:
                        np.matmul(Vc[b:e],Rc.T,out=tmp);tmp*=-2.*gamma;Pb+=tmp
                    Pb+=node_constant
                    minimum=Pb.min(1);offset=row_constant[b:e].copy()
                    if distance=='guarded':
                        bound=error_scale[b:e]
                        suspect=((minimum+offset<=bound)|(bound>lam*1e-8)|~np.isfinite(minimum+offset))
                        if np.any(suspect):
                            indices=np.flatnonzero(suspect)+b
                            costs=cdist(X[indices],W,'sqeuclidean')
                            if gamma:costs+=gamma*cdist(V[indices],R,'sqeuclidean')
                            Pb[suspect]=costs;minimum[suspect]=costs.min(1);offset[suspect]=0.
                            fallback+=int(np.count_nonzero(suspect))
                    Pb-=minimum[:,None];np.divide(Pb,-lam,out=Pb)
                if use_scores:
                    np.exp(Pb,out=Pb);total=Pb.sum(1);Pb/=total[:,None]
                objective+=float(np.sum(minimum+offset-lam*np.log(total)))
            return objective,fallback

        with ThreadPoolExecutor(max_workers=active,thread_name_prefix='som-numpy') as pool:
            setup=time.perf_counter()-begin
            for iteration in range(max_iters):
                # Barrier 1: all old-P products complete before Pout is reused.
                list(pool.map(statistics,range(active)))
                numerator[:]=numerators[0];den[:]=denominators[0]
                for t in range(1,active):
                    numerator+=numerators[t];den+=denominators[t]
                np.divide(numerator,den[:,None],out=W,where=den[:,None]>0)
                repair_tiny_mass_prototypes(X,P,den,W)
                if use_scores:
                    Wc=W-xorigin;Vc=V-rorigin
                    wn=np.einsum('ij,ij->i',Wc,Wc);vn=np.einsum('ij,ij->i',Vc,Vc)
                    node_constant=wn+gamma*rn;row_constant=xn+gamma*vn
                    if distance=='guarded':
                        error_scale=32.*eps*((d+2)*(xn+float(wn.max()))+gamma*(k+2)*(vn+float(rn.max())))
                # Barrier 2: finish disjoint row updates before next statistics.
                pieces=list(pool.map(assignment,range(active)))
                obj=sum(value for value,_ in pieces)
                fallback_rows+=sum(count for _,count in pieces)
                P=Pout;history.append(obj)
                if _stop(obj,previous,tol):break
                previous=obj
        return dict(W=W,P=Pout,V=V,history=np.asarray(history),n_iter=len(history),
                    variant='threaded_numpy_'+distance,setup_seconds=setup,
                    effective_threads=active,blas_threads=1,direct_fallback_rows=fallback_rows)
