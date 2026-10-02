"""V2 fit includes final labels/inertia by default, with explicit finalizers.

The strict-direct Numba finalizer can differ from sklearn near floating ties
or large translations. Both finalizers are part of the timed public fit.
"""
import numpy as np
from .validation import matrix_input
from .policy import policy_or_default
from ._optional import numba_available

def fit_kmeans(X,init,*,max_iter=20,backend='auto',policy=None,finalize=True,finalizer='sklearn'):
    if not isinstance(finalize,(bool,np.bool_)):raise TypeError('finalize must be boolean')
    if finalizer not in ('sklearn','numba','scipy'):raise ValueError('finalizer must be sklearn, numba, or scipy')
    if finalize and finalizer=='numba' and not numba_available():raise ImportError('Numba finalizer is unavailable or disabled')
    from ._kmeans_lagged import fit_kmeans as core
    policy=policy_or_default(policy)
    result=core(X,init,max_iter=max_iter,backend=backend,policy=policy)
    result['core_labels']=result['labels'];result['finalized_labels']=bool(finalize)
    result['finalizer']=finalizer if finalize else None
    if not finalize:
        result['inertia']=None;result['label_contract']='lagged scratch assignment before final center update';return result
    x,_=matrix_input(X,dtype=np.float64)
    if finalizer=='numba':
        from ._backends.kmeans_finalizer import finalize as direct_finalize
        finished=direct_finalize(x,result['centers'],threads=policy.threads)
        result['labels']=finished['labels'];result['inertia']=finished['inertia'];result['label_contract']=finished['label_contract']
    elif finalizer=='scipy':
        from scipy.spatial.distance import cdist
        n=len(x);k=len(result['centers']);rows=policy.rows_for(8*k,n)
        scores=np.empty((rows,k),dtype=np.float64)
        labels=np.empty(n,dtype=np.int64);squared=np.empty(n,dtype=np.float64)
        with policy.activate():
            for start in range(0,n,rows):
                end=min(n,start+rows);block=scores[:end-start]
                cdist(x[start:end],result['centers'],'sqeuclidean',out=block)
                chosen=block.argmin(axis=1);labels[start:end]=chosen
                squared[start:end]=block[np.arange(end-start),chosen]
            if not np.isfinite(squared).all():raise ValueError('squared distances overflowed; rescale inputs')
            with np.errstate(over='ignore',invalid='ignore'):inertia=float(np.sum(squared,dtype=np.float64))
        if not np.isfinite(inertia):raise ValueError('inertia overflowed; rescale inputs')
        result['labels']=labels;result['inertia']=inertia
        result['label_contract']='nearest-final-centers; strict direct float64; first computed tie'
    else:
        from sklearn.metrics import pairwise_distances_argmin_min
        with policy.activate():
            labels,distances=pairwise_distances_argmin_min(x,result['centers'],metric='euclidean')
            with np.errstate(over='ignore',invalid='ignore'):inertia=float(np.dot(distances,distances))
        if not np.isfinite(inertia):raise ValueError('inertia overflow; rescale input')
        result['labels']=labels;result['inertia']=inertia
        result['label_contract']='nearest-final-centers; public sklearn argmin_min'
    return result
