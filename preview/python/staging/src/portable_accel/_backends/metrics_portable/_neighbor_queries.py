"""Audited-provider experiment: exact fitted-index reuse and brute top-k sharing.

Brute sharing uses the same ArgKmin surrogate arithmetic as sklearn1.8,
including X-is-Y norm reuse and strategy='auto'. Explicit self-position and
boundary checks preserve training self-exclusion; strict neighbor gaps rule
out heap-size-dependent ties. KD-tree queries remain separate for each k.
Unsupported versions/providers/data/configurations retain separate queries.
No indexes, inputs or query results survive the enclosing metric call.
"""
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
import scipy
import sklearn
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_info

@lru_cache(maxsize=1)
def _audited_provider():
    if sys.platform!='linux' or (sklearn.__version__,scipy.__version__,np.__version__)!=('1.8.0','1.17.0','2.3.5'):return False
    matches=[x for x in threadpool_info() if Path(x.get('filepath','')).parent.name=='scipy.libs' and x.get('user_api')=='blas']
    return len(matches)==1 and all(matches[0].get(k)==v for k,v in {'internal_api':'openblas','version':'0.3.30','architecture':'SkylakeX','threading_layer':'pthreads'}.items())

def _original(Z,ks):
    return np.concatenate([NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False) for k in ks],axis=1)

def _shared_brute(model,ks):
    A=model._fit_X;n=len(A);lo=min(ks);hi=max(ks)
    if len(ks)<2 or n<64 or hi+2>n or A.dtype!=np.float64 or not A.flags.c_contiguous:return None
    if 40*n*(hi+2)>16*1024*1024:return None
    if model.effective_metric_!='euclidean' or model.effective_metric_params_ or not _audited_provider():return None
    try:
        from sklearn.metrics._pairwise_distances_reduction import ArgKmin
    except ImportError:
        return None
    if not ArgKmin.is_usable_for(A,A,model.effective_metric_):return None
    distances,indices=ArgKmin.compute(X=A,Y=A,k=hi+2,metric=model.effective_metric_,metric_kwargs=model.effective_metric_params_,strategy='auto',return_distance=True)
    if not np.isfinite(distances).all():return None
    own=indices==np.arange(n)[:,None]
    if not np.all(own.sum(axis=1)==1):return None
    positions=np.argmax(own,axis=1)
    if not np.all(positions<=lo):return None
    if not np.all(distances[:,lo]<distances[:,lo+1]):return None
    nonself_distances=distances[~own].reshape(n,hi+1)
    if not np.all(nonself_distances[:,:-1]<nonself_distances[:,1:]):return None
    nonself_indices=indices[~own].reshape(n,hi+1)
    return {k:nonself_indices[:,:k] for k in ks}

def neighbor_queries(Z,ks):
    if len(ks)<2 or sklearn.__version__!='1.8.0':return _original(Z,ks)
    groups={}
    for k in ks:groups.setdefault(bool(Z.shape[1]>15 or k>=len(Z)//2),[]).append(k)
    results={}
    for brute,kvals in groups.items():
        model=NearestNeighbors(n_neighbors=kvals[0]).fit(Z)
        if getattr(model,'_fit_method',None)!=('brute' if brute else 'kd_tree'):return _original(Z,ks)
        shared=_shared_brute(model,kvals) if brute else None
        if shared is not None:results.update(shared)
        else:
            for k in kvals:results[k]=model.kneighbors(n_neighbors=k,return_distance=False)
    return np.concatenate([results[k] for k in ks],axis=1)
