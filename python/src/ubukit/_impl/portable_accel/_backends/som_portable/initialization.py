"""Original full SVD initialization and an explicit same-SVD low-rank option.

The low-rank path retains the same full SVD and W0 formula. Only the initial
probability-distance calculation changes, using the leading SVD coordinates.
It is an exact-real identity, not a truncated/randomized replacement SVD.
"""
import operator
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from .kernel import _controller
from ..._som_numerics import probabilities_from_costs


def initialize(X, R, lam, pca_scale=2.0, threads=1, *, method='direct'):
    X=np.asarray(X,dtype=np.float64,order='C')
    R=np.asarray(R,dtype=np.float64,order='C')
    threads=operator.index(threads)
    if X.ndim!=2 or R.ndim!=2 or min(*X.shape,*R.shape)<1:
        raise ValueError('nonempty two-dimensional X and R required')
    if not np.isfinite(lam) or lam<=0 or not np.isfinite(pca_scale) or threads<1:
        raise ValueError('positive finite lam, finite pca_scale and positive threads required')
    if method not in ('direct','same_svd_lowrank'):
        raise ValueError('method must be direct or same_svd_lowrank')
    k=min(R.shape[1],X.shape[1])
    if X.shape[0]<k:
        raise ValueError('original initializer requires at least min(grid_dim,data_dim) samples')
    with _controller.limit(limits=threads,user_api='blas'):
        mu=X.mean(axis=0,keepdims=True)
        Xc=X-mu
        U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
        Rc=R-R.mean(axis=0,keepdims=True)
        Rc/=np.maximum(np.abs(Rc).max(axis=0,keepdims=True),1e-12)
        scale=pca_scale*(s[:k]/np.sqrt(X.shape[0]))[None,:]
        coordinates=Rc[:,:k]*scale
        W=mu+coordinates@Vt[:k]
        lowrank=method=='same_svd_lowrank'
        if lowrank:
            xn=np.einsum('ij,ij->i',Xc,Xc)
            cn=np.einsum('ij,ij->i',coordinates,coordinates)
            eps=np.finfo(np.float64).eps
            bound=64*eps*(X.shape[1]+2)*(float(xn.max())+float(cn.max()))
            translated=np.max(np.abs(mu))>eps**(-0.25)*max(1.,float(s[0]/np.sqrt(X.shape[0])))
            lowrank=not translated and np.isfinite(bound) and bound<=lam*1e-8
        if lowrank:
            projection=U[:,:k]*s[:k]
            costs=-2.0*(projection@coordinates.T)+cn[None,:]
            P=probabilities_from_costs(costs,lam)
        else:
            P=probabilities_from_costs(cdist(X,W,'sqeuclidean'),lam)
        return np.ascontiguousarray(W),np.ascontiguousarray(P)


def initialize_lowrank(X,R,lam,pca_scale=2.0,threads=1):
    return initialize(X,R,lam,pca_scale,threads,method='same_svd_lowrank')


def fit(X,R,gamma,lam,max_iters=100,tol=1e-4,threads=1,*,pca_scale=2.0,
        initialization='direct',distance='guarded',block_rows=None):
    from .kernel import run
    X=np.asarray(X,dtype=np.float64,order='C')
    R=np.asarray(R,dtype=np.float64,order='C')
    W0,P0=initialize(X,R,lam,pca_scale,threads,method=initialization)
    out=run(X,R,W0,P0,gamma,lam,max_iters,tol,threads,distance=distance,block_rows=block_rows)
    out['initialization']=initialization
    return out
