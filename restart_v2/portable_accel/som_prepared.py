"""Explicit immutable same-X full-SVD preparation for repeated SOM fits.

This is an amortized workload option, not a faster cold first fit. Preparation
owns a read-only X snapshot and performs the same full SVD once. Every fit starts
from fresh W0/P0 and calls the frozen public run_som_olp; there is no warm start.
"""
import hashlib
import operator
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from .policy import ExecutionPolicy
from .som_olp import run_som_olp


def _immutable(array):
    array=np.ascontiguousarray(array)
    # A bytes-backed view cannot be made writeable again through NumPy flags.
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


class PreparedSOM:
    """Owned immutable snapshot and a bounded leading-coordinate SVD cache.

    max_rank bounds supported min(grid_dimension,data_dimension). The entire
    SVD is computed; retaining only needed factors does not replace it with a
    truncated/randomized solver. Private arrays are not public mutation APIs.

    Factorization uses float64 mean-centered data and the full NumPy SVD.
    Its factors stay pinned to preparation_threads; changing fit threads does
    not recompute them under a new reduction order. Construct another object
    to change the input or preparation contract. initialize() produces new
    W0/P0 for each grid, lambda and pca_scale; gamma affects only the fit.
    """
    __slots__=('_X','_mean','_projection','_basis','_singular','_max_xnorm','_leading_std',
               '_rank','_preparation_threads','_input_sha256','_owned_bytes','_preparation_seconds',
               '_source_shape','_source_dtype','_source_sha256','_sealed')

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):
            raise AttributeError('PreparedSOM is immutable; construct a new snapshot')
        object.__setattr__(self,name,value)

    def __init__(self,X,*,max_rank=2,threads=1):
        started=time.perf_counter()
        self._sealed=False
        supplied=np.asarray(X)
        self._source_shape=tuple(supplied.shape)
        self._source_dtype=supplied.dtype.str
        if supplied.dtype.kind not in 'fiu':raise TypeError('X must be real numeric data')
        if isinstance(max_rank,(bool,np.bool_)) or isinstance(threads,(bool,np.bool_)):
            raise ValueError('integer rank and threads required, not booleans')
        max_rank=operator.index(max_rank);threads=operator.index(threads)
        if max_rank<1 or threads<1:raise ValueError('positive max_rank and threads required')
        values=np.ascontiguousarray(supplied,dtype=np.float64)
        if values.ndim!=2 or min(values.shape)<1:raise ValueError('nonempty two-dimensional X required')
        if not np.all(np.isfinite(values)):raise ValueError('X must be finite')
        values=_immutable(values)
        n,d=values.shape;rank=min(max_rank,n,d)
        with threadpool_limits(threads,user_api='blas'):
            mean=values.mean(axis=0,keepdims=True)
            centered=values-mean
            U,s,Vt=np.linalg.svd(centered,full_matrices=False)
            projection=np.ascontiguousarray(U[:,:rank]*s[:rank])
            basis=Vt[:rank].copy(order='C')
            singular=s[:rank].copy()
            max_xnorm=float(np.max(np.einsum('ij,ij->i',centered,centered)))
            leading_std=float(s[0]/np.sqrt(n))
        self._X=values;self._mean=_immutable(mean);self._projection=_immutable(projection)
        self._basis=_immutable(basis);self._singular=_immutable(singular)
        for array in (self._X,self._mean,self._projection,self._basis,self._singular):
            array.setflags(write=False)
        self._max_xnorm=max_xnorm;self._leading_std=leading_std
        self._rank=rank;self._preparation_threads=threads
        self._input_sha256=hashlib.sha256(memoryview(values).cast('B')).hexdigest()
        self._source_sha256=(self._input_sha256 if supplied.dtype==np.dtype(np.float64)
                             else hashlib.sha256(memoryview(np.ascontiguousarray(supplied)).cast('B')).hexdigest())
        self._owned_bytes=sum(a.nbytes for a in (values,mean,projection,basis,singular))
        self._preparation_seconds=time.perf_counter()-started
        self._sealed=True

    def describe(self):
        return dict(cache_contract='owned-float64-mean-centered-full-numpy-svd; guarded-lowrank-P0-or-direct', shape=list(self._X.shape),dtype='float64',source_dtype=self._source_dtype,rank_capacity=self._rank,
                    full_svd=True,preparation_threads=self._preparation_threads,
                    input_sha256=self._input_sha256,source_input_sha256=self._source_sha256,owned_snapshot_and_factor_bytes=self._owned_bytes,
                    preparation_seconds=self._preparation_seconds,
                    memory_scope='steady owned arrays only; excludes original caller X, SVD peak workspace, fit outputs, kernel centering and validation')

    def assert_same_input(self,X):
        """Optional explicit identity check; its scan cost belongs to the caller.

        fit() itself has no X argument and always uses the owned snapshot.
        Shape/dtype/value changes require a fresh PreparedSOM instance.
        """
        supplied=np.asarray(X)
        if tuple(supplied.shape)!=self._source_shape:
            raise ValueError('input shape differs from the prepared source')
        if supplied.dtype.str!=self._source_dtype:
            raise ValueError('input dtype differs from the prepared source')
        values=np.ascontiguousarray(supplied)
        if hashlib.sha256(memoryview(values).cast('B')).hexdigest()!=self._source_sha256:
            raise ValueError('input values differ from the prepared source')

    def initialize(self,R,lam,*,pca_scale=2.,threads=None):
        threads=self._preparation_threads if threads is None else operator.index(threads)
        supplied=np.asarray(R)
        if supplied.dtype.kind not in 'fiu':raise TypeError('R must be real numeric data')
        R=np.ascontiguousarray(supplied,dtype=np.float64)
        if R.ndim!=2 or min(R.shape)<1 or not np.all(np.isfinite(R)):
            raise ValueError('finite nonempty two-dimensional R required')
        if not np.isfinite(lam) or lam<=0 or not np.isfinite(pca_scale) or threads<1:
            raise ValueError('positive finite lam, finite pca_scale and positive threads required')
        n,d=self._X.shape;k=min(R.shape[1],d)
        if k>self._rank:
            raise ValueError('grid rank exceeds prepared capacity; prepare again with a larger max_rank')
        with threadpool_limits(threads,user_api='blas'):
            Rc=R-R.mean(axis=0,keepdims=True)
            Rc/=np.maximum(np.abs(Rc).max(axis=0,keepdims=True),1e-12)
            scale=pca_scale*(self._singular[:k]/np.sqrt(n))[None,:]
            coordinates=Rc[:,:k]*scale
            W=self._mean+coordinates@self._basis[:k]
            cn=np.einsum('ij,ij->i',coordinates,coordinates)
            eps=np.finfo(np.float64).eps
            bound=64*eps*(d+2)*(self._max_xnorm+float(cn.max()))
            translated=np.max(np.abs(self._mean))>eps**(-.25)*max(1.,self._leading_std)
            if not translated and np.isfinite(bound) and bound<=lam*1e-8:
                scores=(-2.0*(self._projection[:,:k]@coordinates.T)+cn[None,:])/-lam
                P=softmax(scores,axis=1)
            else:
                P=softmax(-cdist(self._X,W,'sqeuclidean')/lam,axis=1)
        return np.ascontiguousarray(W),np.ascontiguousarray(P)

    def fit(self,R,*,gamma,lam,max_iters=100,tol=1e-4,pca_scale=2.,policy=None):
        if policy is None:policy=ExecutionPolicy(threads=self._preparation_threads)
        if not isinstance(policy,ExecutionPolicy) or policy.threads is None:
            raise ValueError('an ExecutionPolicy with explicit threads is required')
        # A fit also snapshots the small grid so caller mutation cannot change
        # its initializer/kernel boundary. No input state is carried between fits.
        supplied_grid=np.asarray(R)
        if supplied_grid.dtype.kind not in 'fiu':raise TypeError('R must be real numeric data')
        grid=np.array(supplied_grid,dtype=np.float64,order='C',copy=True)
        W0,P0=self.initialize(grid,lam,pca_scale=pca_scale,threads=policy.threads)
        result=run_som_olp(self._X,grid,W0,P0,gamma=gamma,lam=lam,max_iters=max_iters,
                          tol=tol,backend='threadpool',policy=policy)
        result['initialization_reuse']='same owned X and same full-SVD factors; fresh W0/P0'
        result['preparation_included_in_this_call']=False
        return result
