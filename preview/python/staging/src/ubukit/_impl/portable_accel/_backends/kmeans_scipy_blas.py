"""Explicit JIT-free Lloyd using certified BLAS intervals or direct SciPy.

The provider/shape gate admits only the separately documented NumPy-linked
OpenBLAS route. Every uncertified call, ambiguous label, or unbounded direct
range uses ordered direct squared differences. Default selection is unchanged.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .blas_certificate import _blas_certificate_supported


def _distance_bounds(X, C, xn, cn, scores):
    """Outward intervals for the existing ordered direct Float64 distances."""
    d=X.shape[1];epsd=np.finfo(np.float64).eps*(d+2)
    factor=32.*epsd/(1.-epsd)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        np.matmul(X,C.T,out=scores)
        scores*=-2.;scores+=xn[:,None];scores+=cn[None,:]
        error=factor*(xn[:,None]+cn[None,:])+64.*(d+2)*np.finfo(np.float64).tiny
        lower=np.maximum(0.,np.nextafter(scores-error,-np.inf))
        upper=np.nextafter(scores+error,np.inf)
    return lower,upper


def run(X, init, max_iter, policy):
    n,d=X.shape;k=len(init);centers=init.copy()
    def direct(reason):
        nonlocal centers
        centers=None  # Release the unused output copy before direct fallback allocates its result.
        from .._kmeans_lagged import _scipy_lloyd
        result=_scipy_lloyd(X,init,max_iter,policy)
        result.update(guard_fallbacks=int(n*result['n_iter']),score_selector=reason,
                      certificate_enabled=False,blas_attempted_rows=0,blas_certified_rows=0)
        return result
    maximum_rows=min(policy.block_rows,n)
    epsd=np.finfo(np.float64).eps*(d+2)
    if epsd>=1/64 or not _blas_certificate_supported(X[:maximum_rows],centers):
        return direct('direct-uncertified-provider-or-shape')
    base_fixed=16*(n+k)
    sparse_fixed=64*n+16*k*d+32*(k+1)
    # Includes conservative peaks for q, error, lower/upper and ufunc temporaries.
    per_row=80*k+128
    sparse=(d>=32 and n>=16 and policy.max_scratch_bytes>=base_fixed+sparse_fixed+per_row)
    fixed=base_fixed+(sparse_fixed if sparse else 0)
    if policy.max_scratch_bytes<fixed+per_row:return direct('direct-small-budget')
    rows=min(policy.block_rows,n,(policy.max_scratch_bytes-fixed)//per_row)
    if not _blas_certificate_supported(X[:rows],centers):return direct('direct-small-budget-shape')
    scores=np.empty((rows,k),dtype=np.float64);labels=np.full(n,-1,dtype=np.int64)
    with np.errstate(over='ignore',invalid='ignore'):xnorm=np.einsum('ij,ij->i',X,X)
    allow_scores=bool(np.isfinite(xnorm).all())
    if sparse:
        from scipy.sparse import csc_matrix
        ones=np.ones(n,dtype=np.float64);pointers=np.arange(n+1,dtype=np.int64)
    fallbacks=0;attempted=0
    for iteration in range(1,max_iter+1):
        with np.errstate(over='ignore',invalid='ignore'):cnorm=np.einsum('ij,ij->i',centers,centers)
        use_scores=allow_scores and bool(np.isfinite(cnorm).all());changed=False
        for start in range(0,n,rows):
            end=min(n,start+rows);S=scores[:end-start];needs_direct=True
            if use_scores and _blas_certificate_supported(X[start:end],centers):
                attempted+=end-start
                lower,upper=_distance_bounds(X[start:end],centers,xnorm[start:end],cnorm,S)
                assigned=S.argmin(axis=1);positions=np.arange(end-start)
                best_upper=upper[positions,assigned]
                # Ranking alone is insufficient: the direct reference validates
                # every distance, including nonwinning centers, against overflow.
                finite_range=np.isfinite(upper).all(axis=1)
                lower[positions,assigned]=np.inf
                other_lower=lower.min(axis=1)
                safe=finite_range&(best_upper>=0)&(best_upper<other_lower)
                needs_direct=not bool(np.all(safe))
                if not bool(np.any(safe)):use_scores=False
            if needs_direct:
                cdist(X[start:end],centers,'sqeuclidean',out=S)
                if not np.isfinite(S).all():raise ValueError('squared distances overflowed; rescale input')
                assigned=S.argmin(axis=1);fallbacks+=end-start
            changed|=not np.array_equal(assigned,labels[start:end]);labels[start:end]=assigned
        counts=np.bincount(labels,minlength=k);active=counts>0
        if sparse:
            membership=csc_matrix((ones,labels,pointers),shape=(k,n)).tocsr();sums=membership@X
            np.divide(sums,counts[:,None],out=centers,where=active[:,None])
        else:
            for feature in range(d):
                sums=np.bincount(labels,weights=X[:,feature],minlength=k)
                np.divide(sums,counts,out=centers[:,feature],where=active)
        if not np.isfinite(centers).all():raise ValueError('center sums overflowed; rescale input')
        if not changed:break
    return {'centers':centers,'labels':labels,'n_iter':iteration,'scratch_rows':rows,
            'guard_fallbacks':fallbacks,'score_selector':'certified-interval-gap-direct-block-repair',
            'certificate_enabled':True,'blas_attempted_rows':attempted,
            'blas_certified_rows':n*iteration-fallbacks,
            'centroid_reducer':'scipy-csr-row-order' if sparse else 'numpy-bincount',
            'primary_scratch_budgeted_bytes':fixed+rows*per_row}
