import numpy as np
import pytest
from ubukit._impl.rmcm._graph import _gram_candidates,_radius
from ubukit._impl.rmcm import prepare_rmcm
from .test_graph_parity import base

@pytest.mark.parametrize('d',[1,2,3,7,8,9,16,32,64,128])
@pytest.mark.parametrize('scale',[1e-140,1e-50,1.,1e50,1e140])
@pytest.mark.parametrize('offset',[0.,2.**20,2.**45])
def test_entire_candidate_band_survives(d,scale,offset):
    rng=np.random.default_rng(d);eps=np.finfo(float).eps
    # The first row is intentionally far from the near-neighbor rows. Shifted
    # norms can differ by many orders of magnitude and Gram terms can cancel.
    direction=rng.normal(size=d);direction/=np.linalg.norm(direction)
    rows=[np.full(d,offset),np.full(d,offset)+direction]
    for fraction in [-.99,-.5,0.,.5,.99]:
        rows.append(np.full(d,offset)+direction*(1+fraction*64*eps*d))
    rows += [np.full(d,offset)+direction*1e-6, np.full(d,offset)-direction*1e6]
    X=np.asarray(rows)*scale
    for anchor in [0,len(X)-1]:
        Z=X-X[anchor];norms=np.einsum('ij,ij->i',Z,Z,optimize=False)
        diff=X[1]-X[0];radius=np.sqrt(np.einsum('i,i->',diff,diff))
        if radius==0:continue
        upper=radius*(1+64*eps*d);lower=radius*(1-64*eps*d)
        mask=_gram_candidates(Z,Z,norms,norms,upper*upper,512*eps*d,d)
        direct=np.empty((len(X),len(X)))
        for i in range(len(X)):
            D=X-X[i];direct[i]=np.sqrt(np.einsum('ij,ij->i',D,D,optimize=False))
        band=(direct>=lower)&(direct<=upper)
        assert np.any(band)
        assert np.all(mask[band])

@pytest.mark.parametrize('d',[8,16,32,64,128])
@pytest.mark.parametrize('seed',range(10))
def test_auto_full_outputs_and_materialized_count(d,seed):
    rng=np.random.default_rng(seed);X=rng.normal(size=(69,d));delta=np.sqrt(d)
    a=base.prepare_rmcm(X,delta);b=prepare_rmcm(X,delta,graph_backend='auto')
    assert a.candidate_edges==b.candidate_edges
    np.testing.assert_array_equal(a.neighborhood_matrix().toarray(),b.neighborhood_matrix().toarray())
    aa=a.fit(4,init=X[[0,4,20,40]],max_iter=6);bb=b.fit(4,init=X[[0,4,20,40]],max_iter=6)
    for key in vars(aa):np.testing.assert_equal(getattr(aa,key),getattr(bb,key))

def test_auto_dispatch_and_unaudited_scipy_fallback(monkeypatch):
    import ubukit._impl.rmcm._graph as graph
    import scipy
    original=graph.graph_blocked_strict;calls=[]
    def record(*args):calls.append(args[0].shape);return original(*args)
    monkeypatch.setattr(graph,'graph_blocked_strict',record)
    for n,d,expected in [(63,8,False),(64,7,False),(64,8,True),(70,128,True)]:
        calls.clear();X=np.random.default_rng(8).normal(size=(n,d));prepare_rmcm(X,1.,graph_backend='auto');assert bool(calls)==expected
    monkeypatch.setattr(scipy,'__version__','1.16.99-unaudited')
    def prohibited(*args,**kwargs):raise AssertionError('unaudited BLAS path used')
    monkeypatch.setattr(graph,'_graph_blocked_impl',prohibited)
    prepare_rmcm(np.random.default_rng(4).normal(size=(65,8)),1.,graph_backend='auto')

@pytest.mark.parametrize('d',[8,16,32,64,128])
def test_auto_sparse_radius_hint_is_scale_translation_and_cluster_safe(d):
    from ubukit._impl.rmcm._graph import _prefer_sparse_tree
    rng=np.random.default_rng(d);X=rng.normal(size=(600,d))
    for scale,offset in [(1.,0.),(1e-100,0.),(1e100,1e103)]:
        data=X*scale+offset
        assert _prefer_sparse_tree(data,.01*scale)
        assert not _prefer_sparse_tree(data,np.sqrt(d)*scale)
        for delta in [.01*scale,np.sqrt(d)*scale]:
            a=base.prepare_rmcm(data,delta);b=prepare_rmcm(data,delta,graph_backend='auto')
            assert a.candidate_edges==b.candidate_edges
            np.testing.assert_array_equal(a.neighborhood_matrix().toarray(),b.neighborhood_matrix().toarray())
    clustered=X.copy();clustered[300:]+=1000
    assert not _prefer_sparse_tree(clustered,np.sqrt(d))
    outlier=X.copy();outlier[0]+=1000
    assert not _prefer_sparse_tree(outlier,np.sqrt(d))
