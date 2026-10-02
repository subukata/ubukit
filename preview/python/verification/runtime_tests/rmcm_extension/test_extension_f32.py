import numpy as np
import pytest
from ubukit._impl.rmcm._graph import _gram_candidates, _screening_coordinates, _audited_float32_blas
from ubukit._impl.rmcm import prepare_rmcm
from .test_graph_parity import base

@pytest.mark.parametrize('d',[8,9,16,32,64,128])
@pytest.mark.parametrize('scale',[1e-15,1e-7,1.,1e7,1e15])
@pytest.mark.parametrize('offset',[0.,2.**10,2.**30,2.**45])
def test_float32_entire_candidate_band_and_all_inner_pairs(d,scale,offset):
    rng=np.random.default_rng(d);eps=np.finfo(float).eps
    direction=rng.normal(size=d);direction/=np.linalg.norm(direction)
    X=np.array([np.full(d,offset)+direction*f for f in [0.,1.,1-63*eps*d,1+63*eps*d,.01,4.,-2.]])*scale
    diff=X-X[0];direct=np.sqrt(np.einsum('ij,ij->i',diff,diff,optimize=False));r=direct[1]
    if r==0:return
    upper=r*(1+64*eps*d)
    Z=(X-X[0]).astype(np.float32);norms=np.einsum('ij,ij->i',Z,Z,optimize=False)
    mask=_gram_candidates(Z,Z,norms,norms,upper*upper,float(128*np.finfo(np.float32).eps*d),d)
    for i in range(len(X)):
        v=X-X[i];distance=np.sqrt(np.einsum('ij,ij->i',v,v,optimize=False))
        assert np.all(mask[i,distance<=upper])
    state=_screening_coordinates(X,upper*upper)
    if _audited_float32_blas():assert state[0].dtype==np.float32

@pytest.mark.parametrize('d',[8,32,128])
@pytest.mark.parametrize('scale',[1e-150,1e-30,1e-18,1.,1e18,1e50,1e150])
def test_float32_dispatch_falls_back_without_changing_graph_or_guard(d,scale):
    X=np.random.default_rng(d).normal(size=(71,d))*scale;delta=np.sqrt(d)*scale
    a=base.prepare_rmcm(X,delta);b=prepare_rmcm(X,delta,graph_backend='blocked-strict')
    assert a.candidate_edges==b.candidate_edges
    for x,y in [(a._P.indptr,b._P.indptr),(a._P.indices,b._P.indices),(a._P.data,b._P.data)]:np.testing.assert_array_equal(x,y)
    for limit in [a.candidate_edges-1,a.candidate_edges,a.candidate_edges+1]:
        outcomes=[]
        for f,kw in [(base.prepare_rmcm,{}),(prepare_rmcm,dict(graph_backend='blocked-strict'))]:
            try:outcomes.append(f(X,delta,max_edges=limit,**kw).candidate_edges)
            except Exception as e:outcomes.append(type(e))
        assert outcomes[0]==outcomes[1]


def test_unaudited_float32_provider_and_high_dimension_fall_back(monkeypatch):
    import ubukit._impl.rmcm._graph as graph
    monkeypatch.setattr(graph,'_audited_float32_blas',lambda:False)
    X=np.random.default_rng(8).normal(size=(12,8))
    assert graph._screening_coordinates(X,8.)[0].dtype==np.float64
    monkeypatch.setattr(graph,'_audited_float32_blas',lambda:True)
    X=np.random.default_rng(1024).normal(size=(12,1024))
    assert graph._screening_coordinates(X,1024.)[0].dtype==np.float64

@pytest.mark.parametrize('shape,d', [((3,5),8),((17,23),32),((128,257),128),((3,5),655)])
@pytest.mark.parametrize('scale',[1e-15,1.,1e15])
def test_actual_float32_matmul_disjoint_boundary_tiles(shape,d,scale):
    calls=[]
    class Tracked(np.ndarray):
        def __matmul__(self, other):
            calls.append((self.dtype,other.dtype))
            return super().__matmul__(other)
    rng=np.random.default_rng(d);m,n=shape
    a=rng.normal(size=(m,d))*scale;b=rng.normal(size=(n,d))*scale
    A=a.astype(np.float32).view(Tracked);B=b.astype(np.float32).view(Tracked)
    qa=np.einsum('ij,ij->i',A,A,optimize=False);qb=np.einsum('ij,ij->i',B,B,optimize=False)
    q=a[0]-b[0];r=np.sqrt(np.einsum('i,i->',q,q));upper=r*(1+64*np.finfo(float).eps*d)
    mask=_gram_candidates(A,B,qa,qb,upper*upper,float(128*np.finfo(np.float32).eps*d),d)
    assert calls==[(np.dtype('float32'),np.dtype('float32'))]
    for i in range(m):
        diff=b-a[i];distance=np.sqrt(np.einsum('ij,ij->i',diff,diff,optimize=False))
        assert np.all(mask[i,distance<=upper])

@pytest.mark.parametrize('d',[8,32,128,655])
def test_large_graph_really_uses_float32_sgemm(monkeypatch,d):
    if not _audited_float32_blas():pytest.skip('Float32 BLAS provider not enabled here')
    import ubukit._impl.rmcm._graph as graph
    calls=[];original=graph._gram_candidates
    class Tracked(np.ndarray):
        def __matmul__(self,other):
            calls.append((self.dtype,other.dtype));return super().__matmul__(other)
    def spy(A,B,*args):return original(A.view(Tracked),B.view(Tracked),*args)
    monkeypatch.setattr(graph,'_gram_candidates',spy)
    X=np.random.default_rng(d).normal(size=(257,d));v=X[0]-X[1];delta=np.sqrt(np.einsum('i,i->',v,v))
    a=base.prepare_rmcm(X,delta);b=prepare_rmcm(X,delta,graph_backend='blocked-strict')
    assert (np.dtype('float32'),np.dtype('float32')) in calls
    assert a.candidate_edges==b.candidate_edges
    for x,y in [(a._P.indptr,b._P.indptr),(a._P.indices,b._P.indices),(a._P.data,b._P.data)]:np.testing.assert_array_equal(x,y)
