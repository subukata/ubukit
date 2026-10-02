import numpy as np
import pytest
from ubukit_rmcm import prepare_rmcm
from ubukit_rmcm._graph import _complete_graph_certificate
from .test_graph_parity import base

@pytest.mark.parametrize('d',[1,3,8,32,128])
@pytest.mark.parametrize('kind',['normal','zero','tiny_equal','spacing_boundary'])
def test_complete_certificate_materialized_graph_and_full_outputs(d,kind):
    rng=np.random.default_rng(d);X=rng.normal(size=(29,d));delta=100.
    if kind=='zero':X[:]=0.;delta=0.
    elif kind=='tiny_equal':X[:]=1e-300;delta=1.
    elif kind=='spacing_boundary':
        X[:]=0.;a=np.ldexp(1.,-458);X[:,0]=np.resize([0.,a,np.nextafter(a,np.inf),-a],len(X));delta=1.
    assert _complete_graph_certificate(X,delta)
    a=base.prepare_rmcm(X,delta)
    for backend in ['tree','auto','blocked-strict']:
        b=prepare_rmcm(X,delta,graph_backend=backend)
        assert a.candidate_edges==b.candidate_edges==len(X)**2
        for x,y in [(a._P.indptr,b._P.indptr),(a._P.indices,b._P.indices),(a._P.data,b._P.data)]:np.testing.assert_array_equal(x,y)
        try:
            ar=a.fit(3,init=X[[0,4,8]],max_iter=6)
        except FloatingPointError as error:
            # A rounded mean can differ from identical tiny coordinates; the
            # original fitting-stage underflow must still be preserved.
            with pytest.raises(type(error), match="underflow"):
                b.fit(3,init=X[[0,4,8]],max_iter=6)
        else:
            br=b.fit(3,init=X[[0,4,8]],max_iter=6)
            for key in vars(ar):np.testing.assert_equal(getattr(ar,key),getattr(br,key))
        with pytest.raises(MemoryError):prepare_rmcm(X,delta,max_edges=len(X)**2-1,graph_backend=backend)

@pytest.mark.parametrize('v',[1e-300,1e-160,np.nextafter(np.ldexp(1.,-458),0.)])
def test_unsafe_spacing_retains_original_numeric_and_edge_guards(v):
    X=np.zeros((67,8));X[:,0]=2.;X[2,0]=0.;X[3,0]=v
    assert not _complete_graph_certificate(X,100.)
    for limit in [len(X),len(X)**2]:
        results=[]
        for lib,kw in [(base,{}),(__import__('ubukit_rmcm'),dict(graph_backend='auto'))]:
            try:results.append(lib.prepare_rmcm(X,100.,max_edges=limit,**kw).n_edges)
            except Exception as e:results.append(type(e))
        assert results[0]==results[1]
