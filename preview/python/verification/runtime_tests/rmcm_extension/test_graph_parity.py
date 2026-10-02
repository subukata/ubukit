import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

from .._support import rmcm_oracle as base
from _ubukit_oracle_rmcm.core import _graph as graph_baseline
from ubukit._impl.rmcm._graph import graph_tree,graph_blocked
from ubukit._impl.rmcm import prepare_rmcm


def compare(X, delta, method, block=32):
    outcomes=[]
    for graph in [graph_baseline,method]:
        try:
            outcomes.append(graph(X,delta,len(X)**2,block))
        except Exception as exc:
            outcomes.append(exc)
    if isinstance(outcomes[0],Exception):
        assert type(outcomes[0]) is type(outcomes[1]),(X,delta,outcomes)
        return
    assert not isinstance(outcomes[1],Exception),(X,delta,outcomes[1])
    a,b=outcomes
    for aa,bb in [(a[0].indptr,b[0].indptr),(a[0].indices,b[0].indices),(a[0].data,b[0].data),(a[1],b[1])]:
        np.testing.assert_array_equal(aa,bb)

@pytest.mark.parametrize('method',[graph_tree,graph_blocked])
@pytest.mark.parametrize('d',[1,2,3,4,5,7,8,9,16,31,32,33,64,127])
@pytest.mark.parametrize('seed',range(12))
def test_random_boundaries(method,d,seed):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(19,d))
    delta=np.sqrt(np.einsum('i,i->',X[0]-X[1],X[0]-X[1]))
    for r in [np.nextafter(delta,0),delta,np.nextafter(delta,np.inf)]:
        compare(X,r,method,block=7)

@pytest.mark.parametrize('method',[graph_tree,graph_blocked])
@pytest.mark.parametrize('d',[1,3,8,32,65])
@pytest.mark.parametrize('exponent',[-150,-140,-50,0,50,140,150])
@pytest.mark.parametrize('offset_power',[0,10,30,50])
def test_scales_offsets_boundaries(method,d,exponent,offset_power):
    rng=np.random.default_rng(d+exponent+150)
    scale=10.**exponent
    X=(rng.normal(size=(17,d))+2.**offset_power)*scale
    if np.max(np.abs(X))>np.sqrt(np.finfo(float).max/d)/4:
        return
    diff=X[0]-X[1];delta=np.sqrt(np.einsum('i,i->',diff,diff))
    for r in [np.nextafter(delta,0),delta,np.nextafter(delta,np.inf)]:
        compare(X,r,method,block=5)

@pytest.mark.parametrize('method',[graph_tree,graph_blocked])
@pytest.mark.parametrize('value',[0.,1e-300,1e-200,1e-162,1e-160,1e-155,1e-154,1e-153])
@pytest.mark.parametrize('delta',[0.,1e-300,1e-162,1e-160,1e-154,1.])
def test_underflow_error_parity(method,value,delta):
    compare(np.array([[0.],[value],[2*value]]),delta,method,1)

@pytest.mark.parametrize('d',[1,8,32])
@pytest.mark.parametrize('seed',range(8))
def test_public_preparation_fit_exact(d,seed):
    rng=np.random.default_rng(seed);X=rng.normal(size=(87,d));delta=float(np.sqrt(d));C=X[[0,7,21,57]]
    for backend in ['numpy','csr','adjoint']:
        a=base.prepare_rmcm(X,delta,backend=backend);b=prepare_rmcm(X,delta,backend=backend)
        aa=a.fit(4,init=C,max_iter=9,cycle_window=0);bb=b.fit(4,init=C,max_iter=9,cycle_window=0)
        for key in vars(aa):
            np.testing.assert_equal(getattr(aa,key),getattr(bb,key))

@pytest.mark.parametrize('method',[graph_tree,graph_blocked])
def test_memory_limit_and_sparse_large_n(method):
    with pytest.raises(MemoryError):method(np.zeros((100,8)),1,1000,4)
    X=np.arange(8000.)[:,None]*np.ones((1,2))
    P,degrees,count=method(X,0.1,10000,16)
    assert P.nnz==8000;np.testing.assert_array_equal(degrees,np.ones(8000))
