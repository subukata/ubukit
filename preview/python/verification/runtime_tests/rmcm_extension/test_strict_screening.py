from .test_graph_parity import base,compare
from ubukit_rmcm._graph import graph_blocked_strict
from ubukit_rmcm import prepare_rmcm
import numpy as np
import pytest

@pytest.mark.parametrize('d',[1,2,3,4,7,8,9,16,31,32,64,127])
@pytest.mark.parametrize('seed',range(25))
def test_exact_candidate_count_at_inflated_boundary(d,seed):
    rng=np.random.default_rng(seed);X=rng.normal(size=(23,d))
    distance=np.sqrt(np.einsum('i,i->',X[0]-X[1],X[0]-X[1]))
    center=distance/(1+8*np.finfo(float).eps*d)
    for delta in [np.nextafter(center,0),center,np.nextafter(center,np.inf),distance]:
        old=base.prepare_rmcm(X,delta);new=prepare_rmcm(X,delta,graph_backend='blocked-strict')
        assert old.candidate_edges==new.candidate_edges
        for limit in [max(1,old.candidate_edges-1),old.candidate_edges,old.candidate_edges+1]:
            if limit<old.candidate_edges:
                with pytest.raises(MemoryError):prepare_rmcm(X,delta,max_edges=limit,graph_backend='blocked-strict')
            else:
                q=prepare_rmcm(X,delta,max_edges=limit,graph_backend='blocked-strict')
                np.testing.assert_array_equal(q.neighborhood_matrix().toarray(),old.neighborhood_matrix().toarray())

@pytest.mark.parametrize('d',[1,8,32])
@pytest.mark.parametrize('exponent',[-150,-50,0,50,140])
@pytest.mark.parametrize('offset_power',[0,30,50])
def test_offsets_scales_and_candidate_count(d,exponent,offset_power):
    rng=np.random.default_rng(d+exponent+150);X=(rng.normal(size=(31,d))+2.**offset_power)*10.**exponent
    distance=np.sqrt(np.einsum('i,i->',X[0]-X[1],X[0]-X[1]));delta=distance/(1+8*np.finfo(float).eps*d)
    if np.max(np.abs(X)) > np.sqrt(np.finfo(float).max / d) / 4:
        for func in [base.prepare_rmcm, lambda X, delta: prepare_rmcm(X, delta, graph_backend='blocked-strict')]:
            with pytest.raises(FloatingPointError): func(X, delta)
        return
    a=base.prepare_rmcm(X,delta);b=prepare_rmcm(X,delta,graph_backend='blocked-strict')
    assert a.candidate_edges==b.candidate_edges
    np.testing.assert_array_equal(a.neighborhood_matrix().toarray(),b.neighborhood_matrix().toarray())

@pytest.mark.parametrize('max_edges',[3,5,8,9])
def test_memory_error_precedes_underflow(max_edges):
    X=np.array([[0.],[1e-200],[2e-200]])
    for func in [base.prepare_rmcm,lambda X,delta,max_edges:prepare_rmcm(X,delta,max_edges=max_edges,graph_backend='blocked-strict')]:
        with pytest.raises(MemoryError if max_edges<9 else FloatingPointError):func(X,1.,max_edges=max_edges)
