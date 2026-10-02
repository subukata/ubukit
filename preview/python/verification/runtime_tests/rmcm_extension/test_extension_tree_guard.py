import numpy as np
import pytest
from scipy.spatial import cKDTree
from ubukit._impl.rmcm._graph import _bounded_tree_pairs,_radius,graph_tree
from .test_graph_parity import base

@pytest.mark.parametrize('d',[1,3,8,32,128])
@pytest.mark.parametrize('scale',[1e-140,1e-20,1.,1e20,1e140])
@pytest.mark.parametrize('seed',range(6))
def test_inward_pairs_recover_exact_graph_and_respect_original_guard(d,scale,seed):
    rng=np.random.default_rng(seed+d);X=rng.normal(size=(33,d))*scale
    if seed%2:X+=(2**20)*scale
    delta=float(np.sqrt(np.einsum('i,i->',X[0]-X[1],X[0]-X[1])))
    for delta in [np.nextafter(delta,0.),delta,np.nextafter(delta,np.inf)]:
        ref=base.prepare_rmcm(X,delta);count=ref.candidate_edges
        tree=cKDTree(X);radius=_radius(delta,d);counts=tree.query_ball_point(X,radius,workers=1,return_length=True)
        pairs=_bounded_tree_pairs(X,tree,radius,counts,max(count,1),7)
        if pairs is not None:
            # Recovery staging is bounded by one original directed candidate
            # count plus one upper triangle, without a full N-by-N allocation.
            assert len(pairs)<=int(counts.sum())
        for limit in [max(1,count-1),count,count+1]:
            results=[]
            for fn in [lambda:base.prepare_rmcm(X,delta,max_edges=limit),lambda:graph_tree(X,delta,limit,7)]:
                try:results.append(fn())
                except Exception as e:results.append(e)
            if isinstance(results[0],Exception):assert type(results[0]) is type(results[1]);continue
            a,b=results;assert b[2]==a.candidate_edges
            for x,y in [(a._P.indptr,b[0].indptr),(a._P.indices,b[0].indices),(a._P.data,b[0].data)]:np.testing.assert_array_equal(x,y)

@pytest.mark.parametrize('d',[1,8,32])
def test_many_boundary_rows_and_reversed_queries(d):
    X=np.zeros((41,d));X[:,0]=np.repeat(np.arange(21.),2)[:41]
    ref=base.prepare_rmcm(X,1.);got=graph_tree(X,1.,ref.candidate_edges,3)
    assert got[2]==ref.candidate_edges
    for x,y in [(ref._P.indptr,got[0].indptr),(ref._P.indices,got[0].indices),(ref._P.data,got[0].data)]:np.testing.assert_array_equal(x,y)


def test_unreviewed_tree_version_retains_old_guard(monkeypatch):
    import scipy
    monkeypatch.setattr(scipy,'__version__','1.16.99')
    X=np.arange(20.)[:,None];tree=cKDTree(X);counts=tree.query_ball_point(X,1.,return_length=True)
    assert _bounded_tree_pairs(X,tree,1.,counts,100,4) is None
