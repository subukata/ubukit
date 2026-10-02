import numpy as np
import pytest
from ubukit_rmcm import RMCMGraphCache
from .test_graph_parity import base

def equal(a,b):
    assert a.candidate_edges==b.candidate_edges
    for x,y in [(a._P.indptr,b._P.indptr),(a._P.indices,b._P.indices),(a._P.data,b._P.data),(a.degrees,b.degrees)]:np.testing.assert_array_equal(x,y)

@pytest.mark.parametrize('d',[1,3,8,32,128])
@pytest.mark.parametrize('seed',range(6))
def test_point_ids_birth_death_reordering_and_eager_outputs(d,seed):
    rng=np.random.default_rng(seed+d);X=rng.normal(size=(67,d));ids=np.arange(67,dtype=np.int64)-100;next_id=100
    cache=RMCMGraphCache(.2*np.sqrt(d));delta=np.sqrt(d)
    for frame in range(8):
        if frame:
            keep=np.sort(rng.choice(len(X),len(X)-3,replace=False));X=X[keep]+rng.normal(scale=.0001,size=(len(keep),d));ids=ids[keep]
            births=1+(frame%4);X=np.vstack((X,rng.normal(size=(births,d))));ids=np.r_[ids,np.arange(next_id,next_id+births)];next_id+=births
            permutation=rng.permutation(len(X));X=X[permutation];ids=ids[permutation]
        current_delta=delta*(1.01 if frame%3 else 1.)
        a=cache.prepare(X,current_delta,point_ids=ids);b=base.prepare_rmcm(X,current_delta);equal(a,b)
        init=X[[0,5,10]];ar=a.fit(3,init=init,max_iter=4,cycle_window=0);br=b.fit(3,init=init,max_iter=4,cycle_window=0)
        for key in vars(ar):np.testing.assert_equal(getattr(ar,key),getattr(br,key))
    assert cache.stats['rebuilds']==1
    assert cache.stats['incremental_updates']==7
    assert cache.stats['inserted_points']>0 and cache.stats['removed_points']==21


def test_point_ids_validation_snapshot_and_mode_changes():
    X=np.arange(20.)[:,None];ids=np.arange(20);cache=RMCMGraphCache(1.)
    for bad in [np.arange(19),np.zeros(20,dtype=int),np.arange(20.)/2,np.ones(20,dtype=bool),np.full(20,2**64-1,dtype=np.uint64)]:
        with pytest.raises(ValueError):cache.prepare(X,1.,point_ids=bad)
    a=cache.prepare(X,1.,point_ids=ids);ids[:]=999
    np.testing.assert_array_equal(cache._reference_ids,np.arange(20))
    b=cache.prepare(X,1.);equal(a,b);assert cache._reference_ids is None
    c=cache.prepare(X,1.,point_ids=np.arange(20));equal(a,c)

@pytest.mark.parametrize('budget',[20,50,100,10000])
def test_point_ids_cache_budget_and_original_edge_guard(budget):
    X=np.arange(20.)[:,None];cache=RMCMGraphCache(10.,max_cache_edges=budget);ids=np.arange(20)
    cache.prepare(X,1.,point_ids=ids)
    X=np.r_[X[1:],[[7.1]]];ids=np.r_[ids[1:],100]
    expected=base.prepare_rmcm(X,1.);equal(cache.prepare(X,1.,point_ids=ids),expected)
    for limit in [expected.candidate_edges-1,expected.candidate_edges]:
        if limit<expected.candidate_edges:
            with pytest.raises(MemoryError):cache.prepare(X,1.,point_ids=ids,max_edges=limit)
        else:equal(cache.prepare(X,1.,point_ids=ids,max_edges=limit),expected)


def test_compact_pair_indices_and_old_prepared_snapshot_survive_rebuild():
    X=np.random.default_rng(14).normal(size=(101,8));cache=RMCMGraphCache(.2)
    p=cache.prepare(X,3.,point_ids=np.arange(len(X)));before=p.neighborhood_matrix()
    assert cache._pairs.dtype==np.int32 and not cache._pairs.flags.writeable
    assert cache.stats['cache_bytes']==cache._reference.nbytes+cache._pairs.nbytes+cache._reference_ids.nbytes
    cache.prepare(X+10.,3.,point_ids=np.arange(len(X)))
    assert cache._pairs.dtype==np.int32
    after=p.neighborhood_matrix()
    for x,y in [(before.indptr,after.indptr),(before.indices,after.indices),(before.data,after.data)]:np.testing.assert_array_equal(x,y)
