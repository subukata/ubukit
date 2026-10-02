import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
from .._support import rmcm_oracle as base
from ubukit_rmcm import RMCMGraphCache

def assert_graph(a,b):
    pa,pb=a.neighborhood_matrix(),b.neighborhood_matrix()
    for x,y in [(pa.indptr,pb.indptr),(pa.indices,pb.indices),(pa.data,pb.data),(a.degrees,b.degrees)]:np.testing.assert_array_equal(x,y)
    assert a.candidate_edges==b.candidate_edges

@pytest.mark.parametrize('d',[1,2,3,8,16,32])
@pytest.mark.parametrize('seed',range(8))
def test_moving_frames_and_delta(d,seed):
    rng=np.random.default_rng(seed);X=rng.normal(size=(41,d));cache=RMCMGraphCache(.2)
    init=X[[0,10,20]]
    for frame,change in enumerate([0,.001,.002,.004,.008,.01,.2,2.]):
        current=X+rng.normal(scale=change,size=X.shape);delta=np.sqrt(d)*(1+(.01 if frame==3 else 0))
        a=cache.prepare(current,delta);b=base.prepare_rmcm(current,delta);assert_graph(a,b)
        ar=a.fit(3,init=init,max_iter=7,cycle_window=0);br=b.fit(3,init=init,max_iter=7,cycle_window=0)
        for key in vars(ar):np.testing.assert_equal(getattr(ar,key),getattr(br,key))
    assert cache.stats['rebuilds']>=2 and cache.stats['reuses']>=1

@pytest.mark.parametrize('d',[1,8,32])
def test_variable_n_reordering_snapshots_and_changed_radius(d):
    rng=np.random.default_rng(d);cache=RMCMGraphCache(.3)
    for n in [1,3,9,9,17,4,31,31]:
        X=rng.normal(size=(n,d));snapshot=X.copy()
        for delta in [0.,.5,1.,10.,.1]:
            a=cache.prepare(X,delta);b=base.prepare_rmcm(X,delta);assert_graph(a,b)
        X[:]=99
        np.testing.assert_array_equal(a._X,snapshot)
    before=cache.stats;cache.clear();assert cache.stats['candidate_edges']==0;assert cache.stats['rebuilds']==before['rebuilds']

@pytest.mark.parametrize('skin',[0.,.1,1.,100.])
def test_budget_fallback_and_tight_active_guard(skin):
    X=np.arange(20.)[:,None];cache=RMCMGraphCache(skin,max_cache_edges=50)
    for delta in [0.,.1,1.,3.]:
        for limit in [20,40,58,60,100,400]:
            try:b=base.prepare_rmcm(X,delta,max_edges=limit)
            except MemoryError:
                with pytest.raises(MemoryError):cache.prepare(X,delta,max_edges=limit)
            else:assert_graph(cache.prepare(X,delta,max_edges=limit),b)

@pytest.mark.parametrize('d',[1,8,32])
def test_adjacent_float_boundaries_inside_skin(d):
    rng=np.random.default_rng(d);X=rng.normal(size=(21,d));cache=RMCMGraphCache(.2)
    delta=np.sqrt(np.einsum('i,i->',X[0]-X[1],X[0]-X[1]))
    cache.prepare(X,delta)
    for scale in [1.,np.nextafter(1.,0),np.nextafter(1.,np.inf)]:
        current=X*scale
        for r in [np.nextafter(delta,0),delta,np.nextafter(delta,np.inf)]:
            assert_graph(cache.prepare(current,r),base.prepare_rmcm(current,r))

@pytest.mark.parametrize('value',[0.,1e-200,1e-160,1e-155,1e-153])
@pytest.mark.parametrize('delta',[0.,1e-300,1e-154,1.])
def test_underflow_and_zero_radius(value,delta):
    X=np.array([[0.],[value],[2*value]]);cache=RMCMGraphCache(.2)
    try:b=base.prepare_rmcm(X,delta)
    except FloatingPointError:
        with pytest.raises(FloatingPointError):cache.prepare(X,delta)
    else:assert_graph(cache.prepare(X,delta),b)

@pytest.mark.parametrize('invalid',[None,True,-1,np.nan,np.inf,'1'])
def test_invalid_skin(invalid):
    with pytest.raises(ValueError):RMCMGraphCache(invalid)

def test_serialization_and_materialized_diagnostics():
    import pickle
    from dataclasses import asdict
    X=np.random.default_rng(11).normal(size=(23,8));delta=2.;cache=RMCMGraphCache(.2)
    a=cache.prepare(X,delta);b=base.prepare_rmcm(X,delta)
    assert isinstance(a.candidate_edges, int)
    assert a.__dict__["candidate_edges"] == b.candidate_edges
    copied=pickle.loads(pickle.dumps(a))
    assert copied.candidate_edges==b.candidate_edges
    ar=asdict(copied.fit(3,init=X[[0,4,11]],max_iter=4));br=asdict(b.fit(3,init=X[[0,4,11]],max_iter=4))
    for key in ar:np.testing.assert_equal(ar[key],br[key])
    assert isinstance(a.candidate_edges,int)

@pytest.mark.parametrize('d',[8,16,32,64,128])
@pytest.mark.parametrize('seed',range(5))
def test_high_dimensional_blocked_cache_build_and_eager_count(d,seed):
    rng=np.random.default_rng(seed+100);X=rng.normal(size=(129,d));delta=np.sqrt(d);cache=RMCMGraphCache(.2*delta)
    for frame in range(5):
        current=X+rng.normal(scale=.0005*frame,size=X.shape)
        a=cache.prepare(current,delta*(1+.002*frame));b=base.prepare_rmcm(current,delta*(1+.002*frame))
        assert a.__dict__['candidate_edges']==b.candidate_edges
        assert_graph(a,b)
        aa=a.fit(3,init=X[[0,32,90]],max_iter=3);bb=b.fit(3,init=X[[0,32,90]],max_iter=3)
        for key in vars(aa):np.testing.assert_equal(getattr(aa,key),getattr(bb,key))
    assert cache.stats['rebuilds']==1 and cache.stats['reuses']==4

@pytest.mark.parametrize('max_cache_edges',[96,96*96-1,96*96])
@pytest.mark.parametrize('max_edges',[96,96*96-1,96*96])
def test_blocked_cache_builder_defers_underflow_until_original_guard(max_cache_edges,max_edges):
    X=np.zeros((96,8));X[1,0]=1e-200;cache=RMCMGraphCache(.2,max_cache_edges=max_cache_edges)
    expected=MemoryError if max_edges<96*96 else FloatingPointError
    with pytest.raises(expected):cache.prepare(X,1.,max_edges=max_edges)
    with pytest.raises(expected):base.prepare_rmcm(X,1.,max_edges=max_edges)

def test_overflowing_skin_or_target_falls_back_with_finite_stats():
    X=np.array([[0.,1.],[2.,3.],[4.,5.]])
    for skin,delta in [(np.finfo(float).max,1.),(1.,np.finfo(float).max)]:
        cache=RMCMGraphCache(skin);a=cache.prepare(X,delta);b=base.prepare_rmcm(X,delta);assert_graph(a,b)
        assert cache.stats['candidate_edges']==0 and cache.stats['cover_radius'] is None
