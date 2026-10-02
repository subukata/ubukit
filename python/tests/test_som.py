"""Independent scalar-oracle tests for the public, installed SOM API."""
import json
import math
import subprocess
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np
import pytest
import ubukit
from ubukit._impl.portable_accel._som_classic import _schedule, _mulberry32


def grid(width, height):
    return [[i % width, i // width] for i in range(width*height)]


def winner(x, W):
    return min(range(len(W)), key=lambda j: sum((float(a)-float(b))**2 for a,b in zip(x,W[j])))


def h(i,j,R,sigma):
    if sigma == 0:
        return float(i == j)
    sq = sum((a-b)**2 for a,b in zip(R[i],R[j]))
    return math.exp(-sq/(2*sigma*sigma))


def online_reference(X, W, R, sigmas, rates):
    W = np.array(W, dtype=float, copy=True)
    trace=[]
    for t,(sigma,rate) in enumerate(zip(sigmas,rates)):
        x=X[t%len(X)]
        b=winner(x,W)
        for j in range(len(W)):
            for f in range(W.shape[1]):
                W[j,f] += rate*h(j,b,R,sigma)*(x[f]-W[j,f])
        trace.append(W.copy())
    return trace


def batch_reference(X,W,R,sigmas):
    W=np.array(W,dtype=float,copy=True)
    trace=[]
    for sigma in sigmas:
        labels=[winner(x,W) for x in X]
        new=W.copy()
        for j in range(len(W)):
            weights=[h(j,b,R,sigma) for b in labels]
            den=sum(weights)
            if den:
                for f in range(W.shape[1]):
                    new[j,f]=math.fsum(float(x[f])*a for x,a in zip(X,weights))/den
        W=new
        trace.append(W.copy())
    return trace


X=np.array([[0.,0.,1.],[2.,.5,-1.],[.5,2.,1.5],[3.,3.,-.5],[1.,1.,0.]])
W=np.array([[0.,0.,0.],[3.,0.,1.],[0.,3.,-1.],[3.,3.,0.]])


@pytest.mark.parametrize('schedule',["linear","geometric"])
@pytest.mark.parametrize('batch',[False,True])
def test_all_updates_match_independent_reference(batch,schedule):
    make=ubukit.initialize_som_batch if batch else ubukit.initialize_som
    total=4 if batch else 15
    # Reference schedule is intentionally independent of production helper.
    def values(start,end):
        return [start+(end-start)*i/(total-1) if schedule=='linear' else start*(end/start)**(i/(total-1)) for i in range(total)]
    sigma=values(1.5,.2)
    rate=values(.7,.05)
    state=make(X,grid_shape=(2,2),initial_prototypes=W,max_iterations=total,
               sigma=1.5,sigma_end=.2,schedule=schedule,
               **({} if batch else dict(learning_rate=.7,learning_rate_end=.05)))
    reference=batch_reference(X,W,grid(2,2),sigma) if batch else online_reference(X,W,grid(2,2),sigma,rate)
    for index, expected in enumerate(reference):
        result=state.step()
        np.testing.assert_allclose(result['centers'],expected,atol=3e-14,rtol=3e-14)
        assert result['iterations']==index+1
        assert result['unit']==('epoch' if batch else 'sample')
        assert 'labels' not in result
    assert state.step() is None
    result=state.result()
    assert result['labels'].tolist()==[winner(x,result['centers']) for x in X]
    np.testing.assert_equal(result['embedding'],result['grid'][result['labels']])


def test_batch_is_frozen_and_not_online_epoch():
    X=np.array([[0.],[1.],[2.],[3.],[4.]])
    W=np.array([[0.],[4.]])
    got=ubukit.som_batch(X,grid_shape=(2,1),initial_prototypes=W,epochs=1,sigma=0)
    np.testing.assert_equal(got['centers'],[[1.],[3.5]])
    online=ubukit.som(X,grid_shape=(2,1),initial_prototypes=W,epochs=1,sigma=0)
    assert not np.allclose(online['centers'],got['centers'])


@pytest.mark.parametrize('make',[ubukit.initialize_som,ubukit.initialize_som_batch])
def test_ownership_resume_cancel(make):
    x=X.copy();w=W.copy()
    state=make(x,grid_shape=(2,2),initial_prototypes=w,epochs=3)
    x[:]=99;w[:]=-99
    first=state.step();first['centers'][:]=100;first['grid'][:]=100
    copy=state.centers;copy[:]=200
    assert not np.any(state.centers==100)
    state.run(max_updates=1)
    before=state.result()
    state.cancel()
    assert state.step() is None
    after=state.run()
    np.testing.assert_array_equal(after['centers'],before['centers'])
    assert after['iterations']==2 and after['cancelled'] and after['done']
    assert not np.shares_memory(after['centers'],state.centers)


@pytest.mark.parametrize('fit',[ubukit.som,ubukit.som_batch])
def test_zero_iterations_ties_zero_duplicates_and_d1(fit):
    got=fit(np.zeros((3,1)),grid_shape=(2,2),epochs=0)
    assert got['iterations']==0 and got['done']
    np.testing.assert_equal(got['labels'],0)
    np.testing.assert_equal(got['centers'],0)
    got=fit([[1.],[1.],[1.]],grid_shape=(2,1),initial_prototypes=[[0.],[2.]],epochs=0)
    np.testing.assert_equal(got['labels'],0)
    np.testing.assert_equal(fit([[2.],[2.]],grid_shape=(2,2),epochs=2)['centers'],2.)


@pytest.mark.parametrize('sigma',[0.,.01])
def test_empty_neighborhood_retains_previous(sigma):
    result=ubukit.som_batch([[0.]],grid_shape=(3,1),initial_prototypes=[[0.],[3.],[7.]],epochs=1,sigma=sigma,sigma_end=sigma)
    np.testing.assert_equal(result['centers'],[[0.],[3.],[7.]])


def test_tiny_positive_neighborhood_mass_preserves_weighted_mean():
    sigma=math.sqrt(1/(2*690))
    result=ubukit.som_batch([[1e-100]],grid_shape=(2,1),initial_prototypes=[[1e-100],[1.]],epochs=1,sigma=sigma)
    np.testing.assert_allclose(result['centers'],1e-100,rtol=1e-14,atol=0)


def test_schedule_boundary_zero_and_singleton():
    assert [_schedule(8,.5,i,5,'geometric') for i in range(5)]==pytest.approx([8,4,2,1,.5])
    assert [_schedule(1,0,i,3,'geometric') for i in range(3)]==[1,.5,0]
    assert _schedule(8,.5,0,1,'geometric')==8
    got=ubukit.som([[1.]],grid_shape=(1,1),initial_prototypes=[[0.]],epochs=1,learning_rate=.7,learning_rate_end=.1,sigma=2,sigma_end=.5)
    assert got['learning_rate']==.7 and got['sigma']==2
    assert got['centers'][0,0]==.7


def test_sample_rng_and_default_16_by_16():
    rng=_mulberry32(0)
    assert [next(rng) for _ in range(4)]==[0.26642920868471265,0.0003297457005828619,0.2232720274478197,0.1462021479383111]
    a=ubukit.som(X,epochs=0,random_state=14)
    b=ubukit.som_batch(X,epochs=0,random_state=14)
    np.testing.assert_equal(a['centers'],b['centers'])
    assert a['centers'].shape==(256,3) and a['grid_shape']==(16,16)
    assert a['grid'][16].tolist()==[0,1]
    assert all(any(np.array_equal(c,x) for x in X) for c in a['centers'])


@pytest.mark.parametrize('X',[np.zeros((1,1)),np.ones((1,3)),np.zeros((5,4)),np.array([[0.],[2.],[4.]])])
def test_pca_rank_deficient_is_finite(X):
    result=ubukit.som_batch(X,grid_shape=(3,2),epochs=0,initializer='pca')
    assert result['centers'].shape==(6,X.shape[1])
    assert np.isfinite(result['centers']).all()


def test_pca_line_population_scale_sign():
    got=ubukit.som_batch([[0.,0.],[2.,4.],[4.,8.]],grid_shape=(3,1),initializer='pca',pca_scale=1,epochs=0)
    std=math.sqrt(8/3)
    np.testing.assert_allclose(got['centers'],[[2-std,4-2*std],[2,4],[2+std,4+2*std]])


@pytest.mark.parametrize('fit',[ubukit.som,ubukit.som_batch])
@pytest.mark.parametrize('bad',[[],[[]],[[math.nan]],[[math.inf]],[[1+2j]],[[True]],[["2"]]])
def test_invalid_data(fit,bad):
    with pytest.raises((ValueError,TypeError)):
        fit(bad,epochs=0)


@pytest.mark.parametrize('options',[{'grid_shape':(0,2)},{'grid_shape':(1,)},{'grid_shape':(True,2)},
    {'epochs':-1},{'epochs':.5},{'epochs':True},{'sigma':-1},{'sigma':math.inf},
    {'sigma':.5,'sigma_end':1},{'schedule':'bad'},{'initializer':'bad'},
    {'random_state':-1},{'random_state':2**32},{'random_state':True},
    {'pca_scale':-1},{'initial_prototypes':[[1.]]}, {'max_iterations':-1}])
def test_invalid_options(options):
    with pytest.raises((ValueError,TypeError)):
        ubukit.som(X,**options)


@pytest.mark.parametrize('options',[{'learning_rate':-1},{'learning_rate':1.1},
    {'learning_rate':.2,'learning_rate_end':.3},{'learning_rate':math.nan}])
def test_invalid_online_rates(options):
    with pytest.raises(ValueError):ubukit.som(X,**options)


def test_batch_rejects_learning_rate():
    with pytest.raises(ValueError,match='no learning-rate'):ubukit.som_batch(X,learning_rate=.3)


def test_scratch_limit_and_thread_scope():
    with pytest.raises(ValueError,match='scratch'):
        ubukit.som_batch(X,policy=ubukit.ExecutionPolicy(max_scratch_bytes=32))
    policy=ubukit.ExecutionPolicy(threads=1,block_rows=2,max_scratch_bytes=4000)
    result=ubukit.som_batch(X,grid_shape=(2,2),epochs=2,policy=policy)
    assert result['scratch_rows']==2
    assert result['primary_scratch_budgeted_bytes']<=4000


@pytest.mark.parametrize('fit',[ubukit.som,ubukit.som_batch])
def test_finite_extreme_inputs_and_subnormal_distances(fit):
    big=np.finfo(float).max
    got=fit([[-big],[big]],grid_shape=(2,1),initial_prototypes=[[-big],[big]],epochs=1,sigma=0)
    np.testing.assert_equal(got['labels'],[0,1])
    assert np.isfinite(got['centers']).all()
    tiny=np.nextafter(0.,1.)
    got=fit([[tiny],[3*tiny]],grid_shape=(2,1),initial_prototypes=[[0.],[3*tiny]],epochs=0)
    np.testing.assert_equal(got['labels'],[0,1])


def test_extreme_online_opposite_endpoints_does_not_overflow():
    big=np.finfo(float).max
    got=ubukit.som([[big]],grid_shape=(1,1),initial_prototypes=[[-big]],epochs=1,learning_rate=.5)
    assert got['centers'][0,0]==0


def test_extreme_batch_exact_cancellation_mean():
    got=ubukit.som_batch([[1e300],[1.],[-1e300]],grid_shape=(1,1),epochs=1)
    assert got['centers'][0,0]==pytest.approx(1/3)


def test_public_facade_lazy_and_import_order():
    code='import sys,ubukit;assert "numpy" not in sys.modules;f=ubukit.fit_som;assert callable(ubukit.som);assert f is ubukit.som;assert callable(ubukit.som_batch)'
    subprocess.run([sys.executable,'-c',code],check=True)


def test_partial_run_matches_full_schedule():
    state=ubukit.initialize_som(X,grid_shape=(2,2),initial_prototypes=W,epochs=3)
    state.run(max_updates=3);state.run(max_updates=2)
    got=state.run()
    full=ubukit.som(X,grid_shape=(2,2),initial_prototypes=W,epochs=3)
    np.testing.assert_array_equal(got['centers'],full['centers'])


def test_extreme_online_noop_at_max_for_tiny_rate():
    big=np.finfo(float).max
    got=ubukit.som([[big]],grid_shape=(1,1),initial_prototypes=[[big]],epochs=1,learning_rate=2**-54)
    assert got['centers'][0,0]==big


_FIXTURE=json.loads((Path(__file__).parent/'fixtures'/'som-shared-reference.json').read_text())
@pytest.mark.parametrize('case',_FIXTURE['cases'],ids=lambda c:c['name'])
def test_shared_python_javascript_reference(case):
    aliases={'gridShape':'grid_shape','maxIterations':'max_iterations','sigmaEnd':'sigma_end',
             'learningRate':'learning_rate','learningRateEnd':'learning_rate_end','pcaScale':'pca_scale',
             'randomState':'random_state'}
    options={aliases.get(k,k):v for k,v in case['options'].items()}
    if 'initialPrototypes' in case:options['initial_prototypes']=case['initialPrototypes']
    make=ubukit.initialize_som if case['algorithm']=='som' else ubukit.initialize_som_batch
    state=make(case['data'],**options)
    for item in case['expected']['history']:
        snapshot=state.step()
        np.testing.assert_allclose(snapshot['centers'],item,atol=3e-13,rtol=3e-13)
    result=state.result()
    np.testing.assert_allclose(result['centers'],case['expected']['centers'],atol=3e-13,rtol=3e-13)
    np.testing.assert_array_equal(result['labels'],case['expected']['labels'])


def test_online_rate_one_is_exact_sample_even_with_large_old_center():
    got=ubukit.som([[1.]],grid_shape=(1,1),initial_prototypes=[[1e16]],sigma=0,learning_rate=1,epochs=1)
    assert got['centers'][0,0]==1


def test_batch_ordinary_path_preserves_signed_cancellation_residual():
    got=ubukit.som_batch([[1e100],[1.],[-1e100]],grid_shape=(1,1),sigma=0,epochs=1)
    assert got['centers'][0,0]==pytest.approx(1/3)


@pytest.mark.parametrize('x,rate', [(-1e16+2,.5),(1.,1-2**-53)])
def test_online_convex_interpolation_preserves_small_residual(x,rate):
    old=1e16
    expected=float(Fraction(old)+Fraction(rate)*(Fraction(x)-Fraction(old)))
    result=ubukit.som([[x]],grid_shape=(1,1),initial_prototypes=[[old]],sigma=0,learning_rate=rate,epochs=1)
    assert result['centers'][0,0]==expected
