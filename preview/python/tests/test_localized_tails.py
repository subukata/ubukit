"""Independent correctness/ownership checks for the isolated probability-tail candidate."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import numpy as np
import pytest
import ubukit as uk
from ubukit._impl.portable_accel.som_olp_localized import run_som_olp_localized, _repair_statistics, _max_abs_columns
from ubukit._impl.portable_accel._som_extreme import run as cold_run

POLICY = uk.ExecutionPolicy(threads=1, block_rows=3)
D = lambda x: Decimal.from_float(float(x))


def oracle(X,R,W,P,gamma,lam):
    with localcontext() as ctx:
        ctx.prec = 600
        x=[[D(a) for a in row] for row in X];r=[[D(a) for a in row] for row in R]
        w=[[D(a) for a in row] for row in W];p=[[D(a) for a in row] for row in P]
        n,d=X.shape;m,q=R.shape
        v=[[sum((p[i][j]*r[j][h] for j in range(m)),D(0)) for h in range(q)] for i in range(n)]
        for j in range(m):
            den=sum((p[i][j] for i in range(n)),D(0))
            if den:w[j]=[sum((p[i][j]*x[i][f] for i in range(n)),D(0))/den for f in range(d)]
        w=[[D(float(a)) for a in row] for row in w];v=[[D(float(a)) for a in row] for row in v]
        objective=D(0)
        for i in range(n):
            cost=[sum(((x[i][f]-w[j][f])**2 for f in range(d)),D(0))+D(gamma)*sum(((v[i][h]-r[j][h])**2 for h in range(q)),D(0)) for j in range(m)]
            minimum=min(cost); logits=[-(a-minimum)/D(lam) for a in cost]
            weights=[a.exp() if a>-10000 else D(0) for a in logits];den=sum(weights,D(0))
            p[i]=[a/den for a in weights];objective+=minimum-D(lam)*den.ln()
        return dict(W=np.array(w,float),V=np.array(v,float),P=np.array(p,float),history=np.array([float(objective)]))


def call(X,R,W,P,gamma=.2,lam=.4,**kwargs):
    return run_som_olp_localized(X,R,W,P,gamma=gamma,lam=lam,max_iters=kwargs.pop('max_iters',1),policy=kwargs.pop('policy',POLICY),**kwargs)


@pytest.mark.parametrize('seed',range(10))
def test_random_tail_one_step_decimal(seed):
    rng=np.random.default_rng(seed);X=rng.normal(size=(7,4));R=rng.normal(size=(5,2));W=rng.normal(size=(5,4))
    P=rng.random((7,5));P[:,4]*=1e-300;P/=P.sum(axis=1,keepdims=True)
    expected=oracle(X,R,W,P,.2,.4);result=call(X,R,W,P)
    assert result['variant']=='experimental_localized_probability_tails'
    for k,v in expected.items():np.testing.assert_allclose(result[k],v,rtol=2e-13,atol=1e-14)
    assert result['exact_prototype_repairs']>=4


@pytest.mark.parametrize('scale',[1.,1e100,1e-100])
def test_mixed_exponent_signed_mean_residual(scale):
    X=np.array([[1.],[-1.],[1.]])*scale;R=np.array([[0.],[1.]])
    P=np.array([[1.,0.],[1.,0.],[1e-300,1.]])
    W=np.zeros((2,1));out=call(X,R,W,P,gamma=0.,lam=1.)
    expected=float(Fraction(float(scale))*Fraction(1e-300)/(2+Fraction(1e-300)))
    assert out['W'][0,0]==expected
    assert out['exact_prototype_repairs']>=1


def test_grid_signed_residual_and_products_aggregate_before_rounding():
    tiny=np.nextafter(0.,1.);X=np.ones((1,1));R=np.array([[1.],[-1.],[.5],[.5],[.5]])
    P=np.array([[.5,.5,tiny,tiny,tiny]]);W=np.ones((5,1));out=call(X,R,W,P,gamma=0.,lam=1.)
    expected=float(Fraction(tiny)*Fraction(3,2));assert out['V'][0,0]==expected
    assert out['exact_grid_repairs']==1


def test_tiny_denominator_preserves_exact_binary_product_residual():
    X=np.array([[1.],[-(.7/.3)]]);R=np.array([[0.],[1.]])
    P=np.array([[1.,.7e-300],[1.,.3e-300]]);W=np.zeros((2,1))
    out=call(X,R,W,P,gamma=0.,lam=1.)
    exact=sum(Fraction(float(x))*Fraction(float(p)) for x,p in zip(X[:,0],P[:,1]))/sum(Fraction(float(p)) for p in P[:,1])
    # The reference extended path rounds its numerator and denominator once
    # before dividing; allow its documented float64-significand rounding.
    np.testing.assert_allclose(out['W'][1,0],float(exact),rtol=3e-15,atol=0.)


@pytest.mark.parametrize('backend',['cdist','cdist_optimized','numpy','auto'])
def test_ownership_lagged_outputs_and_determinism(backend):
    rng=np.random.default_rng(17);X=rng.normal(size=(12,3));R=rng.normal(size=(4,2));W=rng.normal(size=(4,3));P=rng.random((12,4));P[:,-1]*=1e-300;P/=P.sum(1,keepdims=True)
    snapshots=[a.copy() for a in (X,R,W,P)];out=call(X,R,W,P,max_iters=3,tol=0.,backend=backend);again=call(X,R,W,P,max_iters=3,tol=0.,backend=backend)
    for a,b in zip((X,R,W,P),snapshots):np.testing.assert_array_equal(a,b)
    for name in ('W','P','V','history'):np.testing.assert_array_equal(out[name],again[name]);assert not any(np.shares_memory(out[name],a) for a in (X,R,W,P))
    for a in (X,R,W,P):a.setflags(write=False)
    call(X,R,W,P,max_iters=2)


@pytest.mark.parametrize('field',['W','P'])
def test_zero_iterations_and_detached_outputs(field):
    X=np.ones((2,1));R=np.arange(2.)[:,None];W=X.copy();P=np.array([[1.,1e-300]]*2)
    out=call(X,R,W,P,max_iters=0)
    np.testing.assert_array_equal(out[field],dict(W=W,P=P)[field]);assert out['V'] is None
    assert not np.shares_memory(out[field],dict(W=W,P=P)[field])


def test_invalid_probability_and_low_scratch_rejected():
    X=np.ones((2,1));R=np.arange(2.)[:,None];W=X.copy();P=np.array([[1.,1e-300]]*2)
    with pytest.raises(ValueError,match='scratch'):call(X,R,W,P,policy=uk.ExecutionPolicy(threads=1,max_scratch_bytes=16))
    limited=call(X,R,W,P,policy=uk.ExecutionPolicy(threads=1,max_scratch_bytes=64))
    assert limited['variant']=='extreme_float64_exponent_fallback'
    P[0,0]=.9
    with pytest.raises(ValueError,match='row-stochastic'):call(X,R,W,P)


def test_cold_cost_rows_and_original_units():
    X=np.array([[0.],[1.]]);R=np.arange(2.)[:,None];W=X.copy();P=np.array([[1.,1e-300],[1e-300,1.]])
    out=call(X,R,W,P,gamma=0.,lam=np.nextafter(0.,1.))
    expected=cold_run(X,R,W,P,0.,np.nextafter(0.,1.),1,0.)
    assert out['direct_fallback_rows']==2
    for k in ('W','V','P','history'):np.testing.assert_allclose(out[k],expected[k],rtol=3e-15,atol=0.)


def test_entropy_tail_preserved_and_composite_overflow_repaired():
    X=np.zeros((2,1));R=np.array([[0.],[1e100]])
    P=np.array([[1.,1e-300]]*2);W=np.zeros((2,1))
    out=call(X,R,W,P,gamma=1e140,lam=1.)
    expected=cold_run(X,R,W,P,1e140,1.,1,0.)
    for k in ('W','V','P','history'):np.testing.assert_allclose(out[k],expected[k],rtol=3e-15,atol=0.)
    assert out['direct_fallback_rows']==2


def test_coordinate_extremes_keep_original_fallback():
    X=np.array([[1e308],[1e308]]);R=np.arange(2.)[:,None];W=X.copy();P=np.array([[1.,1e-300]]*2)
    out=call(X,R,W,P,gamma=0.,lam=1.)
    assert out['variant']=='extreme_float64_exponent_fallback'


def test_ordinary_dispatch_is_exactly_original():
    rng=np.random.default_rng(9);X=rng.normal(size=(6,2));R=np.arange(3.)[:,None];W=rng.normal(size=(3,2));P=np.full((6,3),1/3)
    kwargs=dict(gamma=.1,lam=.5,max_iters=5,tol=0.,backend='cdist_optimized',policy=POLICY)
    old=uk.run_som_olp(X,R,W,P,**kwargs);new=run_som_olp_localized(X,R,W,P,**kwargs)
    for k in ('W','P','V','history'):np.testing.assert_array_equal(old[k],new[k])
    assert old['variant']==new['variant']


def test_gamma_rescues_underflowed_grid_square_hidden_by_data_cost():
    X=np.array([[0.],[2e-125]]);R=np.array([[0.],[1.],[-1.]])
    P=np.array([[1.,1e-180,0.],[1.,0.,1e-180]]);W=np.zeros((3,1))
    out=call(X,R,W,P,gamma=1e140,lam=1e-250)
    expected=oracle(X,R,W,P,1e140,1e-250)
    for key in ('W','V','P','history'):
        np.testing.assert_allclose(out[key],expected[key],rtol=3e-14,atol=0.)
    np.testing.assert_allclose(out['history'],[2e-220],rtol=3e-14,atol=0.)
    assert out['direct_fallback_rows']==2
