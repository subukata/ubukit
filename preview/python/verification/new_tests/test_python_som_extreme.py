"""Independent high-precision oracle checks for the public SOM APIs."""
from decimal import Decimal, localcontext
import importlib.util
import numpy as np
import pytest
import ubukit as uk

BACKENDS=['cdist','cdist_optimized','gemm_guarded','gemm_centered','threadpool']
if importlib.util.find_spec('numba') is not None: BACKENDS += ['numba_direct']
POLICY=uk.ExecutionPolicy(threads=2,block_rows=1)
D=lambda x:Decimal.from_float(float(x))

def oracle(X,R,W,P,gamma,lam):
    with localcontext() as ctx:
        ctx.prec=100
        x=[[D(v) for v in row] for row in X];r=[[D(v) for v in row] for row in R]
        w=[[D(v) for v in row] for row in W];p=[[D(v) for v in row] for row in P]
        n,d=X.shape;m,q=R.shape
        v=[[sum((p[i][j]*r[j][h] for j in range(m)),Decimal(0)) for h in range(q)] for i in range(n)]
        for j in range(m):
            den=sum((p[i][j] for i in range(n)),Decimal(0))
            if den:
                w[j]=[sum((p[i][j]*x[i][f] for i in range(n)),Decimal(0))/den for f in range(d)]
        # The implementation returns float64 W/V between the update blocks.
        w=[[D(float(a)) for a in row] for row in w];v=[[D(float(a)) for a in row] for row in v]
        objective=Decimal(0)
        for i in range(n):
            costs=[sum(((x[i][f]-w[j][f])**2 for f in range(d)),Decimal(0))+
                   D(gamma)*sum(((v[i][h]-r[j][h])**2 for h in range(q)),Decimal(0)) for j in range(m)]
            minimum=min(costs)
            logits=[-(cost-minimum)/D(lam) for cost in costs]
            weights=[value.exp() if value>-10000 else Decimal(0) for value in logits]
            den=sum(weights,Decimal(0));p[i]=[a/den for a in weights]
            objective += minimum-D(lam)*den.ln()
        return np.array(w,dtype=float),np.array(v,dtype=float),np.array(p,dtype=float),float(objective)

CASES=[
    (np.zeros((2,1)),np.array([[-1e200],[1e200]]),np.zeros((2,1)),np.array([[.8,.2],[.3,.7]]),1e-308,1e92),
    (np.zeros((2,1)),np.array([[-1e-200],[1e-200]]),np.zeros((2,1)),np.array([[.8,.2],[.3,.7]]),1e308,1e-92),
    (np.full((3,1),1e308),np.array([[0.],[1.]]),np.full((2,1),1e308),np.array([[.5,.5]]*3),0.,1e-310),
    (np.array([[1e-200],[2e-200]]),np.array([[0.],[1.]]),np.zeros((2,1)),np.array([[1.,1e-320],[1.,1e-320]]),0.,1e-310),
    (np.array([[0.],[2e-162]]),np.array([[0.],[1.]]),np.array([[0.],[2e-162]]),np.eye(2),0.,5e-324),
    (np.array([[0.],[1e155]]),np.array([[0.],[1.]]),np.array([[0.],[1e155]]),np.eye(2),0.,1e308),
]

@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('case',CASES)
def test_one_iteration_matches_decimal(backend,case):
    x,r,w,p,gamma,lam=case
    snapshots=[a.copy() for a in (x,r,w,p)]
    expected=oracle(x,r,w,p,gamma,lam)
    result=uk.run_som_olp(x,r,w,p,gamma=gamma,lam=lam,max_iters=1,backend=backend,policy=POLICY)
    for key,value in zip(('W','V','P','history'),expected):
        np.testing.assert_allclose(result[key],value,rtol=4e-14,atol=0.)
    assert result['variant']=='extreme_float64_exponent_fallback'
    for old,new in zip(snapshots,(x,r,w,p)):np.testing.assert_array_equal(old,new)

@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('scale',[1e145,1e-145])
def test_gamma_lock_and_original_units(backend,scale):
    x=np.array([[0.],[4.],[5.]])*scale;r=np.array([[0.],[1.]])
    p=np.array([[1.,0.],[1.,0.],[0.,1.]])
    out=uk.run_som_olp(x,r,np.array([[2.],[5.]])*scale,p,gamma=4*scale*scale,lam=1e-310,max_iters=5,backend=backend,policy=POLICY)
    np.testing.assert_array_equal(out['P'],p)
    np.testing.assert_allclose(out['W'],np.array([[2.],[5.]])*scale,rtol=1e-15)
    np.testing.assert_allclose(out['history'],np.array([8.,8.])*scale*scale,rtol=2e-15,atol=0.)
    assert out['n_iter']==2

@pytest.mark.parametrize('backend',BACKENDS)
def test_symmetric_fixed_cost_tie(backend):
    x=np.zeros((2,1));r=np.array([[-1e200],[1e200]]);p=np.full((2,2),.5)
    out=uk.run_som_olp(x,r,x.copy(),p,gamma=1e-308,lam=1e-310,max_iters=4,backend=backend,policy=POLICY)
    np.testing.assert_array_equal(out['P'],p)
    np.testing.assert_allclose(out['history'],[2e92,2e92],rtol=2e-15)

@pytest.mark.parametrize('initializer',['original','svd_lowrank','prepared'])
@pytest.mark.parametrize('value',[1e308,1e-308])
def test_constant_pca_initialization(initializer,value):
    x=np.full((3,1),value);r=np.array([[-1e308],[1e308]])
    w,p=(uk.PreparedSOM(x,max_rank=1).initialize(r,1e-310) if initializer=='prepared' else
         uk.initialize_som_olp(x,r,1e-310,initializer=initializer))
    np.testing.assert_array_equal(w,np.full((2,1),value));np.testing.assert_array_equal(p,np.full((3,2),.5))

@pytest.mark.parametrize('initializer',['original','svd_lowrank','prepared'])
def test_tiny_pca_preserves_cost_temperature_ratio(initializer):
    x=np.array([[0.],[2e-162]]);r=np.array([[0.],[1.]])
    w,p=(uk.PreparedSOM(x,max_rank=1).initialize(r,5e-324) if initializer=='prepared' else
         uk.initialize_som_olp(x,r,5e-324,initializer=initializer))
    with localcontext() as ctx:
        ctx.prec=100
        expected=[]
        for row in x:
            costs=[(D(row[0])-D(z[0]))**2 for z in w];lo=min(costs)
            vals=[(-(cost-lo)/D(5e-324)).exp() for cost in costs];total=sum(vals)
            expected.append([float(a/total) for a in vals])
    np.testing.assert_allclose(p,expected,rtol=1e-14,atol=0.)

@pytest.mark.parametrize('backend',BACKENDS)
def test_genuinely_unrepresentable_objective_raises(backend):
    x=np.array([[-1e308],[1e308]]);r=np.array([[0.],[1.]])
    with pytest.raises(ValueError,match='objective is outside'):
        uk.run_som_olp(x,r,np.zeros((2,1)),np.full((2,2),.5),gamma=0.,lam=1.,max_iters=1,backend=backend,policy=POLICY)

def test_zero_iterations_preserve_contract():
    x=np.array([[-1e308],[1e308]]);r=np.array([[0.],[1.]]);w=x.copy();p=np.eye(2)
    out=uk.run_som_olp(x,r,w,p,gamma=0.,lam=1e-310,max_iters=0)
    assert out['V'] is None and len(out['history'])==0
    np.testing.assert_array_equal(out['P'],p);np.testing.assert_array_equal(out['W'],w)

def test_extreme_scratch_cap_is_enforced():
    with pytest.raises(ValueError,match='extreme SOM cost row'):
        uk.run_som_olp(np.full((2,1),1e308),np.array([[0.],[1.]]),np.full((2,1),1e308),np.full((2,2),.5),gamma=0.,lam=1.,max_iters=1,policy=uk.ExecutionPolicy(max_scratch_bytes=31))

@pytest.mark.parametrize('base',[1.,1e10,1e100,1e300])
def test_finite_common_quotient_preserves_gap(base):
    from portable_accel._som_numerics import normalize_costs_inplace
    next_value=np.nextafter(base,np.inf);gap=next_value-base;lam=.7*gap
    costs=np.array([[base,next_value]])
    normalize_costs_inplace(costs,lam)
    expected=np.array([1.,np.exp(-gap/lam)]);expected/=expected.sum()
    np.testing.assert_allclose(costs[0],expected,rtol=2e-15,atol=0.)

@pytest.mark.parametrize('costs,lam',[(np.array([[-np.finfo(float).max,np.finfo(float).max]]),np.finfo(float).max),
                                     (np.array([[0.,1e-323]]),5e-324)])
def test_normalization_extreme_signs_and_subnormal(costs,lam):
    from portable_accel._som_numerics import normalize_costs_inplace
    expected=np.exp(np.array([0.,-2.]));expected/=expected.sum()
    normalize_costs_inplace(costs,lam)
    np.testing.assert_allclose(costs[0],expected,rtol=2e-15,atol=0.)

@pytest.mark.parametrize('backend',BACKENDS)
def test_tiny_mass_created_after_first_iteration(backend):
    x=np.array([[0.],[2e-20]]);r=np.array([[0.],[1.],[2.]])
    w=np.array([[0.],[1e-20],[2e-20]])
    p=np.array([[.85,.15,0.],[0.,.15,.85]])
    out=uk.run_som_olp(x,r,w,p,gamma=0.,lam=1e-40/740,max_iters=2,tol=0.,backend=backend,policy=POLICY)
    assert 0<out['P'][0,1]<1e-300
    np.testing.assert_allclose(out['W'][1,0],1e-20,rtol=2e-15,atol=0.)

@pytest.mark.parametrize('exponent',[-600,-510,0,510,600])
def test_scaled_random_weighted_grid_oracle(exponent):
    rng=np.random.default_rng(1187+exponent)
    x=rng.normal(size=(5,3))*.2;r=np.ldexp(rng.normal(size=(3,2)),exponent)
    p=rng.random((5,3));p/=p.sum(1,keepdims=True);w=rng.normal(size=(3,3))
    # Alternate finite extreme weights without squaring the grid first.
    gamma=np.ldexp(1.,min(1023,max(-1070,-2*exponent)))
    # Residual exponents for |exponent| > 512 are still representable.
    lam=np.ldexp(1.,2*exponent+np.frexp(gamma)[1]-1) if exponent else 1.
    expected=oracle(x,r,w,p,gamma,lam)
    actual=uk.run_som_olp(x,r,w,p,gamma=gamma,lam=lam,max_iters=1)
    for key,value in zip(('W','V','P','history'),expected):
        np.testing.assert_allclose(actual[key],value,rtol=2e-13,atol=0.)

def test_unrepresentable_pca_prototypes_fail_explicitly():
    with pytest.raises(ValueError,match='PCA prototype is outside'):
        uk.initialize_som_olp(np.array([[-1e308],[1e308]]),np.array([[0.],[1.]]),1.)

@pytest.mark.parametrize('values,expected',[
    ([1e308,-1e308,1e-200],1e-200/3),
    ([2.**499,-np.nextafter(2.**499,0.),-(2.**499-np.nextafter(2.**499,0.)),1.],.25),
    ([1e308,1e308,-np.nextafter(1e308,0.),-np.nextafter(1e308,0.),0.],(1e308-np.nextafter(1e308,0.))*2/5),
])
def test_cold_mean_preserves_cross_exponent_cancellation(values,expected):
    from portable_accel._som_extreme import _weighted_mean
    np.testing.assert_allclose(_weighted_mean(np.array(values)),expected,rtol=2e-15,atol=0.)

def test_cold_mean_preserves_cancellation_between_exact_binary_products():
    from portable_accel._som_extreme import _weighted_mean
    values=[1e300,-(.7/.3)*1e300];weights=[.7,.3]
    with localcontext() as ctx:
        ctx.prec=200
        expected=float(sum(D(a)*D(b) for a,b in zip(values,weights))/sum(D(b) for b in weights))
    np.testing.assert_allclose(_weighted_mean(values,weights),expected,rtol=2e-15,atol=0.)
