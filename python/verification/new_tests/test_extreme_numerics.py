"""Finite-m references use decimal, independently of the implementation."""
from decimal import Decimal, localcontext
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from ubukit._impl.fcm import fit_fcm, memberships_from_squared_distances
from ubukit._impl.fcm._robust import distance_parts, memberships_and_logweights, weighted_centers
from ubukit._impl.portable_accel._som_numerics import normalize_costs_inplace

BACKENDS = ['numpy', 'scipy', 'blas'] + (['numba', 'numba_parallel'] if importlib.util.find_spec('numba') else [])


def reference_step(x, centers, m):
    # 420 digits resolves O(1/m) changes all the way through binary64 m=1e308.
    with localcontext() as context:
        context.prec = 420
        xx = [[Decimal(float(v)) for v in row] for row in x]
        cc = [[Decimal(float(v)) for v in row] for row in centers]
        mm = Decimal(float(m))
        uu, ll = [], []
        for row in xx:
            q = [sum((a-b)**2 for a,b in zip(row, center)) for center in cc]
            zeros = sum(v == 0 for v in q)
            if zeros:
                u = [Decimal(1)/zeros if v == 0 else Decimal(0) for v in q]
                logs = [(v.ln()*mm if v else None) for v in u]
            else:
                logits = [-v.ln()/(mm-1) for v in q]
                maximum = max(logits)
                logits = [v-maximum for v in logits]
                e = [v.exp() if v > -10000 else Decimal(0) for v in logits]
                total = sum(e)
                u = [v/total for v in e]
                logs = [(v.ln()*mm if v else None) for v in u]
            uu.append(u); ll.append(logs)
        answer = []
        for j in range(len(cc)):
            maximum = max(row[j] for row in ll if row[j] is not None)
            w = [(row[j]-maximum).exp() if row[j] is not None and row[j]-maximum > -10000 else Decimal(0) for row in ll]
            answer.append([float(sum(v*row[d] for v,row in zip(w,xx))/sum(w)) for d in range(len(xx[0]))])
        return np.array(uu, dtype=float), np.array(answer)


@pytest.mark.parametrize('m', [33., 1000., 1e12, 1e20, 1e100, 1e308])
def test_positive_distance_finite_m_reference(m):
    x = np.array([[-1.], [1.]])
    c = np.array([[-.5], [.5]])
    ratio, _, zeros, count, _ = distance_parts(x,c)
    u, logw = memberships_and_logweights(ratio,zeros,count,m)
    got = weighted_centers(x,logw,c)
    refu, refc = reference_step(x,c,m)
    assert_allclose(u,refu,rtol=3e-15,atol=0)
    assert_allclose(got,refc,rtol=3e-14,atol=0)
    assert np.max(abs(got)) > .79  # Uniform public U must not collapse centers.


@pytest.mark.parametrize('m', [1.0000000000000002, 2., 1000., 1e308])
@pytest.mark.parametrize('scale', [2.**-700, 1., 2.**700])
def test_scaled_distances_and_finite_m_weights(m,scale):
    x = np.array([[0.], [3.], [7.]])*scale
    c = np.array([[1.], [5.]])*scale
    ratio, logq, zeros, count, _ = distance_parts(x,c)
    u,logw=memberships_and_logweights(ratio,zeros,count,m)
    got=weighted_centers(x,logw,c)
    refu,refc=reference_step(x,c,m)
    assert np.isfinite(logq).all()
    assert_allclose(u,refu,atol=2e-15,rtol=2e-14)
    assert_allclose(got/scale,refc/scale,atol=2e-14,rtol=5e-14)


@pytest.mark.parametrize('m',[2.,1000.,1e308])
def test_zero_ties_and_nonzero_rows(m):
    x=np.array([[0.],[2.],[5.],[8.]])
    c=np.array([[0.],[0.],[5.]])
    ratio,_,zeros,count,_=distance_parts(x,c)
    u,logw=memberships_and_logweights(ratio,zeros,count,m)
    got=weighted_centers(x,logw,c)
    refu,refc=reference_step(x,c,m)
    assert_array_equal(u[0],[.5,.5,0.])
    assert_array_equal(u[2],[0.,0.,1.])
    assert_allclose(u,refu,atol=2e-15,rtol=2e-14)
    assert_allclose(got,refc,atol=3e-14,rtol=5e-14)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('shifted',[False,True])
def test_previously_rejected_m1000_fixture(backend,shifted):
    file=Path(__file__).parent/'data/extreme_m1000_fixture.json'
    f=json.loads(file.read_text()); x=np.array(f['data']).reshape(f['n'],f['d']);u=np.array(f['init']).reshape(f['n'],f['k'])
    if shifted:x=x-x[0]
    out=fit_fcm(x,init=u,m=1000.,max_iter=3,tol=0,backend=backend,return_history=True)
    assert np.isfinite(out['centers']).all()
    assert np.isfinite(out['membership']).all()
    assert_allclose(out['membership'].sum(axis=1),1.,atol=2e-15)
    assert out['n_iter']==3 and not out['converged']
    assert out['numerical_diagnostics']['arithmetic']=='scaled_log'
    assert out['numerical_diagnostics']['requested_backend']==backend


@pytest.mark.parametrize('backend',BACKENDS)
def test_huge_range_and_objective_diagnostics(backend):
    x=np.array([[-1e308],[1e308],[1.]])
    out=fit_fcm(x,init=np.ones((3,1)),m=2.,max_iter=1,tol=0,backend=backend)
    assert_allclose(out['centers'],[[1/3]],rtol=1e-15)
    assert_array_equal(out['membership'],np.ones((3,1)))
    assert out['objective']==np.inf
    assert out['numerical_diagnostics']['objective_status']=='overflow'
    assert np.isfinite(out['numerical_diagnostics']['log_objective'])


@pytest.mark.parametrize('backend',BACKENDS)
def test_dynamic_range_distance_recovery(backend):
    # Fast squared distances cannot distinguish first center from x=0.
    x=np.array([[0.],[1e-200],[1.]])
    init=np.array([[1.,0.],[1.,0.],[0.,1.]])
    out=fit_fcm(x,init=init,max_iter=1,tol=0,backend=backend)
    assert_allclose(out['centers'],[[5e-201],[1.]],rtol=1e-15,atol=0)
    assert_array_equal(out['membership'],init)
    assert out['numerical_diagnostics']['recovered_distance_pairs']>0


def test_tiny_lambda_exact_minimum_and_ties():
    costs=np.array([[1e308,1e308,-1e308],[1.,1.,2.],[1e-300,1e-300,2e-300]])
    normalize_costs_inplace(costs,np.nextafter(0.,1.))
    assert_array_equal(costs,[[0.,0.,1.],[.5,.5,0.],[.5,.5,0.]])


def test_huge_m_membership_stop_has_center_warning():
    x=np.array([[-1.],[-.5],[.2],[.7],[1.]])
    init=np.array([[.51,.49],[.51,.49],[.5,.5],[.49,.51],[.49,.51]])
    out=fit_fcm(x,init=init,m=1e20,max_iter=50,tol=1e-5)
    assert out['converged']
    assert out['numerical_diagnostics']['membership_convergence_only']
    assert out['numerical_diagnostics']['center_relative_delta']>1e-5
    fixed=fit_fcm(x,init=init,m=1e20,max_iter=5,tol=0)
    assert fixed['n_iter']==5 and not fixed['converged']


def test_initial_normalization_does_not_erase_whole_cluster():
    out=fit_fcm([[0.],[2.]],init=[[1e308,1e-308],[1e308,1e-308]],max_iter=1,tol=0)
    assert_array_equal(out['centers'],[[1.],[1.]])
    assert_array_equal(out['membership'],[[.5,.5],[.5,.5]])


def test_subnormal_products_aggregate_before_rounding():
    tiny=np.nextafter(0.,1.)
    x=np.array([[0.],[tiny],[tiny],[tiny]])
    logw=np.array([[0.],[-np.log(2.)],[-np.log(2.)],[-np.log(2.)]])
    c=weighted_centers(x,logw,np.zeros((1,1)))
    assert c[0,0]==tiny


def test_compensated_mixed_sign_range():
    x=np.array([[1e308],[-1e308],[1e-200]])
    c=weighted_centers(x,np.zeros((3,1)),np.zeros((1,1)))
    assert_allclose(c,[[1e-200/3]],rtol=3e-15,atol=0)


@pytest.mark.parametrize('scale',[1e-200,1e200,2.**-700,2.**700])
@pytest.mark.parametrize('dimensions',[1,3])
def test_near_one_extreme_norm_tie_keeps_square_residual(scale,dimensions):
    m=np.nextafter(1.,2.)
    x=np.zeros((1,dimensions))
    c=np.array([[scale]*dimensions,[np.nextafter(scale,np.inf)]*dimensions])
    ratio,_,zeros,count,_=distance_parts(x,c,m)
    u,logw=memberships_and_logweights(ratio,zeros,count,m)
    reference,_=reference_step(x,c,m)
    assert_allclose(u,reference,rtol=3e-14,atol=1e-15)


@pytest.mark.parametrize('backend',BACKENDS)
def test_translation_does_not_erase_small_distinct_points(backend):
    x=np.array([[1.],[0.],[1e-200]])
    init=np.array([[0.,1.],[1.,0.],[1.,0.]])
    out=fit_fcm(x,init=init,max_iter=1,tol=0,backend=backend)
    assert_allclose(out['centers'],[[5e-201],[1.]],rtol=1e-15,atol=0)
    assert out['numerical_diagnostics']['trigger']=='translation_erased_coordinate'


@pytest.mark.parametrize('backend',BACKENDS)
def test_intermediate_sum_overflow_does_not_erase_cancellation(backend):
    a=1e308; b=np.nextafter(a,0)
    x=np.array([[a],[a],[-b],[-b],[0.]])
    out=fit_fcm(x,init=np.ones((5,1)),max_iter=1,tol=0,backend=backend)
    expected=2*(a-b)/5
    assert_allclose(out['centers'],[[expected]],rtol=1e-15,atol=0)


@pytest.mark.parametrize('scale',[1.,1e-100,1e100])
def test_huge_m_tiny_initial_membership_near_tie(scale):
    tiny=1e-100
    raw=np.array([[1.,tiny],[1.,np.nextafter(tiny,np.inf)]])*scale
    out=fit_fcm([[0.],[2.]],init=raw,m=1e20,max_iter=1,tol=0)
    assert_array_equal(out['centers'][1],[2.])


def test_huge_m_near_uniform_initial_rows_under_huge_common_scaling():
    raw=np.array([[1.,np.nextafter(1.,2.)],[1e300,1e300]])
    out=fit_fcm([[0.],[2.]],init=raw,m=1e20,max_iter=1,tol=0)
    assert_array_equal(out['centers'],[[2.],[0.]])


def reference_initial_centers(x, raw, m):
    with localcontext() as context:
        context.prec=420
        rows=[[Decimal(float(v)) for v in row] for row in raw]
        rows=[[v/sum(row) for v in row] for row in rows]
        mm=Decimal(float(m))
        answer=[]
        for col in range(raw.shape[1]):
            logs=[mm*row[col].ln() if row[col] else None for row in rows]
            maximum=max(v for v in logs if v is not None)
            weights=[(v-maximum).exp() if v is not None and v-maximum>-10000 else Decimal(0) for v in logs]
            answer.append([float(sum(w*Decimal(float(row[d])) for w,row in zip(weights,x))/sum(weights)) for d in range(x.shape[1])])
        return np.array(answer)


@pytest.mark.parametrize('m',[33.,64.,1000.,1e20,1e308])
@pytest.mark.parametrize('case',['tiny_column','row_scaling','near_uniform'])
def test_initial_weights_against_420_digit_reference(m,case):
    x=np.array([[-1.],[0.],[.7],[1.]])
    if case=='tiny_column':
        t=1e-100
        raw=np.array([[1.,t],[1.,np.nextafter(t,np.inf)],[1.,2*t],[1.,np.nextafter(2*t,np.inf)]])
    elif case=='row_scaling':
        raw=np.array([[1.,np.nextafter(1.,2.)],[1e300,1e300],[1e-300,1e-300],[2.,2.]])
    else:
        raw=np.array([[.5,.5],[.5,np.nextafter(.5,1.)],[np.nextafter(.5,1.),.5],[.5,.5]])
    out=fit_fcm(x,init=raw,m=m,max_iter=1,tol=0)
    reference=reference_initial_centers(x,raw,m)
    assert_allclose(out['centers'],reference,rtol=8e-13,atol=2e-15)


@pytest.mark.parametrize('backend',BACKENDS)
def test_near_one_public_fit_preserves_identical_rows(backend):
    x=np.array([[0.],[1.],[2.6]])
    u=np.tile([.49999999999999956,.5000000000000004],(3,1))
    out=fit_fcm(x,init=u,m=np.nextafter(1.,2.),max_iter=1,tol=0,backend=backend)
    assert_array_equal(out['centers'][0],out['centers'][1])
    assert_array_equal(out['membership'],np.full((3,2),.5))
    assert out['numerical_diagnostics']['trigger']=='near_one_fuzzifier'


@pytest.mark.parametrize('backend',BACKENDS)
def test_near_one_public_fit_uses_precise_returned_center_norms(backend):
    x=np.array([[0.],[2.4]])
    lower=np.nextafter(1.,0.)
    raw=np.array([[1.,lower],[lower,1.]])
    m=np.nextafter(1.,2.)
    out=fit_fcm(x,init=raw,m=m,max_iter=1,tol=0,backend=backend)
    reference,_=reference_step(x,out['centers'],m)
    assert_allclose(out['membership'],reference,rtol=3e-14,atol=1e-15)


@pytest.mark.parametrize('backend',['scipy','blas'])
def test_sensitive_m64_returned_pair_matches_420_digit_oracle(backend):
    rng=np.random.default_rng(994)
    x=rng.normal(size=(113,17));u=rng.random((113,7))
    out=fit_fcm(x,init=u,m=64.,backend=backend,max_iter=6,tol=0)
    expected,_=reference_step(x,out['centers'],64.)
    assert_allclose(out['membership'],expected,rtol=3e-14,atol=2e-15)


@pytest.mark.parametrize('backend',BACKENDS)
def test_restored_center_rounding_is_flagged_without_recomputing_membership(backend):
    from scipy.spatial.distance import cdist
    first=1e100; second=np.nextafter(first,np.inf); third=np.nextafter(second,np.inf)
    x=np.array([[first],[second],[third]])
    init=np.array([[.8,.2],[.5,.5],[.2,.8]])
    out=fit_fcm(x,init=init,m=64.,max_iter=1,tol=0,backend=backend)
    assert out['numerical_diagnostics']['published_centers_rounded']
    assert out['numerical_diagnostics']['membership_frame']=='working_coordinates'
    assert_array_equal(out['centers'],[[first],[third]])
    assert 0.2 < out['membership'][0,1] < 0.4
    # The objective, unlike the lagged membership update, uses the published pair.
    direct=float(np.sum(out['membership']**64*cdist(x,out['centers'],'sqeuclidean')))
    assert_allclose(out['objective'],direct,rtol=3e-13,atol=0)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('small',[1e-100,1e-80])
def test_fast_stopping_norm_recovers_zero_or_subnormal_squares(backend,small):
    x=np.array([[0.],[2*small],[1.]])
    init=np.array([[1.,0.],[1.,0.],[0.,1.]])
    out=fit_fcm(x,init=init,m=2.,max_iter=1,tol=1e-210,backend=backend)
    assert 'numerical_diagnostics' not in out
    expected=np.sqrt(2.)*small*small
    assert_allclose(out['delta'],expected,rtol=2e-15,atol=0)
    assert not out['converged'] and out['n_iter']==1


@pytest.mark.parametrize('entry',['fit','membership'])
def test_wider_fuzzifier_cannot_become_infinite_float64(entry):
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip('platform has no wider exponent range')
    m=np.longdouble(np.finfo(float).max)*np.longdouble(2)
    with pytest.raises(ValueError,match='float64'):
        if entry=='fit':fit_fcm([[0.],[1.]],2,m=m,max_iter=1)
        else:memberships_from_squared_distances([[1.,2.]],m=m)


@pytest.mark.parametrize('entry',['fit','membership'])
def test_wider_near_one_fuzzifier_cannot_round_to_one(entry):
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip('platform has no wider mantissa')
    m=np.longdouble(1)+np.finfo(np.longdouble).eps
    with pytest.raises(ValueError,match='float64'):
        if entry=='fit':fit_fcm([[0.],[1.]],2,m=m,max_iter=1)
        else:memberships_from_squared_distances([[1.,2.]],m=m)
