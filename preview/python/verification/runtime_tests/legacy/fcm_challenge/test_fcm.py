import importlib.util

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist

from ubukit_fcm import fit_fcm, fit_fcm_numpy, memberships_from_squared_distances

BACKENDS = ['numpy', 'scipy', 'blas']
if importlib.util.find_spec('numba'):
    BACKENDS += ['numba', 'numba_parallel']


def fixture(seed=7, n=83, d=5, k=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.random((n, k))


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('m', [1.01, 1.3, 2.0, 3.5, 10.0, 1000.0])
def test_agree_reference(backend, m):
    x, u = fixture()
    ref = fit_fcm_numpy(x, init=u, m=m, max_iter=13, tol=0, return_history=True)
    got = fit_fcm(x, init=u, m=m, max_iter=13, tol=0, backend=backend, return_history=True)
    assert_allclose(got['centers'], ref['centers'], atol=3e-11, rtol=2e-10)
    assert_allclose(got['membership'], ref['membership'], atol=1e-10, rtol=2e-8)
    assert_allclose(got['objective'], ref['objective'], atol=1e-12, rtol=1e-10)
    assert got['n_iter'] == 13
    assert not got['converged']
    assert_allclose(got['membership'].sum(axis=1), 1, atol=1e-15)
    assert np.all(np.diff(got['objective_history']) <= 1e-10)


@pytest.mark.parametrize('backend', BACKENDS)
def test_final_pair_objective_and_input_unchanged(backend):
    x, u = fixture()
    xcopy, ucopy = x.copy(), u.copy()
    got = fit_fcm(x, init=u, max_iter=1, backend=backend)
    direct = np.sum(got['membership'] ** 2 * cdist(x, got['centers'], 'sqeuclidean'))
    assert_allclose(got['objective'], direct, rtol=1e-14)
    assert_allclose(got['membership'], memberships_from_squared_distances(cdist(x, got['centers'], 'sqeuclidean')), atol=1e-14)
    assert_array_equal(x, xcopy)
    assert_array_equal(u, ucopy)
    assert_array_equal(got['labels'], got['membership'].argmax(axis=1))
    assert_allclose(got['fpc'], np.sum(got['membership']**2)/len(x))


@pytest.mark.parametrize('backend', BACKENDS)
def test_duplicates_exact_zero_and_empty_cluster(backend):
    x = np.array([[0., 0.], [0., 0.], [3., 4.], [3., 4.]])
    init = np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]])
    got = fit_fcm(x, init=init, backend=backend, max_iter=10)
    assert got['objective'] == 0
    assert got['converged']
    assert_array_equal(got['membership'], init)
    got = fit_fcm(np.ones((8, 3)), 5, backend=backend, random_state=42)
    assert_array_equal(got['membership'], np.full((8, 5), 0.2))
    assert got['objective'] == 0


def test_zero_membership_and_near_one_math():
    d2 = np.array([[0., 0., 9.], [1e-300, 1., 1e300], [1., 1.0001, 10.]])
    got = memberships_from_squared_distances(d2, 1.0000001)
    assert_array_equal(got[0], [.5, .5, 0])
    assert_array_equal(got[1], [1., 0., 0.])
    assert got[2, 0] == 1.
    # m=2 inverse squared distances, not inverse Euclidean distances.
    assert_allclose(memberships_from_squared_distances([[1., 4.]]), [[.8, .2]])


@pytest.mark.parametrize('backend', BACKENDS)
def test_translation_invariance(backend):
    rng = np.random.default_rng(9)
    # Exactly representable binary increments avoid changing the input data.
    x = rng.integers(-100, 100, size=(71, 4)) / 8.
    u = rng.random((len(x), 4))
    a = fit_fcm(x, init=u, backend=backend, max_iter=10, tol=0)
    b = fit_fcm(x + 2**40, init=u, backend=backend, max_iter=10, tol=0)
    assert_allclose(a['membership'], b['membership'], atol=1e-14, rtol=1e-14)
    assert_allclose(a['centers'], b['centers'] - 2**40, atol=1.23e-4, rtol=0)
    direct = np.sum(b['membership']**2 * cdist(x+2**40, b['centers'], 'sqeuclidean'))
    assert_allclose(b['objective'], direct, rtol=1e-14)


@pytest.mark.parametrize('backend', BACKENDS)
def test_stop_and_fixed_iterations(backend):
    x, u = fixture()
    one = fit_fcm(x, init=u, backend=backend, max_iter=1, tol=100)
    assert one['n_iter'] == 1 and one['converged']
    fixed = fit_fcm(np.ones((2, 1)), 2, backend=backend, max_iter=7, tol=0, random_state=3)
    assert fixed['n_iter'] == 7 and not fixed['converged']
    assert fixed['delta'] == 0


@pytest.mark.parametrize('backend', BACKENDS)
def test_single_cluster(backend):
    x, _ = fixture()
    got = fit_fcm(x, 1, backend=backend)
    assert_allclose(got['centers'][0], x.mean(axis=0))
    assert_array_equal(got['membership'], np.ones((len(x), 1)))
    assert got['fpc'] == 1


@pytest.mark.parametrize('kwargs', [dict(m=1), dict(m=np.inf), dict(m=np.nan), dict(tol=-1), dict(tol=np.inf), dict(max_iter=0), dict(max_iter=1.5), dict(max_iter=True), dict(threads=0), dict(backend='missing')])
def test_invalid_parameters(kwargs):
    with pytest.raises((ValueError, TypeError)):
        fit_fcm(np.ones((3, 2)), 2, **kwargs)


@pytest.mark.parametrize('x', [[], [[np.inf]], [[np.nan]], np.zeros((3, 0)), np.zeros((0, 2)), [1., 2.]])
def test_invalid_x(x):
    with pytest.raises(ValueError):
        fit_fcm(x, 2)


def test_init_random_and_strided():
    x, u = fixture()
    a = fit_fcm(x[:, ::-1], 4, random_state=71)
    b = fit_fcm(x[:, ::-1].copy(), 4, random_state=71)
    assert_array_equal(a['membership'], b['membership'])
    for bad in [np.zeros((len(x), 4)), -u, u[:3], np.full_like(u, np.nan)]:
        with pytest.raises(ValueError):
            fit_fcm(x, init=bad)


@pytest.mark.skipif(importlib.util.find_spec('skfuzzy') is None, reason='optional comparator')
@pytest.mark.parametrize('m', [1.3, 2., 3.5])
def test_scikit_fuzzy_fixed_and_converged(m):
    from skfuzzy.cluster import cmeans
    x, u = fixture(n=200, d=3)
    u /= u.sum(axis=1, keepdims=True)
    for max_iter, tol in [(15, 0), (300, 1e-5)]:
        ref = cmeans(x.T, u.shape[1], m, tol, max_iter, init=u.T)
        got = fit_fcm(x, init=u, m=m, max_iter=max_iter, tol=tol)
        assert got['n_iter'] == ref[5]
        assert_allclose(got['centers'], ref[0], atol=1e-10, rtol=1e-9)
        assert_allclose(got['membership'], ref[1].T, atol=1e-10, rtol=1e-8)
        assert_allclose(got['objective'], np.sum(ref[1]**m*cdist(ref[0], x, 'sqeuclidean')), atol=1e-10)


def test_large_membership_values_normalize_safely():
    x = np.arange(10.)[:, None]
    init = np.full((10, 2), 1e308)
    got = fit_fcm(x, init=init, max_iter=1)
    assert_allclose(got['membership'], .5)


@pytest.mark.parametrize('backend', BACKENDS)
def test_extreme_distance_range_has_explicit_objective_diagnostics(backend):
    # Supersedes the former rejection-only contract: finite-m fitting now
    # supports out-of-range squared distances, without clipping the objective.
    out = fit_fcm(np.array([[0.], [1e200]]), init=np.full((2, 2), .5),
                  max_iter=1, tol=0, backend=backend)
    assert_allclose(out['centers'], [[5e199], [5e199]], rtol=1e-15)
    assert_array_equal(out['membership'], np.full((2, 2), .5))
    assert out['objective'] == np.inf
    assert out['numerical_diagnostics']['objective_status'] == 'overflow'
    assert np.isfinite(out['numerical_diagnostics']['log_objective'])


def test_thread_limit_restored():
    from threadpoolctl import threadpool_info
    before = threadpool_info()
    x, u = fixture()
    fit_fcm(x, init=u, max_iter=2, threads=1)
    after = threadpool_info()
    assert [(p['filepath'], p['num_threads']) for p in before] == [(p['filepath'], p['num_threads']) for p in after]


def test_fractional_power_after_underflow_and_alias():
    d2 = np.array([[1e-300, 1e300]])
    expected = np.exp((np.log(d2.min())-np.log(d2))/(1000-1))
    expected /= expected.sum(axis=1, keepdims=True)
    assert_allclose(memberships_from_squared_distances(d2, 1000), expected, atol=1e-15)
    zero = np.array([[0., 0., 9.]])
    assert_array_equal(memberships_from_squared_distances(zero, out=zero), [[.5, .5, 0.]])


@pytest.mark.parametrize('backend', BACKENDS)
def test_tiny_scale_invariance(backend):
    x = np.array([[0.], [1.], [3.]])
    u = np.array([[.9,.1], [.7,.3], [.1,.9]])
    a = fit_fcm(x, init=u, backend=backend, max_iter=1)
    b = fit_fcm(x*1e-200, init=u, backend=backend, max_iter=1)
    assert_allclose(a['membership'], b['membership'], atol=1e-14)
    assert_allclose(a['centers'], b['centers']/1e-200, atol=1e-14)


@pytest.mark.parametrize('backend', BACKENDS)
def test_objective_recovers_underflowed_weights(backend):
    got = fit_fcm(np.array([[0.], [2e150]]), init=np.full((2,2),.5), m=1100,
                  max_iter=1, backend=backend, return_history=True)
    expected = np.exp(np.log(4) - 1100*np.log(2) + np.log(1e300))
    assert_allclose(got['objective'], expected, rtol=1e-12, atol=0)
    assert_allclose(got['objective_history'], [expected], rtol=1e-12, atol=0)


@pytest.mark.parametrize('backend', BACKENDS)
def test_objective_sum_recovers_tiny_distances(backend):
    x = np.tile([[0.], [2e-162]], (100, 1))
    got = fit_fcm(x, init=np.ones((len(x),1)), max_iter=1, backend=backend)
    # Each squared residual is too small to represent; their sum is not.
    expected = float(np.longdouble(200)*np.longdouble(1e-162)**2)
    assert got['objective'] > 0
    assert abs(got['objective'] - expected) <= np.nextafter(0., 1.)


def test_reject_complex_inputs():
    with pytest.raises(ValueError, match='real-valued'):
        fit_fcm(np.array([[0+0j], [1+9j]]), 1)
    with pytest.raises(ValueError, match='real-valued'):
        fit_fcm(np.array([[0.], [1.]]), init=np.ones((2,2),complex))
    with pytest.raises(ValueError, match='real-valued'):
        memberships_from_squared_distances([[1+4j,9+0j]])


@pytest.mark.parametrize('backend', BACKENDS)
def test_objective_aggregates_before_underflow(backend):
    x = np.tile([[0.], [2.]], (100, 1))
    got = fit_fcm(x, init=np.full((len(x),2),.5), m=1080,
                  max_iter=1, backend=backend, return_history=True)
    expected = float(np.longdouble(400)*np.longdouble(.5)**1080)
    assert expected > 0
    assert abs(got['objective'] - expected) <= np.nextafter(0.,1.)
    assert abs(got['objective_history'][0] - expected) <= np.nextafter(0.,1.)
