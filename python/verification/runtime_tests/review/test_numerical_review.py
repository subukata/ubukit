"""Independent regressions for the 2026-10-02 private numerical review."""
import importlib.util
import numpy as np
import pytest
from scipy.special import softmax
import ubukit as uk
from ubukit._impl.portable_accel._som_numerics import normalize_costs_inplace

HAS_NUMBA = importlib.util.find_spec('numba') is not None
SOM_BACKENDS = ['cdist', 'cdist_optimized', 'gemm_guarded', 'gemm_centered', 'threadpool']
FCM_BACKENDS = ['numpy', 'scipy', 'blas']
METRIC_BACKENDS = ['numpy', 'sqrt_numpy']
if HAS_NUMBA:
    SOM_BACKENDS += ['numba_direct']
    FCM_BACKENDS += ['numba', 'numba_parallel']
    METRIC_BACKENDS += ['numba', 'sqrt_numba']


@pytest.mark.parametrize('backend', SOM_BACKENDS)
@pytest.mark.parametrize('offset', [0., 1e8])
def test_som_tiny_lambda_returns_finite_probabilities(backend, offset):
    x = np.array([[0.], [2.]]) + offset
    grid = np.array([[0.], [1.]])
    initial = np.full((2, 2), .5)
    out = uk.run_som_olp(x, grid, x.copy(), initial, gamma=0., lam=1e-310,
                         max_iters=2, backend=backend,
                         policy=uk.ExecutionPolicy(threads=2, block_rows=1))
    np.testing.assert_array_equal(out['P'], initial)
    np.testing.assert_array_equal(out['W'], np.full((2, 1), offset + 1.))
    np.testing.assert_array_equal(out['history'], [2., 2.])
    np.testing.assert_array_equal(initial, np.full((2, 2), .5))


@pytest.mark.parametrize('initializer', ['original', 'svd_lowrank', 'prepared'])
def test_som_tiny_lambda_initialization(initializer):
    x = np.array([[0.], [2.]])
    grid = np.array([[0.], [1.]])
    if initializer == 'prepared':
        w, p = uk.PreparedSOM(x, max_rank=1).initialize(grid, 1e-310)
    else:
        w, p = uk.initialize_som_olp(x, grid, 1e-310, initializer=initializer)
    assert np.isfinite(w).all()
    assert np.isfinite(p).all()
    np.testing.assert_array_equal(p.sum(axis=1), [1., 1.])
    np.testing.assert_array_equal(np.sort(p, axis=1), [[0., 1.], [0., 1.]])


@pytest.mark.parametrize('lam', [1e-3, 1., 10.])
def test_som_ordinary_normalization_is_bit_identical(lam):
    cost = np.random.default_rng(15).normal(size=(31, 7)) * 8
    expected = softmax(-cost / lam, axis=1)
    actual = cost.copy()
    normalize_costs_inplace(actual, lam)
    np.testing.assert_array_equal(actual, expected)


def test_som_unrepresentable_costs_raise_instead_of_nan():
    with pytest.raises(ValueError, match='SOM costs overflowed'):
        normalize_costs_inplace(np.array([[np.inf, 1.]]), 1.)


@pytest.mark.parametrize('backend', FCM_BACKENDS)
@pytest.mark.parametrize('m,init', [(2., [[1., 2e-154], [1., 2e-154]]),
                                   (1020., [[.5, .5], [.5, .5]])])
def test_fcm_tiny_weights_do_not_collapse_centers(backend, m, init):
    x = np.array([[0.], [1e-20]])
    initial = np.array(init)
    out = uk.fit_fcm(x, init=initial, m=m, max_iter=1, backend=backend)
    np.testing.assert_array_equal(out['centers'], [[5e-21], [5e-21]])
    np.testing.assert_array_equal(out['membership'], np.full((2, 2), .5))
    np.testing.assert_array_equal(initial, init)
    assert np.isfinite(out['objective'])


@pytest.mark.parametrize('backend', FCM_BACKENDS)
def test_fcm_true_empty_cluster_remains_supported(backend):
    x = np.array([[0.], [1e-20]])
    out = uk.fit_fcm(x, init=[[1., 0.], [1., 0.]], max_iter=1, backend=backend)
    np.testing.assert_array_equal(out['centers'], [[5e-21], [5e-21]])
    np.testing.assert_array_equal(out['membership'], np.full((2, 2), .5))


@pytest.mark.parametrize('backend', METRIC_BACKENDS)
@pytest.mark.parametrize('dtype', [np.float16, np.int8, np.int16, np.int32, np.float32, np.float64, '>f4'])
def test_metric_cap_tracks_actual_distance_dtype(backend, dtype):
    data = uk.prepare(np.arange(6.).reshape(-1, 1), dtype=dtype)
    expected_bytes = 6 * 6 * (4 if np.dtype(dtype) == np.dtype(np.float32) else 8)
    with pytest.raises(ValueError, match='full distance matrix exceeds cap'):
        uk.joint_quality(data, data, 1, backend=backend, max_distance_bytes=expected_bytes - 1)
    quality, stats = uk.joint_quality(data, data, 1, backend=backend,
                                      max_distance_bytes=expected_bytes, return_stats=True)
    assert stats['largest_distance_bytes'] == expected_bytes
    assert quality[0].trustworthiness == quality[0].continuity == 1.


def test_metric_sort_scratch_uses_promoted_dtype():
    data = uk.prepare(np.arange(6.).reshape(-1, 1), dtype=np.float16)
    with pytest.raises(ValueError, match='scratch cap cannot hold one row'):
        uk.joint_quality(data, data, 1, backend='numpy', rank_method='sortsearch',
                         policy=uk.ExecutionPolicy(max_scratch_bytes=47))


@pytest.mark.parametrize('backend', FCM_BACKENDS)
@pytest.mark.parametrize('coordinates,small_membership,expected', [
    ([0., 1e-20], [1e-100, 1e-160], 1e-140),
    ([0., 1e-20, -1e-20], [1e-100, 1e-160, 2e-160], -3e-140),
    ([0., 1e-20, -1e-20], [1e-100, 1e-160, 1e-160], 0.),
    ([0., 1e-20], [1e-170, 1e-180], 1e-40),
])
def test_fcm_heterogeneous_tiny_products(backend, coordinates, small_membership, expected):
    x = np.asarray(coordinates)[:, None]
    initial = np.column_stack([np.ones(len(x)), small_membership])
    out = uk.fit_fcm(x, init=initial, m=2., max_iter=1, backend=backend)
    np.testing.assert_allclose(out['centers'][1, 0], expected, rtol=3e-15, atol=0.)
    assert np.isfinite(out['membership']).all()
