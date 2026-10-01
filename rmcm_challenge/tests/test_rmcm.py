from fractions import Fraction
import importlib.util
import numpy as np
import pytest
from scipy import sparse
from ubukit_rmcm import prepare_rmcm, fit_rmcm, fit_rmcm_numpy
from ubukit_rmcm.core import _hard

BACKENDS = ['numpy', 'csr', 'adjoint']
if importlib.util.find_spec('numba'):
    BACKENDS.append('numba')


def literal_step(X, C, delta):
    # Tiny independent O(N²) test oracle only, never a performance baseline.
    D = np.sqrt(np.sum((X[:, None] - X[None, :]) ** 2, axis=2))
    A = D <= delta
    labels = np.argmin(np.sum((X[:, None] - C[None, :]) ** 2, axis=2), axis=1)
    H = np.eye(len(C))[labels]
    R = (A @ H) / A.sum(axis=1, keepdims=True)
    mass = R.sum(axis=0)
    new = C.copy()
    new[mass > 0] = (R.T @ X)[mass > 0] / mass[mass > 0, None]
    return labels, R, new


@pytest.mark.parametrize('backend', BACKENDS)
def test_irregular_degrees_transpose(backend):
    X = np.array([[0.], [1.], [3.]])
    p = prepare_rmcm(X, 2, backend=backend)
    np.testing.assert_array_equal(p.degrees, [2, 3, 2])
    r = p.fit(2, init=[[0.], [3.]], max_iter=1)
    np.testing.assert_array_equal(r.labels, [0, 0, 1])
    np.testing.assert_allclose(r.memberships, [[1, 0], [2/3, 1/3], [.5, .5]])
    np.testing.assert_allclose(r.centers, [[1.], [11/5]], atol=1e-15)
    assert r.stop_reason == 'max_iter' and not r.converged


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('delta', [0., .5, 1.4, 2.5])
def test_one_step_independent_oracle(backend, delta):
    rng = np.random.default_rng(93)
    X = rng.normal(size=(37, 3))
    C = X[[0, 3, 9, 15]].copy()
    labels, R, centers = literal_step(X, C, delta)
    actual = fit_rmcm(X, 4, delta, init=C, backend=backend, max_iter=1)
    np.testing.assert_array_equal(actual.labels, labels)
    np.testing.assert_allclose(actual.memberships, R, atol=2e-15)
    np.testing.assert_allclose(actual.centers, centers, atol=2e-14)


@pytest.mark.parametrize('backend', BACKENDS)
def test_zero_delta_duplicates_and_ties(backend):
    X = np.array([[0.], [0.], [2.], [4.]])
    p = prepare_rmcm(X, 0, backend=backend)
    np.testing.assert_array_equal(p.degrees, [2, 2, 1, 1])
    r = p.fit(2, init=[[0.], [4.]], max_iter=1)
    np.testing.assert_array_equal(r.labels, [0, 0, 0, 1])
    np.testing.assert_array_equal(r.memberships, [[1, 0], [1, 0], [1, 0], [0, 1]])
    np.testing.assert_allclose(r.centers.ravel(), [2/3, 4])


@pytest.mark.parametrize('backend', BACKENDS)
def test_full_neighborhood_common_exact_mean_and_empty(backend):
    X = np.array([[1,-1],[4,5],[-3,-4],[1,3],[-5,3],[2,2],[-2,-2]], float)
    p = prepare_rmcm(X, 100, backend=backend)
    C = np.array([X[0], X[1], X[1]])
    r = p.fit(3, init=C, max_iter=1)
    assert r.empty_cluster_updates == 1
    np.testing.assert_array_equal(r.centers[0], r.centers[1])
    np.testing.assert_array_equal(r.centers[0], X.mean(axis=0))
    np.testing.assert_array_equal(r.centers[2], C[2])
    labels = _hard(X, C, 512)
    expected = np.bincount(labels, minlength=3) / len(X)
    np.testing.assert_allclose(r.memberships, np.tile(expected, (len(X), 1)))


@pytest.mark.parametrize('backend', BACKENDS)
def test_exact_rational_cycle(backend):
    X = np.array([[-1,-3],[5,-3],[3,4],[-1,-2],[1,-3],[2,2]], float)
    r = fit_rmcm(X, 3, 7.3, init=[[5,-3],[1,-3],[2,2]], backend=backend)
    assert r.stop_reason == 'cycle' and r.cycle_length == 2 and r.n_iter == 4
    assert not r.converged
    np.testing.assert_array_equal(r.labels, [1,2,2,1,1,2])
    np.testing.assert_allclose(r.centers[0], [47/32, -3/4])


def test_cycle_window_disabled_reports_max_iter():
    X = np.array([[-1,-3],[5,-3],[3,4],[-1,-2],[1,-3],[2,2]], float)
    r = fit_rmcm(X, 3, 7.3, init=[[5,-3],[1,-3],[2,2]], cycle_window=0, max_iter=9)
    assert r.stop_reason == 'max_iter' and r.n_iter == 9 and not r.converged


def test_complete_state_detection_includes_centers():
    p = prepare_rmcm([[0.], [1.], [2.]], 0)
    original = p._step
    calls = []
    def synthetic(C):
        i = len(calls)
        # Repeated hard labels with different retained-center states are not a cycle.
        states = [([0,0,0], [[0.],[2.]]), ([0,0,0], [[0.],[3.]]),
                  ([0,0,0], [[0.],[4.]]), ([0,0,0], [[0.],[2.]])]
        labels, new = states[i]
        calls.append(i)
        return np.array(labels), np.array(new), 1
    p._step = synthetic
    r = p.fit(2, init=[[0.], [2.]], max_iter=4)
    assert r.n_iter == 4 and r.stop_reason == 'cycle' and r.cycle_length == 3


@pytest.mark.parametrize('backend', BACKENDS)
def test_output_is_membership_that_produced_centers(backend):
    X = np.array([[0.], [2.], [3.], [7.], [8.]])
    r = fit_rmcm(X, 2, 2.1, init=[[0.], [3.]], max_iter=1, backend=backend)
    mass = r.memberships.sum(axis=0)
    np.testing.assert_allclose(r.centers, (r.memberships.T @ X) / mass[:, None])
    assert not np.array_equal(r.labels, _hard(X, r.centers, 512))


@pytest.mark.parametrize('backend', BACKENDS)
def test_repeatable_initialization_no_alias_and_no_membership(backend):
    X = np.arange(24.).reshape(12, 2)
    snapshot = X.copy()
    a = fit_rmcm(X, 4, 3, random_state=42, backend=backend)
    b = fit_rmcm(X, 4, 3, random_state=42, backend=backend, return_memberships=False)
    assert len(np.unique(a.init_indices)) == 4
    np.testing.assert_array_equal(a.init_indices, b.init_indices)
    np.testing.assert_array_equal(a.centers, b.centers)
    np.testing.assert_array_equal(X, snapshot)
    assert b.memberships is None
    a.centers[:] = 0
    np.testing.assert_array_equal(X, snapshot)


def test_preparation_snapshot_is_independent():
    X = np.array([[0.], [1.], [9.]])
    p = prepare_rmcm(X, 1)
    X[:] = 100
    P = p.neighborhood_matrix()
    P.data[:] = 0
    assert np.all(p.neighborhood_matrix().diagonal() > 0)
    assert p.fit(2, init=[[0.], [9.]]).centers[0, 0] == .5


@pytest.mark.parametrize('seed', range(10))
def test_boundary_graph_matches_literal(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(30, 7))
    distance = np.sqrt(np.einsum('i,i->', X[0] - X[1], X[0] - X[1]))
    for delta in [np.nextafter(distance, 0), distance, np.nextafter(distance, np.inf)]:
        P = prepare_rmcm(X, delta).neighborhood_matrix()
        direct = np.empty((len(X), len(X)), bool)
        for i in range(len(X)):
            diff = X - X[i]
            direct[i] = np.sqrt(np.einsum('ij,ij->i', diff, diff)) <= delta
        np.testing.assert_array_equal(P.toarray() > 0, direct)
        np.testing.assert_array_equal(direct, direct.T)
        np.testing.assert_allclose(np.asarray(P.sum(axis=1)).ravel(), 1)


@pytest.mark.parametrize('X', [[], [[np.nan]], [[np.inf]], [[1j]], np.empty((3,0)), [1,2]])
def test_invalid_X(X):
    with pytest.raises(ValueError):
        prepare_rmcm(X, 1)


@pytest.mark.parametrize('delta', [-1, np.inf, np.nan, True, 1j, '1'])
def test_invalid_delta(delta):
    with pytest.raises(ValueError):
        prepare_rmcm([[0.]], delta)


@pytest.mark.parametrize('kwargs', [dict(n_clusters=0), dict(n_clusters=4), dict(n_clusters=True),
    dict(n_clusters=1, max_iter=0), dict(n_clusters=1, max_iter=1.2),
    dict(n_clusters=1, cycle_window=-1), dict(n_clusters=1, threads=0),
    dict(n_clusters=1, return_memberships=1), dict(n_clusters=1, init=[[1j]]),
    dict(n_clusters=1, init=[[np.nan]]), dict(n_clusters=1, init=[[0.,1.]])])
def test_invalid_fit(kwargs):
    p = prepare_rmcm([[0.], [1.], [2.]], 0)
    with pytest.raises((ValueError, TypeError)):
        p.fit(**kwargs)


def test_memory_guard_before_dense_allocation():
    with pytest.raises(MemoryError, match='candidate directed edges'):
        prepare_rmcm(np.zeros((1000,2)), 1, max_edges=10_000)
    with pytest.raises(MemoryError, match='mandatory self-edge'):
        prepare_rmcm(np.zeros((10,2)), 0, max_edges=9)


def test_numeric_extremes_rejected_without_false_ties():
    with pytest.raises(FloatingPointError):
        prepare_rmcm([[1e308], [0.]], 1)
    with pytest.raises(FloatingPointError):
        fit_rmcm([[0.], [1e-200]], 1, 0, init=[[0.]])
    with pytest.raises(FloatingPointError):
        prepare_rmcm([[0.], [1e-200]], 1)


def test_kmeans_identity_and_fixed_point():
    X = np.array([[0.], [1.], [8.], [9.]])
    r = fit_rmcm_numpy(X, 2, 0, init=[[0.], [9.]])
    np.testing.assert_array_equal(r.centers, [[.5], [8.5]])
    assert r.converged and r.stop_reason == 'fixed_point' and r.n_iter == 2


def test_no_fuzzifier_or_neighbor_exclusion_option():
    with pytest.raises(TypeError):
        fit_rmcm([[0.]], 1, 0, m=2)
    with pytest.raises(TypeError):
        fit_rmcm([[0.]], 1, 0, include_self=False)


@pytest.mark.parametrize('backend', BACKENDS)
def test_exact_cycle_with_two_nonempty_clusters(backend):
    X = np.array([[-1,-3],[5,-3],[3,4],[-1,-2],[1,-3],[2,2]], float)
    r = fit_rmcm(X, 2, 7.3, init=[[1,-3],[2,2]], backend=backend)
    assert r.stop_reason == 'cycle' and r.cycle_length == 2 and r.n_iter == 3
    assert r.empty_cluster_updates == 0
    np.testing.assert_array_equal(r.labels, [0,0,1,0,0,1])


@pytest.mark.parametrize('backend', BACKENDS)
def test_conservation_invariants(backend):
    X = np.random.default_rng(74).normal(size=(50, 3))
    r = fit_rmcm(X, 4, 1, backend=backend, init=X[[0, 5, 20, 35]], max_iter=1)
    mass = r.memberships.sum(axis=0)
    np.testing.assert_allclose(mass.sum(), len(X), atol=2e-14)
    np.testing.assert_allclose(mass @ r.centers, X.sum(axis=0), atol=2e-14)
