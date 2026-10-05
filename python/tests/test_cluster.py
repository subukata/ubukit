import numpy as np
import pytest
from conftest import match_centers
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

import ubukit as ub

SOFT = [
    (ub.fcm, {}),
    (ub.efcm, {}),
    (ub.rcm, {}),
    (ub.rcm, {"p": 2.0, "beta": 0.3}),
    (ub.rmcm, {"delta": 0.5}),
]


def fit(method, X, k, options):
    options = dict(options)
    delta = options.pop("delta", None)
    return method(X, k, delta, **options) if delta is not None else method(X, k, **options)


@pytest.mark.parametrize(("method", "options"), [(ub.kmeans, {}), *SOFT])
def test_recovers_blobs(blobs, method, options):
    X, centers = blobs
    r = fit(method, X, 3, {"seed": 0, **options})
    assert r.converged
    assert r.centers.shape == (3, 2)
    assert r.labels.shape == (300,)
    assert match_centers(r.centers, centers) < 0.15


@pytest.mark.parametrize(("method", "options"), SOFT)
def test_memberships_are_row_stochastic(blobs, method, options):
    X, _ = blobs
    U = fit(method, X, 3, {"seed": 1, **options}).membership
    assert U.shape == (300, 3)
    assert (U >= 0).all()
    np.testing.assert_allclose(U.sum(axis=1), 1.0)


@pytest.mark.parametrize("k", [1, 2, 7, 40])
def test_kmeans_iterates_are_lloyds(k):
    # Hamerly's bounds only skip work: labels, centers, objective history and
    # iteration count equal those of a plain Lloyd loop.
    rng = np.random.default_rng(k)
    X = rng.normal(size=(600, 5)) + 3.0 * rng.integers(0, 4, size=(600, 1))
    init = X[rng.choice(600, k, replace=False)] + 0.01
    V, history = init.copy(), []
    for _ in range(300):
        D = cdist(X, V, "sqeuclidean")
        labels = D.argmin(axis=1)
        history.append(D[np.arange(600), labels].sum())
        means = [X[labels == c].mean(axis=0) if (labels == c).any() else V[c] for c in range(k)]
        new = np.array(means)
        if np.array_equal(new, V):
            break
        V = new
    r = ub.kmeans(X, k, init=init)
    np.testing.assert_array_equal(r.labels, labels)
    np.testing.assert_allclose(r.centers, V, atol=1e-10)
    np.testing.assert_allclose(r.history, history, rtol=1e-9)
    assert r.n_iter == len(history)


def test_kmeans_with_duplicate_initial_centers_breaks_ties_by_index():
    X = np.random.default_rng(3).normal(size=(200, 3))
    init = X[[0, 0, 1]]
    r = ub.kmeans(X, 3, init=init, max_iter=1)
    D = cdist(X, init, "sqeuclidean")
    np.testing.assert_array_equal(r.labels, D.argmin(axis=1))
    assert not (r.labels == 1).any()


def test_kmeans_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    init = X[:6] + 0.01
    ours = ub.kmeans(X, 6, init=init)
    ref = KMeans(6, init=init, n_init=1, algorithm="lloyd", tol=0).fit(X)
    np.testing.assert_allclose(ours.centers, ref.cluster_centers_, atol=1e-10)
    np.testing.assert_array_equal(ours.labels, ref.labels_)


@pytest.mark.parametrize(
    ("method", "options"), [(ub.kmeans, {}), (ub.fcm, {"m": 1.7}), (ub.efcm, {"tau": 0.5})]
)
def test_objective_never_increases(blobs, method, options):
    X, _ = blobs
    h = method(X, 4, seed=2, **options).history
    assert len(h) > 1
    assert np.all(np.diff(h) <= 1e-9 * abs(h[0]))


@pytest.mark.parametrize("m", [1.0 + 1e-6, 1.01, 3.0, 50.0, 1000.0])
def test_fcm_is_stable_for_extreme_fuzzifiers(blobs, m):
    X, centers = blobs
    r = ub.fcm(X, 3, m=m, seed=0, max_iter=1000)
    assert np.isfinite(r.membership).all()
    assert r.converged and r.n_iter > 1
    # Centers are the u^m-weighted means of the memberships; u^m itself
    # underflows for large m, so the reference weights use logarithms.
    with np.errstate(divide="ignore"):
        L = m * np.log(r.membership)
    W = np.exp(L - L.max(axis=0))
    np.testing.assert_allclose(r.centers, W.T @ X / W.sum(axis=0)[:, None], atol=1e-9)
    if m < 5:
        assert match_centers(r.centers, centers) < 0.15


def test_fcm_near_one_approaches_kmeans(blobs):
    X, _ = blobs
    init = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]])
    hard = ub.kmeans(X, 3, init=init)
    soft = ub.fcm(X, 3, m=1.0 + 1e-9, init=init)
    np.testing.assert_allclose(soft.centers, hard.centers, atol=1e-6)


def test_rcm_without_margin_is_kmeans(blobs):
    X, _ = blobs
    init = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]])
    rough = ub.rcm(X, 3, alpha=1.0, init=init)
    np.testing.assert_allclose(rough.centers, ub.kmeans(X, 3, init=init).centers)
    assert set(np.unique(rough.membership)) <= {0.0, 1.0}


def test_rcm_memberships_split_equally(blobs):
    X, _ = blobs
    U = ub.rcm(X, 3, alpha=1.5, beta=0.5, seed=0).membership
    for row in U:
        nonzero = row[row > 0]
        np.testing.assert_allclose(nonzero, 1.0 / len(nonzero))


def test_rmcm_with_zero_radius_is_kmeans(blobs):
    X, _ = blobs
    init = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]])
    np.testing.assert_allclose(
        ub.rmcm(X, 3, 0.0, init=init).centers, ub.kmeans(X, 3, init=init).centers
    )


def test_rmcm_refuses_dense_graphs(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match="max_edges"):
        ub.rmcm(X, 3, 100.0, max_edges=1000)


def test_translation_invariance(blobs):
    X, _ = blobs
    init = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]])
    shift = np.array([1e7, -3e6])
    a = ub.fcm(X, 3, init=init)
    b = ub.fcm(X + shift, 3, init=init + shift)
    np.testing.assert_allclose(b.centers - shift, a.centers, atol=1e-7)


def test_seed_reproducibility(blobs):
    X, _ = blobs
    a, b = ub.fcm(X, 4, seed=5), ub.fcm(X, 4, seed=5)
    np.testing.assert_array_equal(a.centers, b.centers)


def test_identical_points_have_zero_span_and_are_accepted():
    r = ub.fcm(np.ones((5, 2)), 2, seed=0)
    np.testing.assert_allclose(r.centers, 1.0)
    assert r.converged


def test_coincident_points_and_centers():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    r = ub.fcm(X, 2, init=X[[0, 2]])
    np.testing.assert_allclose(np.sort(r.centers, axis=0), [[0, 0], [1, 1]])
    assert np.isfinite(r.membership).all()


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda X: ub.kmeans(X, 0), "k"),
        (lambda X: ub.kmeans(X, 301), "k"),
        (lambda X: ub.kmeans(X, 3, init=np.zeros((2, 2))), "init"),
        (lambda X: ub.kmeans(X, 3, init="random"), "init"),
        (lambda X: ub.fcm(X, 3, m=1.0), "m"),
        (lambda X: ub.efcm(X, 3, tau=0), "tau"),
        (lambda X: ub.rcm(X, 3, alpha=0.9), "alpha"),
        (lambda X: ub.rmcm(X, 3, -1.0), "delta"),
        (lambda X: ub.kmeans(np.where(X > 4, np.nan, X), 3), "finite"),
        (lambda X: ub.kmeans(X * 1e200, 3), "scale"),
        (lambda X: ub.kmeans(X * 1e-170, 3), "scale"),
        (lambda X: ub.rmcm(X, 3, 0.0, max_edges=10), "max_edges"),
        (lambda X: ub.kmeans(X[:, 0], 3), "2-D"),
        (lambda X: ub.kmeans(X, True), "k"),
    ],
)
def test_invalid_arguments(blobs, call, match):
    X, _ = blobs
    with pytest.raises(ValueError, match=match):
        call(X)
