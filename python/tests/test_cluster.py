import numpy as np
import pytest
from conftest import match_centers
from scipy.spatial.distance import cdist
from scipy.special import xlogy
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


def test_default_seeding_rarely_merges_separated_blobs():
    # Greedy k-means++ keeps the best of 2 + ln k draws per seed; with a single
    # draw, two seeds often land in one of these well-separated clusters.
    rng = np.random.default_rng(0)
    centers = rng.uniform(-5, 5, (10, 16))
    y = np.repeat(np.arange(10), 200)
    X = centers[y] + rng.normal(size=(2000, 16))
    perfect = sum(ub.ari(y, ub.kmeans(X, 10, seed=s).labels) == 1.0 for s in range(20))
    assert perfect >= 13  # 16 of 20 with greedy seeding, 6 with single draws


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


@pytest.mark.parametrize(
    ("method", "options", "criterion"),
    [
        (ub.fcm, {"m": 1.5}, lambda U, D: np.sum(U**1.5 * D)),
        (ub.fcm, {"m": 2.0}, lambda U, D: np.sum(U**2 * D)),
        (ub.fcm, {"m": 3.0}, lambda U, D: np.sum(U**3 * D)),
        (ub.efcm, {"tau": 0.7}, lambda U, D: np.sum(U * D) + 0.7 * np.sum(xlogy(U, U))),
    ],
)
def test_history_is_the_objective_of_the_memberships(blobs, method, options, criterion):
    # The history is computed by an identity that holds at the memberships of
    # the distances; one iteration from given centers checks it against the
    # definition.
    X, _ = blobs
    init = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]])
    r = method(X, 3, init=init, max_iter=1, **options)
    D = cdist(X, init, "sqeuclidean")
    assert r.history[0] == pytest.approx(criterion(r.membership, D), rel=1e-12)


@pytest.mark.parametrize("m", [1.0 + 1e-6, 1.01, 3.0, 50.0, 1000.0])
def test_fcm_is_stable_for_extreme_fuzzifiers(blobs, m):
    X, centers = blobs
    r = ub.fcm(X, 3, m=m, seed=0, max_iter=1000)
    assert np.isfinite(r.membership).all()
    # No n_iter check: with greedy seeding, m near 1 converges in one step. A
    # stall (u^m underflowing, the centers never moving) fails the check below.
    assert r.converged
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


@pytest.mark.parametrize(("scale", "tau"), [(1.0, 1e-320), (1e148, 1e-12)])
def test_efcm_reaches_kmeans_as_tau_vanishes(blobs, scale, tau):
    # Costs over tau beyond float64 (data of the largest allowed scale, or a
    # subnormal tau) give hard memberships, the limit, not NaN marked converged.
    X, _ = blobs
    init = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]])
    with np.errstate(all="raise"):
        r = ub.efcm(X * scale, 3, tau=tau, init=init * scale)
    assert set(np.unique(r.membership)) == {0.0, 1.0}
    assert np.isfinite(r.history).all()
    hard = ub.kmeans(X * scale, 3, init=init * scale)
    np.testing.assert_allclose(r.centers, hard.centers, rtol=1e-12)


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
    # Bezdek's rule for d = 0: membership 1, shared equally among centers on the point.
    r = ub.fcm(X, 3, init=X[[0, 0, 2]], max_iter=1)
    np.testing.assert_allclose(
        r.membership, [[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1], [0, 0, 1]], atol=1e-300
    )


@pytest.mark.parametrize("k", [3, 5])
def test_fewer_distinct_points_than_clusters(k):
    # k-means++ must repeat a seed; the repeated seed's empty cell keeps it,
    # nothing divides by zero, and the centers stay on the data (efcm's soft
    # weights, e^-20 from the other point, leave them about 1e-8 away).
    X = np.array([[0.0, 0.0]] * 6 + [[1.0, 1.0]] * 4)
    methods = [
        (ub.kmeans, {}),
        (ub.fcm, {}),
        (ub.efcm, {"tau": 0.1}),
        (ub.rcm, {}),
        (ub.rmcm, {"delta": 0.5}),
    ]
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        for method, options in methods:
            r = fit(method, X, k, {"seed": 0, **options})
            assert cdist(r.centers, [[0, 0], [1, 1]]).min(axis=1).max() < 1e-6, method
            if r.membership is not None:
                np.testing.assert_allclose(r.membership.sum(axis=1), 1.0)


def test_collapse_to_the_mean_follows_the_thresholds():
    # The mean is a fixed point; it attracts fcm from m* = 1 / (1 - 2 lambda_max(M))
    # on and efcm from tau* = 2 lambda_max(covariance) on ("Degenerate solutions"
    # in docs/algorithms.md). Start next to the mean on either side of each.
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (4, 8))[rng.integers(0, 4, 400)] + rng.normal(size=(400, 8))
    Y = X - X.mean(axis=0)
    directions = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    m_star = 1.0 / (1.0 - 2.0 * np.linalg.eigvalsh(directions.T @ directions / len(Y))[-1])
    tau_star = 2.0 * np.linalg.eigvalsh(Y.T @ Y / len(Y))[-1]
    near = X.mean(axis=0) + 1e-3 * rng.normal(size=(4, 8))
    radius = np.sqrt((Y**2).sum(axis=1).mean())

    def spread(r):
        return np.median(np.linalg.norm(r.centers - X.mean(axis=0), axis=1)) / radius

    assert spread(ub.fcm(X, 4, init=near, m=0.8 * m_star, tol=0, max_iter=300)) > 0.5
    assert spread(ub.fcm(X, 4, init=near, m=1.25 * m_star, tol=0, max_iter=300)) < 1e-6
    assert spread(ub.efcm(X, 4, init=near, tau=0.5 * tau_star, tol=0, max_iter=300)) > 0.5
    assert spread(ub.efcm(X, 4, init=near, tau=2.0 * tau_star, tol=0, max_iter=300)) < 1e-6


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
