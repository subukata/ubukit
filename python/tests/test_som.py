import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.special import softmax, xlogy

import ubukit as ub
from ubukit.som import _grid, _pca_init


def test_batch_som_matches_dense_kernel_reference(blobs):
    X, _ = blobs
    rows, cols, epochs, s0, s1 = 4, 5, 6, 3.0, 0.5
    W = np.random.default_rng(0).normal(2, 2, (rows * cols, 2))
    R = _grid((rows, cols))
    ref = W.copy()
    for t in range(epochs):
        s = s0 * (s1 / s0) ** (t / (epochs - 1))
        bmu = cdist(X, ref, "sqeuclidean").argmin(axis=1)
        H = np.exp(-cdist(R[bmu], R, "sqeuclidean") / (2 * s * s))
        ref = H.T @ X / H.sum(axis=0)[:, None]
    r = ub.batch_som(X, (rows, cols), epochs=epochs, sigma=s0, sigma_end=s1, init=W)
    np.testing.assert_allclose(r.centers, ref, atol=1e-10)
    np.testing.assert_array_equal(r.embedding, R[r.labels])


def test_online_som_matches_reference_loop(blobs):
    X, _ = blobs
    W = np.random.default_rng(1).normal(2, 2, (9, 2))
    R = _grid((3, 3))
    ref, steps, t = W.copy(), 2 * len(X), 0
    for _ in range(2):
        for x in X:
            f = t / (steps - 1)
            s, eta = 1.0 * (0.5 / 1.0) ** f, 0.5 * (0.01 / 0.5) ** f
            b = ((ref - x) ** 2).sum(axis=1).argmin()
            h = np.exp(-((R - R[b]) ** 2).sum(axis=1) / (2 * s * s))
            ref += eta * h[:, None] * (x - ref)
            t += 1
    r = ub.som(X, (3, 3), epochs=2, sigma=1.0, init=W, shuffle=False)
    np.testing.assert_allclose(r.centers, ref, atol=1e-10)


def test_som_seed_and_shuffle(blobs):
    X, _ = blobs
    a = ub.som(X, (3, 3), epochs=1, seed=3)
    b = ub.som(X, (3, 3), epochs=1, seed=3)
    np.testing.assert_array_equal(a.centers, b.centers)


def test_som_olp_matches_original_update_order(blobs):
    X, _ = blobs
    lam, gamma, n_iter = 0.7, 0.5, 12
    R = _grid((3, 4))
    W0 = np.random.default_rng(2).normal(2, 2, (12, 2))
    # Original order: P0 from W0, then V/W from old P, then new P.
    P = softmax(-cdist(X, W0, "sqeuclidean") / lam, axis=1)
    for _ in range(n_iter - 1):
        V, W = P @ R, P.T @ X / P.sum(axis=0)[:, None]
        P = softmax(
            -(cdist(X, W, "sqeuclidean") + gamma * cdist(V, R, "sqeuclidean")) / lam, axis=1
        )
    r = ub.som_olp(X, (3, 4), lam=lam, gamma=gamma, init=W0, max_iter=n_iter, tol=0)
    np.testing.assert_allclose(r.membership, P, atol=1e-9)
    np.testing.assert_allclose(r.embedding, P @ R, atol=1e-9)
    np.testing.assert_allclose(r.centers, P.T @ X / P.sum(axis=0)[:, None], atol=1e-9)


def test_som_olp_history_is_the_objective(blobs):
    # Two iterations from given prototypes, against the definition (the
    # history uses an identity that holds at the memberships of the costs).
    X, _ = blobs
    lam, gamma = 0.7, 0.5
    R = _grid((3, 4))
    W0 = np.random.default_rng(2).normal(2, 2, (12, 2))
    cost = cdist(X, W0, "sqeuclidean")
    P = softmax(-cost / lam, axis=1)
    expected = [np.sum(P * cost) + lam * np.sum(xlogy(P, P))]
    W = P.T @ X / P.sum(axis=0)[:, None]
    cost = cdist(X, W, "sqeuclidean") + gamma * cdist(P @ R, R, "sqeuclidean")
    P = softmax(-cost / lam, axis=1)
    expected.append(np.sum(P * cost) + lam * np.sum(xlogy(P, P)))
    r = ub.som_olp(X, (3, 4), lam=lam, gamma=gamma, init=W0, max_iter=2, tol=0)
    np.testing.assert_allclose(r.history, expected, rtol=1e-12)


def test_som_olp_accepts_arbitrary_unit_coordinates(blobs):
    X, _ = blobs
    R = np.random.default_rng(0).uniform(size=(7, 3))
    r = ub.som_olp(X, R, lam=1.0, gamma=1.0)
    assert r.embedding.shape == (300, 3)
    np.testing.assert_allclose(r.membership.sum(axis=1), 1.0)
    np.testing.assert_array_equal(ub.som_olp(X, R.tolist(), lam=1.0, gamma=1.0).centers, r.centers)


def test_pca_init_spans_principal_axes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3)) * [5.0, 2.0, 0.1]
    X -= X.mean(axis=0)
    W = _pca_init(X, _grid((3, 3)), 2.0)
    assert np.abs(W[:, 2]).max() < 0.1
    assert np.ptp(W[:, 0]) > np.ptp(W[:, 1]) > 0
    np.testing.assert_allclose(W.mean(axis=0), 0, atol=1e-12)


def test_pca_init_orientation_does_not_depend_on_rounding():
    # Standardized 2-D data have axes (1, +-1)/sqrt(2) whose components tie in
    # magnitude; reordering the rows changes only the rounding.
    R = _grid((3, 4))
    for seed in range(30):
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(200, 2)) @ [[1.0, 0.0], [0.6, 0.8]]
        X = (Z - Z.mean(axis=0)) / Z.std(axis=0)
        W = _pca_init(X - X.mean(axis=0), R, 2.0)
        P = X[rng.permutation(200)]
        np.testing.assert_allclose(_pca_init(P - P.mean(axis=0), R, 2.0), W, atol=1e-9)


def test_pca_init_agrees_between_gram_matrices():
    # 20 x 50 data use X X^T; three stacked copies (60 x 50) have the same
    # covariance and use X^T X.
    X = np.random.default_rng(1).normal(size=(20, 50)) * np.linspace(3, 0.1, 50)
    X -= X.mean(axis=0)
    R = _grid((3, 4))
    np.testing.assert_allclose(
        _pca_init(X, R, 2.0), _pca_init(np.vstack([X, X, X]), R, 2.0), atol=1e-10
    )


def test_tiny_values_take_the_ordinary_path():
    X = np.random.default_rng(0).normal(size=(300, 3))
    X[0, 0] = 1e-200
    r = ub.batch_som(X, (4, 4), epochs=3)
    assert np.isfinite(r.centers).all()


@pytest.mark.parametrize(
    "call",
    [
        lambda X: ub.batch_som(X, (0, 3)),
        lambda X: ub.batch_som(X, np.zeros((4, 2))),
        lambda X: ub.som(X, (3, 3), lr=1.5),
        lambda X: ub.som(X, (3, 3), sigma=0),
        lambda X: ub.som_olp(X, (3, 3), lam=0, gamma=1),
        lambda X: ub.som_olp(X, (3, 3), lam=1, gamma=-1),
        lambda X: ub.som_olp(X, (3, 3), lam=1, gamma=1, pca_scale=0.0),
        lambda X: ub.batch_som(X, (3, 3), init=np.zeros((4, 2))),
    ],
)
def test_invalid_arguments(blobs, call):
    X, _ = blobs
    with pytest.raises(ValueError):
        call(X)
