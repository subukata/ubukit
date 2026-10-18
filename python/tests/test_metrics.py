import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.manifold import trustworthiness as sk_trustworthiness

import ubukit as ub


@pytest.mark.parametrize(
    ("n", "ka", "kb"),
    # The last case has clusters large enough that the expected-MI sums are windowed.
    [(10, 2, 3), (200, 5, 7), (1000, 40, 3), (300, 300, 4), (20000, 3, 4)],
)
def test_ari_ami_match_sklearn(n, ka, kb):
    rng = np.random.default_rng(n + ka)
    a, b = rng.integers(0, ka, n), rng.integers(0, kb, n)
    assert ub.ari(a, b) == pytest.approx(skm.adjusted_rand_score(a, b), abs=1e-12)
    for average in ("arithmetic", "geometric", "min", "max"):
        expected = skm.adjusted_mutual_info_score(a, b, average_method=average)
        assert ub.ami(a, b, average=average) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([0, 0, 1, 1], [5, 5, 3, 3]),
        ([0, 0, 0], [1, 1, 1]),
        ([0, 1, 2, 3], [0, 0, 0, 0]),
        ([0, 1, 2, 3], [3, 2, 1, 0]),
        (["x", "y", "y"], ["a", "a", "b"]),
        ([], []),
    ],
)
def test_ari_ami_edge_cases_match_sklearn(a, b):
    assert ub.ari(a, b) == pytest.approx(skm.adjusted_rand_score(a, b))
    assert ub.ami(a, b) == pytest.approx(skm.adjusted_mutual_info_score(a, b), abs=1e-12)


@pytest.mark.parametrize("k", [1, 5, 12])
def test_trustworthiness_and_continuity_match_sklearn(k):
    rng = np.random.default_rng(k)
    X = rng.normal(size=(150, 6))
    Y = X[:, :2] + 0.3 * rng.normal(size=(150, 2))
    assert ub.trustworthiness(X, Y, k) == pytest.approx(sk_trustworthiness(X, Y, n_neighbors=k))
    assert ub.continuity(X, Y, k) == pytest.approx(sk_trustworthiness(Y, X, n_neighbors=k))


def reference_trustworthiness(X, Y, k):
    """Definition with exact (distance, index) ordering."""
    n = len(X)
    dx = ((X[:, None] - X[None]) ** 2).sum(axis=2)
    dy = ((Y[:, None] - Y[None]) ** 2).sum(axis=2)
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    penalty = 0
    for i in range(n):
        rank = np.empty(n, dtype=int)
        rank[np.argsort(dx[i], kind="stable")] = np.arange(1, n + 1)
        neighbors = np.argsort(dy[i], kind="stable")[:k]
        penalty += np.maximum(rank[neighbors] - k, 0).sum()
    return 1 - 2 * penalty / (n * k * (2 * n - 3 * k - 1))


def test_trustworthiness_with_grid_ties():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    X[10:20] = X[10]  # duplicated inputs
    Y = rng.integers(0, 4, size=(120, 2)).astype(float)  # SOM-like grid embedding
    for k in (1, 3, 7):
        assert ub.trustworthiness(X, Y, k) == pytest.approx(reference_trustworthiness(X, Y, k))
        assert ub.continuity(X, Y, k) == pytest.approx(reference_trustworthiness(Y, X, k))


def ties():
    """Duplicated inputs and an integer grid embedding: exact distance ties in both spaces."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4))
    X[10:20] = X[10]
    return X, rng.integers(0, 4, size=(120, 2)).astype(float)


@pytest.mark.parametrize("metric", [ub.trustworthiness, ub.continuity])
def test_several_k_give_the_values_of_separate_calls_in_their_order(metric):
    X, Y = ties()
    ks = [7, 2, 4]
    values = metric(X, Y, ks)
    assert isinstance(values, np.ndarray)
    np.testing.assert_array_equal(values, [metric(X, Y, k) for k in ks])


@pytest.mark.parametrize("metric", [ub.trustworthiness, ub.continuity])
def test_a_neighborhood_gives_the_values_of_its_points(metric):
    X, Y = ties()
    nX, nY = ub.Neighborhood(X), ub.Neighborhood(Y)
    expected = metric(X, Y, [5, 1, 9])
    for A, B in ((nX, Y), (X, nY), (nX, nY)):
        np.testing.assert_array_equal(metric(A, B, [5, 1, 9]), expected)
    assert metric(nX, Y) == metric(X, Y)


def test_a_neighborhood_keeps_the_distances_it_was_built_with():
    X, Y = ties()
    reference = ub.Neighborhood(X)  # X is float64 already, so it is not copied on input
    expected = ub.trustworthiness(X, Y, 5)
    X[:] = 0.0
    assert ub.trustworthiness(reference, Y, 5) == expected
    assert repr(reference) == "Neighborhood(n=120)"


def test_invalid_arguments():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        ub.trustworthiness(X, X, 5)
    with pytest.raises(ValueError):
        ub.trustworthiness(X, X[:9], 2)
    with pytest.raises(ValueError, match="scale"):
        ub.trustworthiness(X + 1e200, X, 2)
    Z = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValueError, match="scale"):
        ub.trustworthiness(Z, Z * 1e-170, 2)
    with pytest.raises(ValueError, match="same number of rows"):
        ub.trustworthiness(ub.Neighborhood(Z), Z[:9], 2)
    with pytest.raises(ValueError, match="non-empty"):
        ub.trustworthiness(Z, Z, [])
    with pytest.raises(ValueError, match="n / 2"):
        ub.continuity(Z, Z, [2, 5])
    with pytest.raises(ValueError, match="k must be an integer"):
        ub.trustworthiness(Z, Z, [2, 2.5])
    with pytest.raises(TypeError):
        ub.fcm(ub.Neighborhood(Z), 2)  # a Neighborhood is not data
    with pytest.raises(ValueError):
        ub.ari([0, 1], [0])
    with pytest.raises(ValueError):
        ub.ami(5, 7)
    with pytest.raises(ValueError):
        ub.ami([0, 1], [0, 1], average="median")
    # NumPy's own error named neither the labels nor the cause.
    with pytest.raises(TypeError, match=r"^labels must be values of one kind"):
        ub.ari([0, None], [0, 1])


@pytest.mark.parametrize(
    "mixed", [[0, "0"], [True, 1], [np.True_, 1], np.array([0, "0"], dtype=object)]
)
@pytest.mark.parametrize("metric", [ub.ari, ub.ami])
def test_a_labeling_holds_values_of_one_kind(metric, mixed):
    # NumPy made [0, "0"] two equal strings and [True, 1] two equal integers,
    # which JavaScript keeps apart: ARI 0 here and 1 there.
    message = r"^labels must be values of one kind: numbers, strings or booleans$"
    with pytest.raises(TypeError, match=message):
        metric(mixed, [0, 1])
    with pytest.raises(TypeError, match=message):
        metric([0, 1], mixed)


def test_labels_of_one_kind_are_taken_as_before():
    assert ub.ari([1, 1.0, 2], ["a", "a", "b"]) == 1.0
    assert ub.ari(np.array([True, True, False]), [np.int64(3), 3, 4]) == 1.0
