import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.manifold import trustworthiness as sk_trustworthiness

import ubukit as ub


@pytest.mark.parametrize(("n", "ka", "kb"), [(10, 2, 3), (200, 5, 7), (1000, 40, 3), (300, 300, 4)])
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


def test_invalid_arguments():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        ub.trustworthiness(X, X, 5)
    with pytest.raises(ValueError):
        ub.trustworthiness(X, X[:9], 2)
    with pytest.raises(ValueError, match="scale"):
        ub.trustworthiness(X + 1e200, X, 2)
    with pytest.raises(ValueError):
        ub.ari([0, 1], [0])
    with pytest.raises(ValueError):
        ub.ami([0, 1], [0, 1], average="median")
