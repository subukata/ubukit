"""engine="numba": the compiled kernels give the results of the NumPy reference."""

import sys

import numpy as np
import pytest

import ubukit as ub


@pytest.fixture
def numba():
    return pytest.importorskip("numba")


def grid_ties(seed):
    """Duplicated inputs and a SOM-like integer embedding: many exact distance ties."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(150, 6))
    X[10:20] = X[10]
    return X, rng.integers(0, 4, size=(150, 2)).astype(float)


def distinct_sizes(m):
    """Labels whose clusters have the sizes 1..m, against a permutation of themselves."""
    a = np.repeat(np.arange(m), np.arange(1, m + 1))
    return a, np.random.default_rng(m).permutation(a)


@pytest.mark.parametrize("sigma", [None, 1e-200])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("grid", [(4, 5), "coordinates"])
def test_som_kernel_reproduces_the_reference(numba, blobs, shuffle, grid, sigma):
    # sigma = 1e-200: a width whose square underflows, the winner-only limit.
    X, _ = blobs
    if grid == "coordinates":
        grid = np.random.default_rng(0).uniform(size=(7, 3))
    options = {"epochs": 3, "seed": 4, "shuffle": shuffle, "sigma": sigma}
    if sigma is not None:
        options["sigma_end"] = sigma
    ref = ub.som(X, grid, **options)
    new = ub.som(X, grid, engine="numba", **options)
    np.testing.assert_allclose(new.centers, ref.centers, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(new.labels, ref.labels)


@pytest.mark.parametrize("k", [1, 5, 12])
def test_trustworthiness_kernel_reproduces_the_reference(numba, k):
    X, Y = grid_ties(k)
    for metric in (ub.trustworthiness, ub.continuity):
        assert metric(X, Y, k, engine="numba") == metric(X, Y, k)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (np.arange(50) % 4, np.arange(50) % 3),
        # Clusters large enough that the expected-MI sums are windowed.
        tuple(np.random.default_rng(1).integers(0, k, 20000) for k in (3, 4)),
        distinct_sizes(60),
    ],
)
def test_ami_kernel_reproduces_the_reference(numba, a, b):
    for average in ("arithmetic", "geometric", "min", "max"):
        expected = ub.ami(a, b, average=average)
        assert ub.ami(a, b, average=average, engine="numba") == pytest.approx(expected, abs=1e-12)


def test_parallel_kernels_do_not_depend_on_the_thread_count(numba):
    X, Y = grid_ties(0)
    a, b = distinct_sizes(60)

    def results():
        return ub.trustworthiness(X, Y, 5, engine="numba"), ub.ami(a, b, engine="numba")

    threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = results()
    finally:
        numba.set_num_threads(threads)
    assert results() == one


@pytest.mark.parametrize(
    "call",
    [
        lambda X: ub.som(X, (3, 3), engine="fast"),
        lambda X: ub.ami([0, 1], [0, 1], engine="cython"),
        lambda X: ub.continuity(X, X, 2, engine=None),
    ],
)
def test_engine_is_checked(blobs, call):
    X, _ = blobs
    with pytest.raises(ValueError, match="engine"):
        call(X)


def test_without_numba_the_error_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)  # makes "import numba" fail
    monkeypatch.delitem(sys.modules, "ubukit._numba", raising=False)
    monkeypatch.delattr(ub, "_numba", raising=False)
    with pytest.raises(ImportError, match=r"pip install 'ubukit\[numba\]'"):
        ub.ami([0, 1], [0, 1], engine="numba")
