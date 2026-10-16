"""ubukit.apply: trained prototypes applied to new data, the prototypes fixed."""

import pickle

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.special import softmax

import ubukit as ub

CENTERS = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]])
CLUSTERINGS = {
    "kmeans": ((), {}),
    "fcm": ((), {"m": 1.5}),
    "efcm": ((), {"tau": 0.7}),
    "rcm": ((), {"alpha": 1.3, "beta": 0.2}),
    "rmcm": ((0.8,), {}),
}


@pytest.mark.parametrize("name", CLUSTERINGS)
def test_a_clustering_applied_assigns_as_its_training_does(blobs, name):
    # The first iteration of training from the same centers assigns the same
    # memberships and labels with the same objective; applying never moves
    # the centers.
    X, _ = blobs
    args, options = CLUSTERINGS[name]
    applied = getattr(ub.apply, name)(X, CENTERS, *args, **options)
    first = getattr(ub, name)(X, 3, *args, init=CENTERS, max_iter=1, **options)
    np.testing.assert_array_equal(applied.labels, first.labels)
    if first.membership is None:
        assert applied.membership is None
    else:
        np.testing.assert_array_equal(applied.membership, first.membership)
    np.testing.assert_allclose(applied.history, first.history, rtol=1e-12)
    np.testing.assert_array_equal(applied.centers, CENTERS)
    assert (applied.n_iter, applied.stop_reason) == (1, "schedule")
    assert applied.embedding is None and applied.unit_coordinates is None


def test_the_centers_stay_exactly_as_given():
    # Centered on data at 1e10 and restored, the center 0.1 would come back as
    # 0.10000038146972656, and new data sent to the run would add up such changes.
    C = np.array([[0.1], [5.0]])
    np.testing.assert_array_equal(ub.apply.fcm([[1e10]], C).centers, C)
    run = ub.steps.apply.som_olp([[1e10]], C, [[0.0], [1.0]], lam=1.0, gamma=1.0, max_iter=None)
    results = [next(run).result()]
    for t in range(2, 5):
        results.append(run.send([[t * 1e10]]).result())
    for r in results:
        np.testing.assert_array_equal(r.centers, C)


@pytest.mark.parametrize("name", ["som", "batch_som"])
def test_a_map_applied_gives_the_best_matching_units(blobs, name):
    X, _ = blobs
    m = ub.batch_som(X, (3, 4), epochs=5)
    r = getattr(ub.apply, name)(X + 0.3, m.centers, m.unit_coordinates)
    np.testing.assert_array_equal(r.labels, cdist(X + 0.3, m.centers, "sqeuclidean").argmin(axis=1))
    np.testing.assert_array_equal(r.embedding, m.unit_coordinates[r.labels])
    np.testing.assert_array_equal(r.unit_coordinates, m.unit_coordinates)
    assert r.membership is None and (r.n_iter, r.stop_reason) == (1, "schedule")
    # The grid may also be given as (rows, cols).
    np.testing.assert_array_equal(getattr(ub.apply, name)(X, m.centers, (3, 4)).labels, m.labels)


def olp_reference(X, W, R, lam, gamma, tol, max_iter):
    """The memberships from the distances alone, then with the latent positions."""
    D = cdist(X, W, "sqeuclidean")
    P = softmax(-D / lam, axis=1)
    for t in range(2, max_iter + 1):
        new = softmax(-(D + gamma * cdist(P @ R, R, "sqeuclidean")) / lam, axis=1)
        done = np.abs(new - P).max() <= tol
        P = new
        if done:
            return P, t
    return P, max_iter


def test_som_olp_applied_iterates_the_memberships_with_the_prototypes_fixed(blobs):
    X, _ = blobs
    m = ub.som_olp(X, (3, 4), lam=0.5, gamma=1.0)
    new = X[::7] + 0.4
    r = ub.apply.som_olp(new, m.centers, m.unit_coordinates, lam=0.5, gamma=1.0)
    P, n = olp_reference(new, m.centers, m.unit_coordinates, 0.5, 1.0, 1e-6, 1000)
    assert (r.n_iter, r.stop_reason) == (n, "tol") and n > 2
    np.testing.assert_allclose(r.membership, P, atol=1e-9)
    np.testing.assert_allclose(r.embedding, P @ m.unit_coordinates, atol=1e-9)
    np.testing.assert_array_equal(r.centers, m.centers)
    # From the second iteration on, the objective at each assignment never increases.
    assert np.all(np.diff(r.history[1:]) <= 1e-9 * np.abs(r.history[1:-1]))
    # One iteration: the memberships of the distances alone, and the limit.
    once = ub.apply.som_olp(new, m.centers, m.unit_coordinates, lam=0.5, gamma=1.0, max_iter=1)
    np.testing.assert_allclose(
        once.membership, softmax(-cdist(new, m.centers, "sqeuclidean") / 0.5, axis=1)
    )
    assert once.stop_reason == "limit"


def test_som_olp_applied_places_each_point_on_its_own(blobs):
    # Each point's memberships depend on it alone, so applying to part of
    # the points gives their rows of applying to all (to the tolerance).
    X, _ = blobs
    m = ub.som_olp(X, (3, 4), lam=0.5, gamma=1.0)
    options = {"lam": 0.5, "gamma": 1.0, "tol": 1e-12}
    whole = ub.apply.som_olp(X, m.centers, m.unit_coordinates, **options)
    part = ub.apply.som_olp(X[:40], m.centers, m.unit_coordinates, **options)
    np.testing.assert_allclose(part.membership, whole.membership[:40], atol=1e-9)


def test_rmcm_applied_depends_on_the_points_given_with_it():
    # The neighborhoods are those within the data given: the point at 4 is
    # all in the first cluster beside another at 4, half in each beside 6,
    # and given alone it gets the hard assignment.
    C = np.array([[0.0], [10.0]])
    np.testing.assert_array_equal(ub.apply.rmcm([[4.0], [4.0]], C, 3.0).membership[0], [1, 0])
    np.testing.assert_array_equal(ub.apply.rmcm([[4.0], [6.0]], C, 3.0).membership[0], [0.5, 0.5])
    np.testing.assert_array_equal(ub.apply.rmcm([[4.0]], C, 3.0).membership, [[1, 0]])


def test_fewer_points_than_centers(blobs):
    r = ub.apply.fcm([[2.0, 1.0]], CENTERS)
    assert r.membership.shape == (1, 3) and r.labels.tolist() == [0]


def test_one_assignment_ends_the_run_and_som_olp_takes_new_rows(blobs):
    X, _ = blobs
    run = ub.steps.apply.fcm(X, CENTERS)
    next(run)
    with pytest.raises(StopIteration) as end:
        run.send(X + 1.0)  # one assignment: new rows need a new application
    assert end.value.value.stop_reason == "schedule"
    m = ub.som_olp(X, (3, 3), lam=0.5, gamma=1.0)
    run = ub.steps.apply.som_olp(X, m.centers, (3, 3), lam=0.5, gamma=1.0, max_iter=None)
    p = next(run)
    while p.result().stop_reason is None:
        p = next(run)
    moved = run.send(X + 0.2).result()  # the same points, moved: the run goes on
    assert moved.n_iter == p.iteration + 1


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda X: ub.apply.fcm(X, np.ones((3, 3))), ValueError, "columns"),
        (lambda X: ub.apply.som(X, np.ones((5, 2)), (2, 2)), ValueError, "4 rows"),
        (lambda X: ub.apply.fcm(X, "k-means++"), TypeError, "numbers"),
        (lambda X: ub.apply.fcm(X, CENTERS, m=1.0), ValueError, "^m must"),
        # Arguments are checked before the data, which are invalid too.
        (lambda X: ub.apply.fcm(X * np.nan, CENTERS * np.nan, m=1.0), ValueError, "^m must"),
        (
            lambda X: ub.apply.som_olp(X * np.nan, np.ones((4, 2)), (2, 2), lam=0.0, gamma=1.0),
            ValueError,
            "^lam must",
        ),
        (
            lambda X: ub.apply.som_olp(X, np.ones((4, 2)), (2, 2), lam=0.0, gamma=1.0),
            ValueError,
            "lam",
        ),
        (lambda X: ub.apply.fcm(X, CENTERS, max_iter=3), TypeError, "max_iter"),
        (lambda X: ub.apply.som(X, np.ones((4, 2)), (2, 2), epochs=3), TypeError, "epochs"),
    ],
)
def test_invalid_arguments(blobs, call, error, match):
    X, _ = blobs
    with pytest.raises(error, match=match):
        call(X)


def test_apply_lists_every_method_and_pickles():
    names = ["batch_som", "efcm", "fcm", "kmeans", "rcm", "rmcm", "som", "som_olp"]
    assert sorted(ub.apply.__all__) == sorted(ub.steps.apply.__all__) == names
    for name in names:
        function = getattr(ub.apply, name)
        assert function.__wrapped__ is getattr(ub.steps.apply, name)
        assert pickle.loads(pickle.dumps(function)) is function
