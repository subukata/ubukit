"""ubukit.steps: the fitting functions as generators of per-iteration Results."""

import numpy as np
import pytest

import ubukit as ub

# Four clusters for three blobs, so that every method takes several iterations.
CONVERGING = {
    "kmeans": ((4,), {"seed": 1}),
    "fcm": ((4,), {"seed": 1}),
    "efcm": ((4,), {"seed": 1, "tau": 0.5}),
    "rcm": ((4,), {"seed": 1}),
    "rmcm": ((4, 0.5), {"seed": 1}),
    "som_olp": (((3, 3),), {"lam": 0.5, "gamma": 1.0}),
}
FIELDS = ("centers", "labels", "membership", "n_iter", "converged", "history", "embedding")


def assert_same(a, b):
    for field in FIELDS:
        x, y = getattr(a, field), getattr(b, field)
        if x is None or y is None:
            assert x is y, field
        else:
            np.testing.assert_array_equal(x, y, err_msg=field)


@pytest.mark.parametrize("name", CONVERGING)
def test_progress_results_are_the_run_stopped_there(blobs, name):
    # result() at iteration t equals the run with max_iter = t, even when it is
    # called after the run has finished (list() runs the generator to the end).
    X, _ = blobs
    args, options = CONVERGING[name]
    progress = list(getattr(ub.steps, name)(X, *args, **options, max_iter=1000))
    assert [p.iteration for p in progress] == list(range(1, len(progress) + 1))
    assert len(progress) > 1
    for t in {1, 2, len(progress)}:
        stopped = getattr(ub, name)(X, *args, **options, max_iter=t)
        assert_same(progress[t - 1].result(), stopped)


@pytest.mark.parametrize(
    ("name", "options"), [("som", {"epochs": 3, "seed": 2}), ("batch_som", {"epochs": 4})]
)
def test_scheduled_maps_are_complete_at_their_last_epoch(blobs, name, options):
    X, _ = blobs
    results = [p.result() for p in list(getattr(ub.steps, name)(X, (3, 3), **options))]
    assert_same(results[-1], getattr(ub, name)(X, (3, 3), **options))
    assert [r.converged for r in results] == [False] * (len(results) - 1) + [True]


def test_steps_lists_every_fitting_function():
    fitting = ["batch_som", "efcm", "fcm", "kmeans", "rcm", "rmcm", "som", "som_olp"]
    assert sorted(vars(ub.steps)) == fitting


def test_steps_validate_when_they_start():
    run = ub.steps.kmeans(np.zeros((5, 2)), 9)
    with pytest.raises(ValueError, match="k"):
        next(run)
