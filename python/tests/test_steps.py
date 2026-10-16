"""ubukit.steps: the fitting functions as generators of per-iteration Results."""

import pickle

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
MAPS = {"som": (((3, 3),), {"epochs": 6, "seed": 2}), "batch_som": (((3, 3),), {"epochs": 6})}
FIELDS = (
    "centers",
    "labels",
    "membership",
    "n_iter",
    "stop_reason",
    "history",
    "embedding",
    "unit_coordinates",
)
ITERATES = tuple(f for f in FIELDS if f != "stop_reason")


def assert_same(a, b, fields=FIELDS):
    for field in fields:
        x, y = getattr(a, field), getattr(b, field)
        if x is None or y is None:
            assert x is y, field
        else:
            np.testing.assert_array_equal(x, y, err_msg=field)


def finish(run):
    """Run a fitting generator to the end and return its Result."""
    while True:
        try:
            next(run)
        except StopIteration as end:
            return end.value


@pytest.mark.parametrize("name", CONVERGING)
def test_progress_results_are_the_run_stopped_there(blobs, name):
    # result() at iteration t has the iterates of the run with max_iter = t,
    # even when it is called after the run has finished (list() runs the
    # generator to the end). The stop reasons follow each run's own settings:
    # the run with max_iter = t reaches its limit at t while this one goes on
    # (None), until both meet the tolerance at the last iteration.
    X, _ = blobs
    args, options = CONVERGING[name]
    progress = list(getattr(ub.steps, name)(X, *args, **options, max_iter=1000))
    assert [p.iteration for p in progress] == list(range(1, len(progress) + 1))
    assert len(progress) > 1
    for t in {1, 2, len(progress)}:
        snapshot = progress[t - 1].result()
        stopped = getattr(ub, name)(X, *args, **options, max_iter=t)
        assert_same(snapshot, stopped, ITERATES)
        expected = ("tol", "tol") if t == len(progress) else (None, "limit")
        assert (snapshot.stop_reason, stopped.stop_reason) == expected


@pytest.mark.parametrize("name", [*CONVERGING, *MAPS])
def test_changing_a_progress_result_leaves_the_run_unchanged(blobs, name):
    # A Result shares no arrays with the run: SOM-OLP's next step reads the
    # memberships, so changing them in place used to change the run, and
    # the maps' steps read their unit coordinates.
    X, _ = blobs
    args, options = {**CONVERGING, **MAPS}[name]
    run = getattr(ub.steps, name)(X, *args, **options)
    first = next(run).result()
    first.labels[:] = 0
    if first.membership is not None:
        first.membership[:] = 1.0 / first.membership.shape[1]
    if first.unit_coordinates is not None:
        first.unit_coordinates[:] = 0.0
    assert_same(finish(run), getattr(ub, name)(X, *args, **options))


COORDINATE_MAPS = {
    "som": lambda X, R: ub.steps.som(X, R, epochs=3, seed=2),
    "som_olp": lambda X, R: ub.steps.som_olp(X, R, lam=1.0, gamma=1.0, max_iter=4),
}


@pytest.mark.parametrize("name", COORDINATE_MAPS)
def test_a_map_owns_the_unit_coordinates_it_was_given(blobs, name):
    # Changing the array passed in after the first iteration changes neither
    # the run nor the Results it gives; the run used to read that array, so
    # the embeddings of earlier iterations moved with it.
    X, _ = blobs
    R = np.random.default_rng(4).normal(size=(7, 3))
    expected = list(COORDINATE_MAPS[name](X, R.copy()))
    run = COORDINATE_MAPS[name](X, R)
    first = next(run)
    R += 100.0
    for a, b in zip([first, *run], expected, strict=True):
        assert_same(a.result(), b.result())


@pytest.mark.parametrize(
    ("name", "options"), [("som", {"epochs": 3, "seed": 2}), ("batch_som", {"epochs": 4})]
)
def test_scheduled_maps_are_complete_at_their_last_epoch(blobs, name, options):
    X, _ = blobs
    results = [p.result() for p in list(getattr(ub.steps, name)(X, (3, 3), **options))]
    assert_same(results[-1], getattr(ub, name)(X, (3, 3), **options))
    assert [r.stop_reason for r in results] == [None] * (len(results) - 1) + ["schedule"]


def test_stop_reasons(blobs):
    # "tol" before the limit, "limit" without the tolerance, "tol" when both
    # hold at the same iteration, and "tol" with no limit.
    X, _ = blobs
    free = ub.fcm(X, 4, seed=1)
    assert free.stop_reason == "tol" and free.n_iter > 2
    assert ub.fcm(X, 4, seed=1, max_iter=2).stop_reason == "limit"
    assert ub.fcm(X, 4, seed=1, max_iter=free.n_iter).stop_reason == "tol"
    assert ub.fcm(X, 4, seed=1, max_iter=None).stop_reason == "tol"


@pytest.mark.parametrize("name", ["som", "batch_som"])
def test_maps_run_their_whole_schedule(name):
    # Identical points: the prototypes never move, yet a schedule runs every
    # epoch and reports "schedule", not "tol".
    r = getattr(ub, name)(np.ones((20, 2)), (2, 2), epochs=4)
    assert (r.n_iter, r.stop_reason) == (4, "schedule")


def test_new_data_after_the_tolerance_holds_continue_the_run(blobs):
    # A Progress result records the rule that holds at its iteration; new rows
    # sent then continue the run, and the recorded result does not change.
    X, _ = blobs
    run = ub.steps.fcm(X, 3, seed=0, max_iter=None)
    p = next(run)
    while p.result().stop_reason is None:
        p = next(run)
    held = p.result()
    after = run.send(X + np.array([0.5, -0.5])).result()
    assert held.stop_reason == p.result().stop_reason == "tol"
    assert after.n_iter == held.n_iter + 1
    assert_same(p.result(), held)


@pytest.mark.parametrize(
    ("start", "reason"),
    [
        (lambda X: ub.steps.fcm(X, 3, seed=0, max_iter=1), "limit"),
        # Identical points meet the tolerance at the limit.
        (lambda X: ub.steps.fcm(np.ones((5, 2)), 2, seed=0, max_iter=1), "tol"),
        (lambda X: ub.steps.batch_som(X, (2, 2), epochs=1), "schedule"),
    ],
)
def test_the_limit_and_the_schedule_end_the_run_even_with_new_data(blobs, start, reason):
    X, _ = blobs
    run = start(X)
    first = next(run).result()
    with pytest.raises(StopIteration) as end:
        run.send(X + 1.0)
    assert end.value.value.stop_reason == first.stop_reason == reason


def test_som_olp_memberships_can_move_after_the_tolerance_holds():
    # tol = 0 says only that the prototypes did not move: on identical points
    # they stay put, yet the latent positions of the first memberships move
    # the next ones along the grid.
    X = np.zeros((10, 2))
    run = ub.steps.som_olp(
        X, [[0.0], [1.0], [2.0]], lam=0.5, gamma=1.0, init=np.zeros((3, 2)), tol=0
    )
    first = next(run).result()
    second = run.send(X).result()
    assert first.stop_reason == "tol"
    np.testing.assert_array_equal(second.centers, first.centers)
    assert np.abs(second.membership - first.membership).max() > 0.4


def assert_close(a, b, fields=("centers", "labels", "membership", "embedding", "unit_coordinates")):
    for field in fields:
        x, y = getattr(a, field), getattr(b, field)
        if x is None or y is None:
            assert x is y, field
        else:
            np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-10, err_msg=field)


@pytest.mark.parametrize("name", [*CONVERGING, *MAPS])
def test_sending_the_same_data_changes_nothing(blobs, name):
    # The data are an input of every iteration; sending them again carries the
    # prototypes and kept state over, so the iterates are the same.
    X, _ = blobs
    args, options = {**CONVERGING, **MAPS}[name]
    plain = list(getattr(ub.steps, name)(X, *args, **options))
    run = getattr(ub.steps, name)(X, *args, **options)
    fed = [next(run)] + [run.send(X) for _ in plain[1:]]
    for a, b in zip(plain, fed, strict=True):
        assert_close(a.result(), b.result())


# Maps with constant schedules: their next epoch depends only on the prototypes.
STEADY = {
    "som": (
        ((3, 3),),
        {"sigma": 1.0, "sigma_end": 1.0, "lr": 0.1, "lr_end": 0.1, "shuffle": False},
    ),
    "batch_som": (((3, 3),), {"sigma": 1.0, "sigma_end": 1.0}),
}


@pytest.mark.parametrize("name", ["kmeans", "fcm", "efcm", "rcm", "rmcm", *STEADY])
def test_new_data_continue_from_the_current_prototypes(blobs, name):
    # These methods carry only their prototypes (other state is recomputed),
    # so the iteration after new data, here moved and shifted, equals one
    # iteration on them started from where the run had got to.
    X, _ = blobs
    moved = X + np.random.default_rng(5).normal(0, 0.3, X.shape) + [2.0, -1.0]
    args, options = {**CONVERGING, **STEADY}[name]
    run = getattr(ub.steps, name)(X, *args, **options)
    prototypes = next(run).result().centers
    after = run.send(moved).result()
    one = {"epochs": 1} if name in STEADY else {"max_iter": 1}
    once = getattr(ub, name)(moved, *args, **{**options, "init": prototypes, **one})
    assert_close(after, once)


def test_som_olp_keeps_memberships_for_the_same_points(blobs):
    # The latent positions come from the previous memberships while the number
    # of rows stays the same; with another number there are none, as in the
    # first iteration.
    X, _ = blobs
    args, options = CONVERGING["som_olp"]
    run = ub.steps.som_olp(X, *args, **options)
    first = next(run).result()
    kept = run.send(X).result()
    once = ub.som_olp(X, *args, **options, init=first.centers, max_iter=1)
    assert not np.allclose(kept.centers, once.centers)  # the latent term was used
    fewer = X[:-30]
    restarted = run.send(fewer).result()
    once = ub.som_olp(fewer, *args, **options, init=kept.centers, max_iter=1)
    assert_close(restarted, once)


def test_new_data_are_checked(blobs):
    X, _ = blobs
    run = ub.steps.fcm(X, 3, seed=0)
    next(run)
    with pytest.raises(ValueError, match="features"):
        run.send(np.ones((10, 3)))
    run = ub.steps.fcm(X, 3, seed=0)
    next(run)
    with pytest.raises(ValueError, match="finite"):
        run.send(np.full_like(X, np.nan))


def test_no_iteration_limit_runs_to_convergence(blobs):
    X, _ = blobs
    assert_close(ub.fcm(X, 3, seed=0, max_iter=None), ub.fcm(X, 3, seed=0, max_iter=10_000))


def test_steps_lists_every_fitting_function():
    # Each fitting function runs its generator in ubukit.steps, and pickle
    # finds it under its public name (for multiprocessing).
    fitting = ["batch_som", "efcm", "fcm", "kmeans", "rcm", "rmcm", "som", "som_olp"]
    assert sorted(set(ub.steps.__all__) - {"apply"}) == fitting
    for name in fitting:
        function = getattr(ub, name)
        assert function.__wrapped__ is getattr(ub.steps, name)
        assert pickle.loads(pickle.dumps(function)) is function


def test_steps_validate_when_they_start():
    run = ub.steps.kmeans(np.zeros((5, 2)), 9)
    with pytest.raises(ValueError, match="k"):
        next(run)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda X: ub.kmeans(X, 3, max_iter=0), "max_iter"),
        (lambda X: ub.fcm(X, 3, tol=-1.0), "tol"),
        (lambda X: ub.efcm(X, 3, max_iter=0), "max_iter"),
        (lambda X: ub.rcm(X, 3, max_iter=0), "max_iter"),
        (lambda X: ub.rmcm(X, 3, 0.5, max_iter=0), "max_iter"),
        (lambda X: ub.som(X, (3, 3), epochs=0), "epochs"),
        (lambda X: ub.batch_som(X, (3, 3), sigma=0.0), "sigma"),
        (lambda X: ub.som_olp(X, (3, 3), lam=1.0, gamma=1.0, tol=-1.0), "tol"),
    ],
)
def test_arguments_are_checked_before_the_data(call, match):
    # The data are invalid too: a wrong argument is reported before any work
    # on the data (centering, seeding, the PCA start).
    with pytest.raises(ValueError, match=match):
        call(np.full((10, 2), np.nan))
