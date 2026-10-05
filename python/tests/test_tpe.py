import numpy as np
import pytest

import ubukit as ub

SPACE = {
    "x": ub.uniform(-5, 5),
    "lr": ub.loguniform(1e-5, 1e-1),
    "n": ub.integer(1, 20),
    "kind": ub.choice("a", "b", "c"),
}


def objective(p):
    return (
        (p["x"] - 1.5) ** 2 + np.log10(p["lr"] / 1e-3) ** 2 + (p["n"] - 7) ** 2 + (p["kind"] != "b")
    )


def test_tpe_beats_random_search():
    tpe = [ub.minimize(objective, SPACE, 80, seed=s).best_value for s in range(5)]
    rand = [ub.minimize(objective, SPACE, 80, seed=s, n_startup=80).best_value for s in range(5)]
    assert np.median(tpe) < 0.25 * np.median(rand)


def test_samples_respect_the_space():
    tpe = ub.TPE(SPACE, seed=0, n_startup=5)
    for _ in range(40):
        p = tpe.ask()
        assert -5 <= p["x"] <= 5
        assert 1e-5 <= p["lr"] <= 1e-1
        assert isinstance(p["n"], int) and 1 <= p["n"] <= 20
        assert p["kind"] in ("a", "b", "c")
        tpe.tell(p, objective(p))
    result = tpe.result()
    assert len(result.params) == 40
    assert result.best_value == result.values.min()


def test_seed_reproducibility():
    a = ub.minimize(objective, SPACE, 30, seed=7)
    b = ub.minimize(objective, SPACE, 30, seed=7)
    assert a.params == b.params


def test_tell_accepts_external_trials():
    tpe = ub.TPE({"x": ub.uniform(0, 1)}, seed=0)
    tpe.tell({"x": 0.25}, 3.0)
    assert tpe.result().best_params == {"x": 0.25}


@pytest.mark.parametrize(
    "call",
    [
        lambda: ub.uniform(1, 1),
        lambda: ub.loguniform(0, 1),
        lambda: ub.integer(3, 2),
        lambda: ub.integer(0, 2**60),
        lambda: ub.uniform(-1e308, 1e308),
        lambda: ub.choice(),
        lambda: ub.TPE({}),
        lambda: ub.TPE({"x": (0, 1)}),
        lambda: ub.TPE({"x": ub.uniform(0, 1)}, gamma=1.0),
        lambda: ub.TPE({"x": ub.uniform(0, 1)}).tell({"y": 0.5}, 1.0),
        lambda: ub.TPE({"x": ub.uniform(0, 1)}).tell({"x": 0.5}, float("nan")),
        lambda: ub.TPE({"x": ub.uniform(0, 1)}).tell({"x": 5.0}, 1.0),
        lambda: ub.TPE({"x": ub.uniform(0, 1)}).tell({"x": "0.5"}, 1.0),
        lambda: ub.TPE({"n": ub.integer(1, 3)}).tell({"n": 1.5}, 1.0),
        lambda: ub.TPE({"k": ub.choice("a", "b")}).tell({"k": "c"}, 1.0),
        lambda: ub.TPE({"x": ub.uniform(0, 1)}).result(),
    ],
)
def test_invalid_arguments(call):
    with pytest.raises(ValueError):
        call()
