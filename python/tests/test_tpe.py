from types import MappingProxyType

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


def test_a_space_may_be_any_mapping():
    # The type of a space is a Mapping, so a read-only mapping works as a dict does.
    proxy = ub.minimize(objective, MappingProxyType(SPACE), 15, seed=3)
    assert proxy.params == ub.minimize(objective, SPACE, 15, seed=3).params


def test_minimize_passes_only_the_options_of_tpe():
    with pytest.raises(TypeError, match="n_startpu"):
        ub.minimize(objective, SPACE, 5, n_startpu=2)


def test_the_settings_are_read_only():
    # They were plain attributes: n_candidates = 0 set later skipped the check
    # and made the next proposal fail.
    tpe = ub.TPE(SPACE, n_startup=3, n_candidates=5, gamma=0.2)
    assert (tpe.n_startup, tpe.n_candidates, tpe.gamma) == (3, 5, 0.2)
    for name in ("n_startup", "n_candidates", "gamma"):
        with pytest.raises(AttributeError):
            setattr(tpe, name, 0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: ub.TPE({"x": ub.uniform(0, 1)}, seed=True),
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


def test_a_space_too_narrow_to_search_is_refused():
    # Its logarithms coincide, so the first ask from the model divided by zero.
    with pytest.raises(ValueError, match="too narrow to search on a log scale"):
        ub.loguniform(1000, 1000.0000000000001)
    with pytest.raises(ValueError, match=r"^value must be a finite number$"):
        ub.TPE({"x": ub.uniform(0, 1)}).tell({"x": 0.5}, float("inf"))


def test_a_boolean_is_not_a_numeric_option():
    # True == 1 in Python, so True was taken for the option 1; JavaScript compares with ===.
    tpe = ub.TPE({"k": ub.choice(1, 2), "flag": ub.choice(False, True)}, seed=0)
    with pytest.raises(ValueError, match="not an option"):
        tpe.tell({"k": True, "flag": True}, 1.0)
    with pytest.raises(ValueError, match="not an option"):
        tpe.tell({"k": 1, "flag": 1}, 1.0)
    tpe.tell({"k": 1, "flag": True}, 1.0)
    assert tpe.result().best_params == {"k": 1, "flag": True}
