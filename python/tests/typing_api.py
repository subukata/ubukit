"""How typed code calls UbuKit: checked by mypy (see pyproject.toml), never run.

Each ``type: ignore`` marks a call that must be rejected; mypy reports an
ignore that is not needed, so a part of the API that loses its types fails.
"""

import numpy as np

import ubukit as ub

X = np.zeros((10, 2))


def fit() -> ub.Result:
    return ub.fcm(X, 3, m=2.0, seed=0)


def follow_moving_data() -> ub.Result:
    run = ub.steps.som_olp(X, (3, 3), lam=0.5, gamma=1.0, max_iter=None)
    p: ub.Progress = next(run)
    p = run.send(X + 1.0)
    return p.result()


def wrong_arguments() -> None:
    ub.fcm(X, "3")  # type: ignore[arg-type]
    ub.steps.fcm(X, "3")  # type: ignore[arg-type]
    ub.steps.som_olp(X, (3, 3), lam=0.5)  # type: ignore[call-arg]
