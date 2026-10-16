"""How typed code calls UbuKit: checked by mypy (see pyproject.toml), never run.

Each ``type: ignore`` marks a call that must be rejected; mypy reports an
ignore that is not needed, so a part of the API that loses its types fails.
"""

from typing import Literal

import numpy as np

import ubukit as ub

X = np.zeros((10, 2))


def stopped_by_tolerance() -> bool:
    reason: Literal["tol", "schedule", "limit"] | None = ub.fcm(X, 3).stop_reason
    return reason == "tol"


def fit() -> ub.Result:
    return ub.fcm(X, 3, m=2.0, seed=0)


def follow_moving_data() -> ub.Result:
    run = ub.steps.som_olp(X, (3, 3), lam=0.5, gamma=1.0, max_iter=None)
    p: ub.Progress = next(run)
    p = run.send(X + 1.0)
    return p.result()


def score_many_layouts() -> float:
    # One Neighborhood of X scores many embeddings, at one k or several.
    reference = ub.Neighborhood(X)
    one: float = ub.trustworthiness(reference, X + 1.0, 3)
    several: np.ndarray = ub.continuity(reference, X + 1.0, [2, 3])
    return one + float(several.sum())


def wrong_arguments() -> None:
    _ = ub.fcm(X, 3).converged  # type: ignore[attr-defined]  # replaced by stop_reason
    ub.trustworthiness(X, X, "3")  # type: ignore[arg-type]
    ub.fcm(ub.Neighborhood(X), 3)  # type: ignore[arg-type]
    ub.fcm(X, "3")  # type: ignore[arg-type]
    ub.steps.fcm(X, "3")  # type: ignore[arg-type]
    ub.steps.som_olp(X, (3, 3), lam=0.5)  # type: ignore[call-arg]
