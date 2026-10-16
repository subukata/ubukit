"""How typed code calls UbuKit: checked by mypy (see pyproject.toml), never run.

Each ``type: ignore`` marks a call that must be rejected; mypy reports an
ignore that is not needed, so a part of the API that loses its types fails.
"""

from typing import Literal

import numpy as np

import ubukit as ub

X = np.zeros((10, 2))


def unit_of_each_point() -> np.ndarray:
    m = ub.batch_som(X, (2, 3))
    coordinates = m.unit_coordinates  # None for clusterings
    if coordinates is None:
        raise ValueError("a map has unit coordinates")
    return coordinates[m.labels]


def place_new_points() -> ub.Result:
    m = ub.som_olp(X, (3, 3), lam=0.5, gamma=1.0)
    run = ub.steps.apply.fcm(X, m.centers, m=1.5)
    p: ub.Progress = next(run)
    p.result()
    C = X[:2]
    ub.apply.kmeans(X, C)
    ub.apply.efcm(X, C, tau=1.0)
    ub.apply.rcm(X, C, alpha=1.2, beta=0.1, p=2.0)
    ub.apply.rmcm(X, C, 0.5, max_edges=100)
    ub.apply.som(X, m.centers, (3, 3))
    ub.apply.batch_som(X, m.centers, np.zeros((9, 2)))
    ub.apply.som_olp(X, m.centers, (3, 3), lam=0.5, gamma=1.0, max_iter=None, tol=0.0)
    return ub.apply.som_olp(X + 1.0, m.centers, (3, 3), lam=0.5, gamma=1.0)


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
    _ = ub.batch_som(X, (2, 3)).unit_coordinates[0]  # type: ignore[index]  # may be None
    ub.apply.fcm(X, X, k=3)  # type: ignore[call-arg]  # the centers give k
    ub.apply.som_olp(X, X, (3, 3), lam=0.5)  # type: ignore[call-arg]
    ub.trustworthiness(X, X, "3")  # type: ignore[arg-type]
    ub.fcm(ub.Neighborhood(X), 3)  # type: ignore[arg-type]
    ub.fcm(X, "3")  # type: ignore[arg-type]
    ub.steps.fcm(X, "3")  # type: ignore[arg-type]
    ub.steps.som_olp(X, (3, 3), lam=0.5)  # type: ignore[call-arg]
