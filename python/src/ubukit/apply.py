"""Trained prototypes applied to new data, the prototypes fixed.

Each function takes the data and the prototypes (``centers`` of a Result),
with the parameters the prototypes were trained with, and returns the Result
of the data under them: labels, memberships, embedding, and
``centers`` exactly as given. Training functions update their prototypes;
these never do. SOM-OLP iterates its memberships until they settle; the
other methods assign once (``stop_reason == "schedule"``).

    m = ub.som_olp(X, (10, 10), lam=0.5, gamma=1.0)
    new = ub.apply.som_olp(X_new, m.centers, m.unit_coordinates, lam=0.5, gamma=1.0)
    new.embedding  # the new points on the map

``ubukit.steps.apply`` holds the same functions as generators.
"""

from . import steps as _steps
from ._core import stepwise as _stepwise

kmeans = _stepwise(_steps.apply.kmeans, "kmeans", __name__)
fcm = _stepwise(_steps.apply.fcm, "fcm", __name__)
efcm = _stepwise(_steps.apply.efcm, "efcm", __name__)
rcm = _stepwise(_steps.apply.rcm, "rcm", __name__)
rmcm = _stepwise(_steps.apply.rmcm, "rmcm", __name__)
som = _stepwise(_steps.apply.som, "som", __name__)
batch_som = _stepwise(_steps.apply.batch_som, "batch_som", __name__)
som_olp = _stepwise(_steps.apply.som_olp, "som_olp", __name__)

__all__ = ["batch_som", "efcm", "fcm", "kmeans", "rcm", "rmcm", "som", "som_olp"]
