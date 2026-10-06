"""UbuKit: fuzzy, rough and classical clustering, self-organizing maps,
evaluation metrics and TPE search on NumPy and SciPy.

Every fitting function runs to the end and returns a Result. ``steps`` holds
the same functions as generators that yield a Progress after every iteration,
whose ``result()`` is the Result as if the run had stopped there:

    for p in ub.steps.som_olp(X, (10, 10), lam=0.5, gamma=1.0):
        draw(p.result())
"""

from . import steps
from ._core import Progress, Result, stepwise
from .metrics import ami, ari, continuity, trustworthiness
from .tpe import TPE, TPEResult, choice, integer, loguniform, minimize, uniform

# The fitting functions: each runs its generator in ``steps`` to the end.
kmeans = stepwise(steps.kmeans)
fcm = stepwise(steps.fcm)
efcm = stepwise(steps.efcm)
rcm = stepwise(steps.rcm)
rmcm = stepwise(steps.rmcm)
som = stepwise(steps.som)
batch_som = stepwise(steps.batch_som)
som_olp = stepwise(steps.som_olp)

__version__ = "0.1.0"

__all__ = [
    "TPE",
    "Progress",
    "Result",
    "TPEResult",
    "ami",
    "ari",
    "batch_som",
    "choice",
    "continuity",
    "efcm",
    "fcm",
    "integer",
    "kmeans",
    "loguniform",
    "minimize",
    "rcm",
    "rmcm",
    "som",
    "som_olp",
    "steps",
    "trustworthiness",
    "uniform",
]
