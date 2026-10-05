"""UbuKit: fuzzy, rough and classical clustering, self-organizing maps,
evaluation metrics and TPE search on NumPy and SciPy.

Every fitting function runs to the end and returns a Result. ``steps`` holds
the same functions as generators that yield a Progress after every iteration,
whose ``result()`` is the Result as if the run had stopped there:

    for p in ub.steps.som_olp(X, (10, 10), lam=0.5, gamma=1.0):
        draw(p.result())
"""

from types import SimpleNamespace

from ._core import STEPS, Progress, Result
from .cluster import efcm, fcm, kmeans, rcm, rmcm
from .metrics import ami, ari, continuity, trustworthiness
from .som import batch_som, som, som_olp
from .tpe import TPE, TPEResult, choice, integer, loguniform, minimize, uniform

steps = SimpleNamespace(**STEPS)
"""The fitting functions as generators, e.g. ``steps.kmeans(X, 3)``."""

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
