"""UbuKit: fuzzy, rough and classical clustering, self-organizing maps,
evaluation metrics and TPE search on NumPy and SciPy."""

from ._core import Result
from .cluster import efcm, fcm, kmeans, rcm, rmcm
from .metrics import ami, ari, continuity, trustworthiness
from .som import batch_som, som, som_olp
from .tpe import TPE, TPEResult, choice, integer, loguniform, minimize, uniform

__version__ = "0.1.0"

__all__ = [
    "TPE",
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
    "trustworthiness",
    "uniform",
]
