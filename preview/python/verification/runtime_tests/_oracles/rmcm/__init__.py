"""Rough Membership C-means with fixed delta neighborhoods."""
from .core import RMCMResult, PreparedRMCM, prepare_rmcm, fit_rmcm, fit_rmcm_numpy

__version__ = "0.1.0a1"
__all__ = ["RMCMResult", "PreparedRMCM", "prepare_rmcm", "fit_rmcm", "fit_rmcm_numpy"]
