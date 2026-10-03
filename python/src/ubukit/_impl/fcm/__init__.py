"""Portable, standard squared-Euclidean fuzzy c-means.

The optional Numba implementation is imported only when explicitly selected.
"""
from .core import fit_fcm, fit_fcm_numpy, memberships_from_squared_distances

__version__ = "0.1.0a3"
__all__ = ["fit_fcm", "fit_fcm_numpy", "memberships_from_squared_distances"]
