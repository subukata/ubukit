import numpy as np
import pytest


@pytest.fixture
def blobs():
    """Three well-separated Gaussian blobs (300 x 2) and their true centers."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    X = np.vstack([rng.normal(c, 0.5, (100, 2)) for c in centers])
    return X, centers


def match_centers(found, expected):
    """Max distance after matching each expected center to its nearest found one."""
    d = np.linalg.norm(found[:, None] - expected[None], axis=2)
    return d.min(axis=0).max()
