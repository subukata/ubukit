"""Run from the repository root: python examples/rcm.py."""
import numpy as np
from rough_cmeans import fit_rcm

X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
result = fit_rcm(
    X, n_clusters=2, init=X[[0, 3]],
    alpha=1.1, beta=0.5, backend="numpy", max_iter=100,
)

print("centers:", result.centers, sep="\n")  # (K, D)
print("membership:", result.memberships, sep="\n")  # (K, N), unlike FCM
print("stop:", result.stop_reason, "converged:", result.converged)

assert result.centers.shape == (2, 2)
assert result.memberships.shape == (2, 6)
np.testing.assert_allclose(result.memberships.sum(axis=0), 1.0)
assert result.p == 1.0
