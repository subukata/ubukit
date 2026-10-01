"""Run from the repository root: python examples/exrcm.py."""
import numpy as np
from rough_cmeans import fit_exrcm

X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
result = fit_exrcm(
    X, n_clusters=2, init=X[[0, 3]],
    p=2.0, alpha=1.1, beta=0.5, backend="numpy", max_iter=100,
)

print("centers:", result.centers, sep="\n")  # (K, D)
print("mask:", result.upper_memberships, sep="\n")  # (K, N), bool
print("membership:", result.memberships, sep="\n")  # (K, N)
print("stop:", result.stop_reason, "converged:", result.converged)

assert result.memberships.shape == (2, 6)
assert result.upper_memberships.dtype == np.bool_
np.testing.assert_allclose(result.memberships.sum(axis=0), 1.0)
assert result.p == 2.0
