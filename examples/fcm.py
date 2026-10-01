"""Run from the repository root: python examples/fcm.py."""
import numpy as np
from ubukit_fcm import fit_fcm

X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
result = fit_fcm(
    X, n_clusters=2, m=2.0, random_state=4,
    backend="scipy", max_iter=100, tol=1e-5, threads=1,
)

print("centers:", result["centers"], sep="\n")  # (K, D)
print("membership:", result["membership"], sep="\n")  # (N, K)
print("labels:", result["labels"])  # One maximum-membership cluster per point
print("objective:", result["objective"], "converged:", result["converged"])

assert result["centers"].shape == (2, 2)
assert result["membership"].shape == (6, 2)
np.testing.assert_allclose(result["membership"].sum(axis=1), 1.0)
assert np.isfinite(result["objective"])
