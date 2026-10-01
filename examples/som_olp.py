"""Run from the repository root: python examples/som_olp.py."""
import numpy as np
from portable_accel import ExecutionPolicy, fit_som_olp

X = np.random.default_rng(4).normal(size=(24, 3))  # (N, D)
R = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  # (M, Q)
result = fit_som_olp(
    X, R, gamma=0.5, lam=1.0, max_iters=50, tol=1e-4,
    backend="threadpool", initializer="svd_lowrank",
    policy=ExecutionPolicy(threads=1),
)

print("embedding shape:", result["V"].shape)   # (N, Q): 24 x 2
print("prototype shape:", result["W"].shape)   # (M, D): 4 x 3
print("membership shape:", result["P"].shape)  # (N, M): 24 x 4
print("iterations:", result["n_iter"])

assert result["V"].shape == (24, 2)
assert result["W"].shape == (4, 3)
assert result["P"].shape == (24, 4)
np.testing.assert_allclose(result["P"].sum(axis=1), 1.0)
assert np.isfinite(result["V"]).all()
