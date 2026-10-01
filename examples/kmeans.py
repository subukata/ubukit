"""Run from the repository root: python examples/kmeans.py."""
import numpy as np
from portable_accel import ExecutionPolicy, fit_kmeans

# X: (sample count N, feature count D); init: (cluster count K, D)
X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
init = X[[0, 3]].copy()  # Two explicit starting centers
result = fit_kmeans(
    X, init, backend="numpy", max_iter=30,
    policy=ExecutionPolicy(threads=1),
)

print("centers:", result["centers"], sep="\n")  # (K, D)
print("labels:", result["labels"])             # (N,)
print("inertia:", result["inertia"])           # Sum of squared distances
print("iterations:", result["n_iter"])

assert result["centers"].shape == (2, 2)
np.testing.assert_array_equal(result["labels"], [0, 0, 0, 1, 1, 1])
np.testing.assert_allclose(result["inertia"], 8 / 3)
