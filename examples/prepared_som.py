"""Run from the repository root: python examples/prepared_som.py."""
import numpy as np
from portable_accel import ExecutionPolicy, PreparedSOM

X = np.random.default_rng(4).normal(size=(24, 3))
R = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
policy = ExecutionPolicy(threads=1)
prepared = PreparedSOM(X, max_rank=2, threads=policy.threads)

for gamma, lam in [(0.5, 1.0), (0.8, 1.2)]:
    result = prepared.fit(R, gamma=gamma, lam=lam, max_iters=50, policy=policy)
    print(f"gamma={gamma}, lam={lam}: embedding={result['V'].shape}")
    assert result["V"].shape == (24, 2)
    np.testing.assert_allclose(result["P"].sum(axis=1), 1.0)

# The snapshot and full SVD are prepared once; fit starts afresh each time.
print("owned bytes:", prepared.describe()["owned_snapshot_and_factor_bytes"])
