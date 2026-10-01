"""Run from the repository root: python examples/neighborhood_quality.py."""
import numpy as np
from portable_accel import ExecutionPolicy, joint_quality

X = np.random.default_rng(4).normal(size=(24, 3))  # Original space: (N, D)
Y = X[:, :2].copy()                             # Embedding: (same N, Q)
qualities = joint_quality(
    X, Y, ks=[3, 5], backend="numpy",
    max_distance_bytes=64 * 2**20,
    policy=ExecutionPolicy(threads=1),
)

for q in qualities:  # Always a list, including for ks=3
    print(f"k={q.k}: T={q.trustworthiness:.4f}, C={q.continuity:.4f}")

assert [q.k for q in qualities] == [3, 5]
for q in qualities:
    assert 0 <= q.trustworthiness <= 1
    assert 0 <= q.continuity <= 1
