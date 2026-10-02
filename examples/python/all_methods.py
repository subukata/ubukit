"""Executable README examples for the private UbuKit dev6 trial.

Run with the installed virtualenv: python -I -B examples/all_methods.py
Add --with-numba only after installing this local wheel's [numba] extra.
"""
import json
import sys
import numpy as np
import ubukit as uk

X = np.array([[0., 0.], [0., 1.], [1., 0.],
              [8., 8.], [8., 9.], [9., 8.]])
centers = X[[0, 3]].copy()
policy = uk.ExecutionPolicy(threads=1)

# Owned reusable data, via either constructor or convenience function.
snapshot = uk.PreparedData(X)
data = uk.prepare(X)
assert isinstance(data, uk.PreparedData)
np.testing.assert_array_equal(snapshot.X, data.X)

# k-means: labels (N,), centers (K,D).
km = uk.fit_kmeans(data, centers, backend="numpy", max_iter=10, policy=policy)
assert km["labels"].shape == (6,)

# Fuzzy c-means: the dict uses the singular key "membership", shape (N,K).
fcm = uk.fit_fcm(X, 2, random_state=4, backend="scipy", max_iter=10, threads=1)
fcm_reference = uk.fit_fcm_numpy(X, 2, random_state=4, max_iter=10, threads=1)
assert fcm["membership"].shape == fcm_reference["membership"].shape == (6, 2)

# Rough c-means and extended rough c-means: facade arrays are (N,K).
rcm = uk.fit_rcm(X, 2, init=centers, backend="numpy", max_iter=10)
exrcm = uk.fit_exrcm(X, 2, init=centers, alpha=1.1, beta=0., p=1.,
                   backend="numpy", max_iter=10)
membership, upper = uk.assign_rcm(X, rcm.centers, backend="numpy")
assert membership.shape == upper.shape == exrcm.memberships.shape == (6, 2)

# Rough membership c-means: one-shot, NumPy reference, and prepared graph.
rmcm = uk.fit_rmcm(X, 2, delta=1.5, init=centers, threads=1)
rmcm_reference = uk.fit_rmcm_numpy(X, 2, delta=1.5, init=centers, threads=1)
graph = uk.prepare_rmcm(X, delta=1.5)
assert isinstance(graph, uk.PreparedRMCM)  # Use the factory, not private graph internals.
rmcm_repeated = graph.fit(2, init=centers, threads=1)
assert isinstance(rmcm, uk.RMCMResult)
assert rmcm.memberships.shape == rmcm_reference.memberships.shape == rmcm_repeated.memberships.shape == (6, 2)

# SOM-OLP: one-shot, explicit initialization/run, and reusable preparation.
grid = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
som = uk.fit_som_olp(X, grid, gamma=.1, lam=1., max_iters=3, policy=policy)
W0, P0 = uk.initialize_som_olp(X, grid, lam=1., policy=policy)
som_from_init = uk.run_som_olp(X, grid, W0, P0, gamma=.1, lam=1.,
                              max_iters=3, policy=policy)
prepared_som = uk.PreparedSOM(X, threads=1)
som_repeated = prepared_som.fit(grid, gamma=.1, lam=1., max_iters=3)
assert som["V"].shape == som_from_init["V"].shape == som_repeated["V"].shape == (6, 2)

# Neighborhood quality: records contain trustworthiness and continuity.
Q = np.random.default_rng(73).normal(size=(12, 3))
quality = uk.joint_quality(Q, Q, ks=[1, 2], backend="numpy", policy=policy)
assert all(q.trustworthiness == 1 and q.continuity == 1 for q in quality)

# ARI, AMI, and a joint helper, all accepting raw label arrays.
truth = np.array([0, 0, 0, 1, 1, 1])
predicted = np.array([0, 0, 1, 1, 2, 2])
ari = uk.adjusted_rand_score(truth, predicted)
ami = uk.adjusted_mutual_info_score(truth, predicted, average_method="arithmetic")
scores = uk.adjusted_scores(truth, predicted)
assert scores == {"ari": ari, "ami": ami}
for method in ["arithmetic", "geometric", "min", "max"]:
    assert np.isfinite(uk.adjusted_mutual_info_score(truth, predicted, average_method=method))
if "--with-numba" in sys.argv:
    compiled = uk.adjusted_scores(truth, predicted, backend="numba")
    np.testing.assert_allclose(list(compiled.values()), list(scores.values()), atol=1e-8, rtol=0)

print(json.dumps({"status": "passed", "version": uk.__version__,
                  "facade_exports": len(uk.__all__), "optional_numba_example": "--with-numba" in sys.argv,
                  "ari": ari, "ami": ami}))
