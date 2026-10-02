"""Run with an installed wheel/sdist, outside the source checkout."""
import importlib.util
import json
import sys
from pathlib import Path

import ubukit
from importlib.metadata import version
assert ubukit.__version__ == version("ubukit-bundled-local-preview") == "0.0.0.dev3"
assert "numpy" not in sys.modules, "Facade import must not eagerly load numerical runtimes"
assert "numba" not in sys.modules

import numpy as np
import portable_accel
import ubukit_fcm
import rough_cmeans
import ubukit_rmcm

for name, (module_name, attribute) in ubukit._EXPORTS.items():
    module = __import__(module_name)
    if name in ubukit._ADAPTED_EXPORTS:
        assert getattr(ubukit, name) is not getattr(module, attribute), name
    else:
        assert getattr(ubukit, name) is getattr(module, attribute), name
assert "fit_rmcm" in dir(ubukit)
try:
    ubukit.not_an_api
except AttributeError:
    pass
else:
    raise AssertionError("Unknown facade attribute did not fail")

X = np.array([[0., 0.], [0., 1.], [1., 0.], [8., 8.], [8., 9.], [9., 8.]])
original = X.copy()
policy = ubukit.ExecutionPolicy(threads=1)
km = ubukit.fit_kmeans(X, X[[0, 3]], backend="numpy", max_iter=10, policy=policy)
np.testing.assert_array_equal(km["labels"], [0, 0, 0, 1, 1, 1])
np.testing.assert_allclose(km["inertia"], 8 / 3)
fcm = ubukit.fit_fcm(X, 2, random_state=4, backend="scipy", max_iter=10, threads=1)
assert fcm["membership"].shape == (6, 2)
np.testing.assert_allclose(fcm["membership"].sum(axis=1), 1)
for fun in (ubukit.fit_rcm, ubukit.fit_exrcm):
    result = fun(X, 2, init=X[[0, 3]], backend="numpy", max_iter=10)
    assert result.memberships.shape == (6, 2)
    assert result.upper_memberships.shape == (6, 2)
    np.testing.assert_allclose(result.memberships.sum(axis=1), 1)
rmcm = ubukit.fit_rmcm(X, 2, delta=1.5, init=X[[0, 3]], backend="adjoint", threads=1)
assert rmcm.memberships.shape == (6, 2)
np.testing.assert_allclose(rmcm.memberships.sum(axis=1), 1)
prepared = ubukit.prepare_rmcm(X, 1.5)
assert prepared.fit(2, init=X[[0, 3]], threads=1).memberships.shape == (6, 2)
R = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
som = ubukit.fit_som_olp(X, R, gamma=.1, lam=1, max_iters=3,
                         backend="threadpool", initializer="svd_lowrank", policy=policy)
assert som["V"].shape == (6, 2)
assert ubukit.PreparedSOM(X, threads=1).fit(R, gamma=.1, lam=1, max_iters=2)["P"].shape == (6, 4)
metric_input = np.random.default_rng(73).normal(size=(12, 3))  # Avoid ambiguous distance ties.
quality = ubukit.joint_quality(metric_input, metric_input, ks=[1, 2], backend="numpy", policy=policy)
assert all(q.trustworthiness == 1 and q.continuity == 1 for q in quality)
np.testing.assert_array_equal(X, original)
assert "numba" not in sys.modules, "Explicit base paths should not import optional Numba"

numba_ran = False
if "--with-numba" in sys.argv:
    assert importlib.util.find_spec("numba") is not None
    assert ubukit.fit_kmeans(X, X[[0, 3]], backend="numba", max_iter=2, policy=policy)["labels"].shape == (6,)
    assert ubukit.fit_fcm(X, 2, backend="numba", random_state=4, max_iter=2, threads=1)["membership"].shape == (6, 2)
    assert ubukit.fit_exrcm(X, 2, backend="numba", init=X[[0, 3]], max_iter=2).memberships.shape == (6, 2)
    assert ubukit.fit_rmcm(X, 2, delta=1.5, backend="numba", init=X[[0, 3]], max_iter=2, threads=1).memberships.shape == (6, 2)
    numba_ran = True

print(json.dumps({"status": "passed", "facade_exports": len(ubukit.__all__),
                  "algorithm_entrypoints": 7, "existing_unchanged_aliases": 16, "added_metric_aliases": 3, "adapted_entrypoints": 3, "numba_executed": numba_ran,
                  "import_paths": {m.__name__: str(Path(m.__file__).resolve()) for m in
                                   (ubukit, portable_accel, ubukit_fcm, rough_cmeans, ubukit_rmcm)}}))
