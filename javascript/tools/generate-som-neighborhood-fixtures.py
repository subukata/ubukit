#!/usr/bin/env python3
"""Tiny independent Python evidence. Run from package root or any directory.
Requires NumPy/SciPy/sklearn already available in the recovered Python environment.
No download, fit-cache reuse, or performance claim. BLAS thread count is one.
"""
import hashlib
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

HERE = pathlib.Path(__file__).resolve().parents[1]
REFERENCE = HERE.parent / 'restart_v2'
spec = importlib.util.spec_from_file_location('som_oracle', REFERENCE / 'som_candidate/oracle.py')
oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
spec = importlib.util.spec_from_file_location('metric_oracle', REFERENCE / 'portable_accel/_backends/metrics_numpy.py')
metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)


def pack(a):
    return {'data': a.ravel().tolist(), 'nSamples': a.shape[0], 'nFeatures': a.shape[1]}


def stable_quality(x, y, ks):
    n = len(x)
    # Explicit scalar direct differences and rooted float64, not sklearn's Gram
    # kernel. Sorting includes the index tie key; this is the JS exact contract.
    def ranks(z):
        rows, orders = [], []
        for i in range(n):
            ds = []
            for j in range(n):
                s = 0.
                for a, b in zip(z[i], z[j]):
                    delta = float(a) - float(b); s += delta * delta
                ds.append(math.sqrt(s))
            order = sorted((j for j in range(n) if j != i), key=lambda j: (ds[j], j))
            rank = [0] * n
            for r, j in enumerate(order, 1): rank[j] = r
            rows.append(rank); orders.append(order)
        return rows, orders
    rx, nx = ranks(x); ry, ny = ranks(y)
    result = []
    for k in dict.fromkeys(ks):
        pt = sum(max(0, rx[i][j] - k) for i in range(n) for j in ny[i][:k])
        pc = sum(max(0, ry[i][j] - k) for i in range(n) for j in nx[i][:k])
        factor = 2. / (n * k * (2. * n - 3. * k - 1.))
        result.append({'k': k, 'trustworthiness': 1. - pt * factor,
                       'continuity': 1. - pc * factor,
                       'trustworthinessPenalty': pt, 'continuityPenalty': pc})
    return result


def canonical_pca(x, r, lam):
    # Only nondegenerate cases compare to this. Canonicalize SVD signs to the JS
    # loading convention; upstream NumPy SVD itself does not impose these signs.
    mu = x.mean(axis=0); xc = x - mu
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(r.shape[1], x.shape[1])
    for row in vt:
        if row[np.argmax(np.abs(row))] < 0: row *= -1
    rc = r - r.mean(axis=0)
    rc /= np.maximum(np.max(np.abs(rc), axis=0), 1e-12)
    w = mu + (rc[:, :k] * (2. * s[:k] / np.sqrt(len(x)))) @ vt[:k]
    from scipy.special import softmax
    from scipy.spatial.distance import cdist
    return w, softmax(-cdist(x, w, 'sqeuclidean') / lam, axis=1)


def create():
    rng = np.random.default_rng(812)
    grid = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    result = {'provenance': {
        'somCommit': '4361175b776987d65c348d0132b31d43505e1069',
        'somSourceSha256': hashlib.sha256((REFERENCE / 'som_candidate/upstream/somolp.py').read_bytes()).hexdigest(),
        'somOracleSha256': hashlib.sha256((REFERENCE / 'som_candidate/oracle.py').read_bytes()).hexdigest(),
        'metricContract': 'direct-difference-rooted-float64; self excluded; distance then index ties',
    }, 'som': [], 'pca': [], 'neighborhood': []}
    for name, x, gamma, lam, iterations, tol in [
        ('ordinary', rng.normal(size=(19, 6)), .5, 1.2, 5, 0.),
        ('large-offset', rng.normal(size=(11, 3)) + 1.e9, .7, 1.4, 3, 0.),
        ('underflow', np.array([[-100., 0.], [-99., .1], [100., 0.], [101., .1]]), 0., .0001, 3, 0.),
        ('constant', np.full((7, 3), 2.), 1., .1, 20, 1.e-4),
    ]:
        w, p = oracle.initialize(x, grid, lam)
        expected = oracle.run(x, grid, w, p, gamma, lam, max_iters=iterations, tol=tol, trace=True)
        result['som'].append({'name': name, 'input': pack(x), 'grid': pack(grid),
            'initialPrototypes': w.ravel().tolist(), 'initialMemberships': p.ravel().tolist(),
            'options': {'gamma': gamma, 'lambda': lam, 'maxIterations': iterations, 'tolerance': tol},
            'expected': {key: expected[key].ravel().tolist() for key in ('W', 'P', 'V', 'history')},
            'iterations': expected['n_iter'],
            'firstV': expected['trajectory'][0]['V'].ravel().tolist()})
    # Zero-mass prototype retention and zero-iteration semantics use custom P.
    x = np.array([[0., 0.], [1., 1.], [2., 2.]])
    w = np.array([[1., 1.], [91., 93.], [21., 23.], [31., 33.]])
    p = np.zeros((3, 4)); p[:, 0] = 1.
    for iterations in (0, 1):
        expected = oracle.run(x, grid, w, p, 0., 1., max_iters=iterations, tol=0.)
        result['som'].append({'name': f'empty-unit-{iterations}', 'input': pack(x), 'grid': pack(grid),
            'initialPrototypes': w.ravel().tolist(), 'initialMemberships': p.ravel().tolist(),
            'options': {'gamma': 0., 'lambda': 1., 'maxIterations': iterations, 'tolerance': 0.},
            'expected': {key: None if expected[key] is None else expected[key].ravel().tolist() for key in ('W', 'P', 'V', 'history')},
            'iterations': expected['n_iter']})
    for name, x in [('feature-covariance', rng.normal(size=(13, 4))), ('dual-covariance', rng.normal(size=(4, 7)))]:
        w, p = canonical_pca(x, grid, 1.2)
        result['pca'].append({'name': name, 'input': pack(x), 'grid': pack(grid),
                             'lambda': 1.2, 'W': w.ravel().tolist(), 'P': p.ravel().tolist()})
    for name, x, y, ks in [
        ('ordinary', rng.normal(size=(17, 6)), rng.normal(size=(17, 2)), [1, 3, 5, 3]),
        ('duplicate-ties', np.repeat(rng.normal(size=(7, 4)), 2, axis=0), rng.normal(size=(14, 2)), [1, 2, 4]),
        ('all-equal', np.zeros((9, 3)), np.zeros((9, 2)), [1, 3, 4]),
        ('root-rounded-ties', np.array([[0., 0.], [1., 2.**-26], [1., 0.], [-1., 0.], [3., 2.], [4., -1.], [5., 1.]]), rng.normal(size=(7, 2)), [1, 2, 3]),
    ]:
        expected = stable_quality(x, y, ks)
        sklearn_expected = [dict(zip(('k', 'trustworthiness', 'continuity', 'trustworthinessPenalty', 'continuityPenalty'), q)) for q in metrics.joint_sklearn_numpy(x, y, ks)]
        result['neighborhood'].append({'name': name, 'input': pack(x), 'embedding': pack(y), 'ks': ks,
                                      'expected': expected, 'sklearn': sklearn_expected})
    return result


if __name__ == '__main__':
    with threadpool_limits(limits=1): result = create()
    target = HERE / 'fixtures/som-neighborhood.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {target}: {len(result["som"])} SOM, {len(result["pca"])} PCA, {len(result["neighborhood"])} metric fixtures')
