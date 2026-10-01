"""Restart-v2 correctness tests; these are fresh runs, not historical evidence."""
import importlib.util
import json
import subprocess
import sys
import unittest
import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from . import joint_quality
from ._sqrt_numpy import sqrt_queried_ranks_numpy


@threadpool_limits.wrap(limits=1)
def reference(X, Y, ks):
    n = len(X)
    inverses = []
    for Z in (X, Y):
        D = pairwise_distances(Z, metric='euclidean')
        np.fill_diagonal(D, np.inf)
        order = np.argsort(D, axis=1)
        inverse = np.empty((n, n), dtype=np.int64)
        inverse[np.arange(n)[:, None], order] = np.arange(1, n+1)
        inverses.append(inverse)
    result = []
    for k in ks:
        nx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
        ny = NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(return_distance=False)
        pt = int(np.maximum(inverses[0][np.arange(n)[:, None], ny] - k, 0).sum())
        pc = int(np.maximum(inverses[1][np.arange(n)[:, None], nx] - k, 0).sum())
        result.append((k, trustworthiness(X, Y, n_neighbors=k),
                       trustworthiness(Y, X, n_neighbors=k), pt, pc))
    return result


class CandidateTests(unittest.TestCase):
    def fixtures(self):
        rng = np.random.default_rng(20261001)
        for dtype in ('float32', 'float64'):
            X = rng.normal(size=(31, 7)).astype(dtype)
            Y = rng.normal(size=(31, 2)).astype(dtype)
            yield dtype, X, Y
            yield dtype+'_duplicates', np.round(X), np.round(Y)
            yield dtype+'_zero', np.zeros_like(X), np.zeros_like(Y)
        g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        yield 'grid', g, g[:, ::-1]

    def test_numpy_contracts(self):
        for name, X, Y in self.fixtures():
            ks = [1, 3, (len(X)-1)//2]
            expected = reference(X, Y, ks)
            for backend, methods in [('numpy', ['full', 'broadcast', 'searchsorted', 'sortsearch']),
                                     ('sqrt_numpy', ['broadcast', 'sortsearch'])]:
                for method in methods:
                    with self.subTest(fixture=name, backend=backend, method=method):
                        got = joint_quality(X, Y, ks, backend=backend, rank_method=method,
                            threads=1, block_rows=7, max_scratch_bytes=4096)
                        self.assertEqual([tuple(q) for q in got], expected)

    def test_no_numba_import_on_numpy_paths(self):
        code = """
import builtins, sys, numpy as np
real_import = builtins.__import__
def reject_numba(name, *a, **kw):
    if name == 'numba' or name.startswith('numba.') or name == 'llvmlite' or name.startswith('llvmlite.'):
        raise AssertionError('Numba imported on NumPy path')
    return real_import(name, *a, **kw)
builtins.__import__ = reject_numba
from restart_v2.metrics_candidate import joint_quality
X = np.arange(39., dtype=float).reshape(13, 3)
for b in ['numpy', 'sqrt_numpy']:
    joint_quality(X, X[:, :2], [1, 3], backend=b)
assert 'numba' not in sys.modules
"""
        completed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sqrt_rank_boundary_values(self):
        rng = np.random.default_rng(915)
        for dtype in ('float32', 'float64'):
            value = np.array(1.0, dtype=dtype)
            before = np.nextafter(value, np.array(0, dtype=dtype))
            after = np.nextafter(value, np.array(np.inf, dtype=dtype))
            tiny = np.nextafter(np.array(0, dtype=dtype), value)
            D = np.array([[0, tiny, before, value, after, 4, 4, np.inf]], dtype=dtype)
            Q = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 3]], dtype=np.int64)
            order = np.argsort(np.sqrt(D), axis=1)
            inv = np.empty_like(order);np.put_along_axis(inv, order, np.arange(1, D.shape[1]+1), axis=1)
            expected = np.take_along_axis(inv, Q, axis=1)
            for method in ('sortsearch', 'broadcast'):
                np.testing.assert_array_equal(sqrt_queried_ranks_numpy(D, Q, method=method, max_scratch_bytes=16), expected)

    def test_api_options_and_immutability(self):
        X = np.arange(51., dtype=float).reshape(17, 3);Y = X[:, :2].copy()
        X0, Y0 = X.copy(), Y.copy()
        results = joint_quality(X, Y, [3, 1, 3])
        self.assertEqual([q.k for q in results], [3, 1])
        self.assertEqual(len(joint_quality(X, Y, 3)), 1)
        with self.assertRaises(AttributeError):results[0].k = 9
        np.testing.assert_array_equal(X, X0);np.testing.assert_array_equal(Y, Y0)
        for options in [dict(backend='bad'), dict(threads=0), dict(block_rows=1.5),
                        dict(max_scratch_bytes=0), dict(max_distance_bytes=1)]:
            with self.assertRaises((ValueError, MemoryError)):
                joint_quality(X, Y, 3, **options)

    @unittest.skipUnless(importlib.util.find_spec('numba'), 'Optional Numba not installed')
    def test_optional_numba_contracts(self):
        for name, X, Y in self.fixtures():
            ks = [1, 3, (len(X)-1)//2]
            expected = reference(X, Y, ks)
            for backend in ('numba', 'sqrt_numba'):
                for method in ('scan', 'histogram'):
                    with self.subTest(fixture=name, backend=backend, method=method):
                        self.assertEqual([tuple(q) for q in joint_quality(X, Y, ks, backend=backend,
                            threads=1, rank_method=method, block_rows=7)], expected)

if __name__ == '__main__':unittest.main(verbosity=2)
