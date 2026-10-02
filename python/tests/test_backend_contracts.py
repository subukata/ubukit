"""Focused direct-import contracts for the deliberately retained duplicate.

Run with one thread and a writable cache outside verified baseline source trees.
Numba tests skip if the optional dependency is absent. These are not the full
UbuKit verification suite or a performance benchmark.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('UBUKIT_SOURCE_ROOT', HERE / 'src')).resolve()
sys.path.insert(0, str(HERE / 'tools'))
from check_duplicate_backend import PAIRED_PATHS, check_pair

LEGACY = 'ubukit._impl.portable_accel._backends.metrics_numba'
CORE = 'ubukit._impl.portable_accel._backends.metrics_portable._numba_core'


def child(code):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    return subprocess.run(
        [sys.executable, '-I', '-B', '-c',
         f'import sys; sys.path.insert(0, {str(SOURCE)!r});\n' + code],
        env=env, capture_output=True, text=True, check=True,
    ).stdout


BLOCK_NUMBA = '''
import importlib.abc, importlib.util
class NoNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numba', 'llvmlite'):
            raise ModuleNotFoundError("No module named 'numba'", name='numba')
sys.meta_path.insert(0, NoNumba())
original_find_spec = importlib.util.find_spec
def find_spec(name, *args, **kwargs):
    if name.split('.')[0] in ('numba', 'llvmlite'):
        return None
    return original_find_spec(name, *args, **kwargs)
importlib.util.find_spec = find_spec
'''


class SourceAndOptionalContracts(unittest.TestCase):
    def test_source_pair_is_identical(self):
        result = check_pair(SOURCE)
        self.assertEqual(result['status'], 'identical')
        self.assertGreater(result['bytes_per_copy'], 0)

    def test_guard_detects_deliberate_divergence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, path in enumerate(PAIRED_PATHS):
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(f'# changed {index}\n')
            with self.assertRaisesRegex(ValueError, 'diverged'):
                check_pair(root)

    def test_root_imports_remain_lazy_without_numba(self):
        child(BLOCK_NUMBA + '''
import ubukit, ubukit._impl.portable_accel
assert not any(name.split('.')[0] in ('numpy', 'scipy', 'sklearn', 'numba', 'llvmlite')
               for name in sys.modules)
''')

    def test_legacy_import_fails_before_sklearn_without_numba(self):
        child(BLOCK_NUMBA + f'''
import importlib
try:
    importlib.import_module({LEGACY!r})
except ModuleNotFoundError as error:
    assert error.name == 'numba'
else:
    raise AssertionError('Legacy Numba import unexpectedly succeeded')
assert not any(name.split('.')[0] in ('sklearn', 'numba', 'llvmlite')
               for name in sys.modules)
assert {LEGACY!r} not in sys.modules
''')

    def test_active_direct_import_also_requires_numba(self):
        child(BLOCK_NUMBA + f'''
import importlib
try:
    importlib.import_module({CORE!r})
except ModuleNotFoundError as error:
    assert error.name == 'numba'
else:
    raise AssertionError('Active Numba import unexpectedly succeeded')
assert {CORE!r} not in sys.modules
''')

    def test_public_numpy_and_auto_survive_without_numba(self):
        child(BLOCK_NUMBA + '''
import numpy as np
import ubukit
x = np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.],[2.,1.],[3.,1.],[4.,2.],[5.,3.]])
y = x[:, ::-1].copy()
policy = ubukit.ExecutionPolicy(threads=1, block_rows=3)
a = ubukit.joint_quality(x, y, ks=[1,2,3], backend='numpy', policy=policy)
b = ubukit.joint_quality(x, y, ks=[1,2,3], backend='auto', policy=policy)
assert a == b
for backend in ('numba', 'sqrt_numba'):
    try:
        ubukit.joint_quality(x, y, ks=1, backend=backend, policy=policy)
    except ImportError:
        pass
    else:
        raise AssertionError('Explicit missing Numba backend succeeded')
assert not any(name.split('.')[0] in ('numba', 'llvmlite') for name in sys.modules)
''')


class NumbaDirectContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('numba') is None:
            raise unittest.SkipTest('Numba is not installed in this interpreter')
        sys.path.insert(0, str(SOURCE))
        cls.legacy = importlib.import_module(LEGACY)
        cls.core = importlib.import_module(CORE)
        import numpy as np
        import numba
        cls.np = np
        numba.set_num_threads(1)
        for module in (cls.legacy, cls.core):
            assert Path(module.__file__).is_relative_to(SOURCE), module.__file__
        cls.x = np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.],[2.,1.],[3.,1.],[4.,2.],[5.,3.]])
        cls.y = np.array([[0.,0.],[1.,0.],[0.,0.],[1.,0.],[3.,1.],[2.,1.],[5.,3.],[4.,2.]])

    def qualities(self, rows):
        return [dataclasses.asdict(row) for row in rows]

    def test_direct_exports_signatures_and_ownership(self):
        names = {node.name for node in ast.parse((SOURCE / PAIRED_PATHS[0]).read_text()).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        self.assertEqual({n for n in vars(self.legacy) if not n.startswith('_')},
                         {n for n in vars(self.core) if not n.startswith('_')})
        for name in names:
            a, b = getattr(self.legacy, name), getattr(self.core, name)
            with self.subTest(name=name):
                self.assertEqual(inspect.signature(a), inspect.signature(b))
                self.assertEqual(a.__module__, LEGACY)
                self.assertEqual(b.__module__, CORE)
                self.assertIsNot(a, b)

    def test_quality_class_pickle_and_mutability(self):
        a = self.legacy._quality(1, 8, 3, 4)
        b = self.core._quality(1, 8, 3, 4)
        self.assertEqual(dataclasses.asdict(a), dataclasses.asdict(b))
        self.assertNotEqual(a, b)
        self.assertIs(type(a), self.legacy.Quality)
        self.assertIs(type(b), self.core.Quality)
        for value in (a, b):
            self.assertIs(type(pickle.loads(pickle.dumps(value))), type(value))
            self.assertEqual(pickle.loads(pickle.dumps(value)), value)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                value.k = 2

    def test_legacy_monkeypatch_stays_module_local(self):
        marker = object()
        original_core = self.core._scan_ranks
        np = self.np
        d = np.array([[self.np.inf, 1., 2.]])
        q = np.array([[1]], dtype=np.int64)
        rank = np.array([[9]], dtype=np.int64)
        with patch.object(self.legacy, 'Quality', return_value=marker):
            self.assertIs(self.legacy._quality(1, 8, 3, 4), marker)
            self.assertIsInstance(self.core._quality(1, 8, 3, 4), self.core.Quality)
        with patch.object(self.legacy, '_scan_ranks', return_value=(rank, np.array([False]))):
            out = self.legacy.queried_ranks(d, q, method='scan', tie_policy='sklearn')
            np.testing.assert_array_equal(out, [[9]])
            self.assertIs(self.core._scan_ranks, original_core)

    def test_queried_ranks_ties_dtypes_and_fresh_outputs(self):
        np = self.np
        q = np.array([[2,1,2],[0,3,1]], dtype=np.int64)
        for dtype in (np.float32, np.float64):
            d = np.array([[np.inf,1.,1.,3.],[0.,2.,2.,0.]], dtype=dtype)
            saved_d, saved_q = d.copy(), q.copy()
            for tie in ('index', 'sklearn'):
                ref = self.legacy.queried_ranks(d, q, method='full', tie_policy=tie)
                for module in (self.legacy, self.core):
                    for method in ('full', 'scan', 'histogram'):
                        with self.subTest(dtype=dtype, tie=tie, module=module.__name__, method=method):
                            result = module.queried_ranks(d, q, method=method, tie_policy=tie)
                            np.testing.assert_array_equal(result, ref)
                            self.assertFalse(np.shares_memory(result, d))
                            self.assertFalse(np.shares_memory(result, q))
                            self.assertTrue(result.flags.writeable)
                            result[0,0] = -1
            np.testing.assert_array_equal(d, saved_d)
            np.testing.assert_array_equal(q, saved_q)

    def test_index_and_sklearn_metrics_exactly_match(self):
        np = self.np
        for dtype in (np.float32, np.float64):
            x, y = self.x.astype(dtype), self.y.astype(dtype)
            saved_x, saved_y = x.copy(), y.copy()
            for method in ('full', 'scan', 'histogram'):
                for name in ('joint_exact', 'joint_sklearn'):
                    with self.subTest(dtype=dtype, method=method, name=name):
                        a = getattr(self.legacy, name)(x, y, [1,2,3], rank_method=method, block_size=None)
                        b = getattr(self.core, name)(x, y, [1,2,3], rank_method=method, block_size=None)
                        self.assertEqual(self.qualities(a), self.qualities(b))
                        self.assertTrue(all(type(q) is self.legacy.Quality for q in a))
                        self.assertTrue(all(type(q) is self.core.Quality for q in b))
            np.testing.assert_array_equal(x, saved_x)
            np.testing.assert_array_equal(y, saved_y)

    def test_legacy_scalar_iterable_sqrt_and_reference_contracts(self):
        legacy, core = self.legacy, self.core
        for module in (legacy, core):
            scalar = module.trustworthiness_continuity(self.x, self.y, 2, rank_method='full')
            many = module.trustworthiness_continuity(self.x, self.y, [2], rank_method='full')
            self.assertIs(type(scalar), module.Quality)
            self.assertEqual([scalar], many)
            self.assertEqual(module.sklearn_reference(self.x, self.y, [1,2,3]),
                             module.joint_sklearn(self.x, self.y, [1,2,3], rank_method='full'))
        sqrt = importlib.import_module('ubukit._impl.portable_accel._backends.metrics_sqrt')
        a = sqrt.joint_sklearn_sqrt(self.x, self.y, [1,2,3], rank_method='scan')
        b = legacy.joint_sklearn(self.x, self.y, [1,2,3], rank_method='full')
        self.assertEqual(a, b)
        self.assertTrue(all(type(q) is legacy.Quality for q in a))

    def test_simple_reexports_change_legacy_contracts(self):
        # Demonstrate why a plain alias shim is rejected, without installing it.
        import types
        shim = types.ModuleType(LEGACY)
        shim.__dict__.update({name: value for name, value in vars(self.core).items()
                              if not name.startswith('__')})
        self.assertEqual(inspect.signature(shim.joint_exact), inspect.signature(self.legacy.joint_exact))
        self.assertEqual(shim.Quality.__module__, CORE)
        self.assertIs(type(shim._quality(1, 8, 3, 4)), self.core.Quality)
        marker = object()
        shim.Quality = lambda *args: marker
        self.assertIsNot(shim._quality(1, 8, 3, 4), marker)
        self.assertIs(shim.queried_ranks, self.core.queried_ranks)
        self.assertIs(shim._scan_ranks, self.core._scan_ranks)


if __name__ == '__main__':
    unittest.main(verbosity=2)
