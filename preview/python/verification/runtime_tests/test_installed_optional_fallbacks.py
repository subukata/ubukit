"""Installed subprocess coverage for absence of the optional Numba runtime."""
import os
import subprocess
import sys


def test_optional_numba_absence_and_auto_fallbacks():
    code = r'''
import importlib.abc, importlib.util, sys
class BlockNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numba', 'llvmlite'):
            raise ImportError('optional backend intentionally unavailable')
sys.meta_path.insert(0, BlockNumba())
original_find_spec = importlib.util.find_spec
def without_numba(name, *args, **kwargs):
    if name.split('.')[0] in ('numba', 'llvmlite'):
        return None
    return original_find_spec(name, *args, **kwargs)
importlib.util.find_spec = without_numba
import ubukit
assert not any(n in sys.modules for n in ('numpy','scipy','sklearn','numba','llvmlite'))
import numpy as np
x = np.array([[0.,0.],[0.,1.],[1.,0.],[8.,8.],[8.,9.],[9.,8.]])
c=x[[0,3]].copy()
policy=ubukit.ExecutionPolicy(threads=1)
a=ubukit.fit_kmeans(x,c,backend='auto',finalizer='scipy',max_iter=3,policy=policy)
b=ubukit.fit_kmeans(x,c,backend='numpy',finalizer='scipy',max_iter=3,policy=policy)
for key in ('centers','labels','core_labels','inertia','n_iter'):
    np.testing.assert_array_equal(a[key],b[key])
assert ubukit.fit_fcm(x,2,backend='scipy',random_state=1,max_iter=2,threads=1)['membership'].shape==(6,2)
assert ubukit.fit_exrcm(x,2,init=c,backend='scipy',max_iter=2).memberships.shape==(6,2)
assert ubukit.fit_rmcm(x,2,delta=1.5,init=c,backend='adjoint',max_iter=2,threads=1).memberships.shape==(6,2)
operations = [
    lambda: ubukit.fit_kmeans(x,c,backend='numba',max_iter=1,policy=policy),
    lambda: ubukit.fit_kmeans(x,c,backend='numpy',finalizer='numba',max_iter=1,policy=policy),
    lambda: ubukit.fit_fcm(x,2,backend='numba',max_iter=1,threads=1),
    lambda: ubukit.fit_exrcm(x,2,init=c,backend='numba',max_iter=1),
    lambda: ubukit.fit_rmcm(x,2,delta=1.5,init=c,backend='numba',max_iter=1,threads=1),
]
for operation in operations:
    try:
        operation()
    except ImportError:
        pass
    else:
        raise AssertionError('explicit unavailable Numba backend did not raise ImportError')
assert not any(n.split('.')[0] in ('numba','llvmlite') for n in sys.modules)
print('five explicit optional-backend failures and auto fallback passed')
'''
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code], env=env,
                            text=True, capture_output=True, check=True)
    assert 'auto fallback passed' in result.stdout


def test_bounded_sparse_graph_memory_guard():
    import numpy as np
    import pytest
    from ubukit_rmcm._graph import graph_tree, graph_blocked
    for method in (graph_tree, graph_blocked):
        with pytest.raises(MemoryError):
            method(np.zeros((100,8)),1,1000,4)
        points=np.arange(257.)[:,None]*np.ones((1,2))
        graph,degrees,count=method(points,.1,1024,16)
        assert graph.nnz == count == len(points)
        np.testing.assert_array_equal(degrees,np.ones(len(points)))
