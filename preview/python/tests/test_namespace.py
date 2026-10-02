"""Installed dev6 namespace, laziness, identity, and current-pickle contracts."""
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile


LEGACY_ROOTS = ('portable_accel', 'ubukit_fcm', 'ubukit_rmcm', 'rough_cmeans',
                '_numba_kernel', 'external_metrics', '_external_metrics_numba')
EXPORTS = {
    'TPEOptimizer', 'optimize', 'float_range', 'int_range', 'categorical', 'Trial',
    'OptimizationResult', 'SearchSpaceExhausted', 'ProposalError',
    'adjusted_rand_score', 'adjusted_mutual_info_score', 'adjusted_scores',
    'ExecutionPolicy', 'PreparedData', 'PreparedSOM', 'prepare', 'fit_kmeans',
    'fit_som_olp', 'initialize_som_olp', 'run_som_olp', 'joint_quality',
    'fit_fcm', 'fit_fcm_numpy', 'fit_rcm', 'fit_exrcm', 'assign_rcm',
    'fit_rmcm', 'fit_rmcm_numpy', 'prepare_rmcm', 'PreparedRMCM', 'RMCMResult',
    'som', 'som_batch', 'fit_som', 'fit_som_batch', 'initialize_som',
    'initialize_som_batch', 'SOMState',
}


def isolated(code):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code], cwd=directory,
                                env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr
        return result


def test_only_ubukit_is_owned_as_a_top_level_namespace():
    dist = importlib.metadata.distribution('ubukit-bundled-local-preview')
    assert dist.version == '0.0.0.dev6'
    assert set((dist.read_text('top_level.txt') or '').split()) == {'ubukit'}
    runtime = [Path(item) for item in dist.files if str(item).endswith('.py')]
    assert len(runtime) == 63
    assert all(path.parts[0] == 'ubukit' for path in runtime)
    assert not any(Path(item).parts[0].removesuffix('.py') in LEGACY_ROOTS for item in dist.files)


def test_root_and_private_namespace_initializers_are_lazy():
    isolated('''
import sys
import ubukit
assert ubukit.__version__ == '0.0.0.dev6'
assert not any(name.startswith('ubukit._impl') for name in sys.modules)
import ubukit._impl
assert not any(name.split('.')[0] in ('numpy', 'scipy', 'sklearn', 'numba', 'llvmlite')
               for name in sys.modules)
''')


def test_export_identity_and_signatures_survive_legacy_name_collisions():
    isolated(f'''
import importlib, inspect, pathlib, sys, tempfile
with tempfile.TemporaryDirectory() as directory:
    for name in {LEGACY_ROOTS!r}:
        pathlib.Path(directory, name + '.py').write_text('raise AssertionError("legacy namespace imported")\\n')
    sys.path.insert(0, directory)
    import ubukit
    assert set(ubukit.__all__) == {EXPORTS!r}
    assert len(ubukit.__all__) == 38
    assert set(ubukit._ADAPTED_EXPORTS) == {{'fit_rcm', 'fit_exrcm', 'assign_rcm'}}
    for name, (module_name, attribute) in ubukit._EXPORTS.items():
        direct = getattr(importlib.import_module(module_name, 'ubukit'), attribute)
        public = getattr(ubukit, name)
        assert callable(public), name
        # Exception subclasses inherit an uninspectable builtin constructor;
        # their direct object identity is asserted below.
        if not (isinstance(direct, type) and issubclass(direct, BaseException)):
            assert inspect.signature(public) == inspect.signature(direct), name
        if name in ubukit._ADAPTED_EXPORTS:
            assert public is not direct, name
        else:
            assert public is direct, name
    assert not any(name.split('.')[0] in {LEGACY_ROOTS!r} for name in sys.modules)
    assert not any(name.split('.')[0] in ('numba', 'llvmlite') for name in sys.modules)
''')


def test_current_policy_pickle_uses_relocated_class_identity():
    isolated('''
import pickle
import ubukit
value = ubukit.ExecutionPolicy(threads=1)
assert type(value).__module__ == 'ubukit._impl.portable_accel.policy'
restored = pickle.loads(pickle.dumps(value))
assert type(restored) is ubukit.ExecutionPolicy
assert restored == value
''')
