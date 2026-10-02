"""Run from an isolated wheel/sdist installation.

Usage: python -I installed_python_gate.py STAGING [base|numba]
Checks actual installed distribution ownership and source bytes, then SOM smoke.
Does not install, publish, or modify a checkout. Writes one JSON report to stdout.
"""
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import platform
import sys
import sysconfig

source = Path(sys.argv[1]).resolve()
with_numba = len(sys.argv) > 2 and sys.argv[2] == 'numba'
name = 'ubukit-bundled-local-preview'
dist = metadata.distribution(name)
import ubukit
assert ubukit.__version__ == dist.version
assert 'numpy' not in sys.modules, 'Facade eagerly imported NumPy'
assert 'numba' not in sys.modules, 'Facade eagerly imported optional Numba'
assert (util.find_spec('numba') is not None) == with_numba
site = Path(sysconfig.get_paths()['purelib']).resolve()
manifest = json.loads((source / 'SOURCE_MANIFEST.json').read_text())
assert dist.version == manifest['version']
files = {str(p).replace('\\', '/'): p for p in dist.files}
verified = []
for entry in manifest['files']:
    relative = entry['path'].removeprefix('src/')
    assert relative in files, f'Runtime file missing from RECORD: {relative}'
    actual = Path(dist.locate_file(files[relative])).resolve()
    assert actual.is_relative_to(site), str(actual)
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == entry['sha256'], relative
    verified.append(relative)

# RECORD/file ownership collisions are not detected by pip check alone.
expected_tops = {'ubukit', 'portable_accel', 'ubukit_fcm', 'ubukit_rmcm',
                 'rough_cmeans', '_numba_kernel', 'external_metrics', '_external_metrics_numba'}
providers = metadata.packages_distributions()
normalize = lambda s: s.lower().replace('_', '-').replace('.', '-')
for top in expected_tops:
    owners = providers.get(top, [])
    assert owners and {normalize(x) for x in owners} == {normalize(name)}, (top, owners)
    spec = util.find_spec(top)
    assert spec and spec.origin and Path(spec.origin).resolve().is_relative_to(site), top

import numpy as np
from threadpoolctl import threadpool_info
x = np.array([[0., 0.], [1., 0.], [8., 8.], [9., 8.]])
original = x.copy()
w0 = x[[0, 3]].copy()
for algorithm in ('som', 'som_batch'):
    fn = getattr(ubukit, algorithm)
    result = fn(x, grid_shape=(2, 1), initial_prototypes=w0,
                sigma=0., epochs=2, **({'learning_rate': .5} if algorithm == 'som' else {}))
    assert result['centers'].shape == (2, 2)
    assert result['labels'].shape == (4,)
    assert result['embedding'].shape == (4, 2)
    assert np.isfinite(result['centers']).all()
    assert result['iterations'] == (8 if algorithm == 'som' else 2)
    assert result['unit'] == ('sample' if algorithm == 'som' else 'epoch')
np.testing.assert_array_equal(x, original)
# Traditional SOM has no Numba backend; inherited API harness separately compiles existing paths.
versions = {p: metadata.version(p) for p in
            ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl') + (('numba', 'llvmlite') if with_numba else ())}
print(json.dumps({'status': 'passed', 'distribution': name, 'version': dist.version,
                  'python': sys.version, 'platform': platform.platform(),
                  'machine': platform.machine(), 'processor': platform.processor(),
                  'blas_libraries': threadpool_info(), 'dependencies': versions,
                  'runtime_files_verified': len(verified), 'owners': {x: providers[x] for x in sorted(expected_tops)},
                  'numba_installed': with_numba, 'numba_execution_checked_here': False}, indent=2))
