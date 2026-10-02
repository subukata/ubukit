"""Reject source-tree imports and verify wheel RECORD ownership of loaded code."""
import base64
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import sysconfig

RUNTIME_ROOTS = ('ubukit', 'portable_accel', 'ubukit_fcm', 'ubukit_rmcm',
                 'rough_cmeans', '_numba_kernel', 'external_metrics', '_external_metrics_numba')

def check_installed(distribution='ubukit-bundled-local-preview', verify_hashes=False):
    if sys.prefix == sys.base_prefix:
        raise AssertionError('Run with a dedicated installed-wheel virtualenv, not the host interpreter')
    dist = importlib.metadata.distribution(distribution)
    direct_url = dist.read_text('direct_url.json')
    if direct_url and '"editable": true' in direct_url:
        raise AssertionError('Editable installation is not an installed-wheel verification')
    roots = {Path(sysconfig.get_path(key)).resolve() for key in ('purelib', 'platlib')}
    files = {Path(dist.locate_file(item)).resolve(): item for item in (dist.files or [])}
    def owned(path):
        p = Path(path).resolve()
        assert any(p.is_relative_to(root) for root in roots), ('outside virtualenv site-packages', str(p))
        assert p in files, ('not owned by installed distribution RECORD', str(p))
        if verify_hashes:
            expected = files[p].hash
            if expected:
                actual = base64.urlsafe_b64encode(hashlib.new(expected.mode, p.read_bytes()).digest()).rstrip(b'=').decode()
                assert actual == expected.value, ('installed source differs from RECORD', str(p))
        return str(p)
    paths = {}
    for name in RUNTIME_ROOTS:
        spec = importlib.util.find_spec(name)
        assert spec and spec.origin, ('missing installed runtime', name)
        paths[name] = owned(spec.origin)
    loaded = {}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in RUNTIME_ROOTS:
            path = getattr(module, '__file__', None)
            if path:
                loaded[name] = owned(path)
    return {'distribution':dist.metadata['Name'], 'version':dist.version,
            'prefix':sys.prefix, 'runtime_roots':paths, 'loaded_modules':loaded,
            'record_hashes_verified':verify_hashes}
