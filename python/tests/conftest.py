"""Reject editable/source-tree imports in the traditional SOM gate."""
from importlib import metadata
from pathlib import Path
import hashlib
import json
import sysconfig


def pytest_sessionstart(session):
    import ubukit
    import ubukit._impl.portable_accel._som_classic as som_module
    distribution = metadata.distribution('ubukit')
    purelib = Path(sysconfig.get_paths()['purelib']).resolve()
    owned = {str(path) for path in distribution.files}
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'SOURCE_MANIFEST.json').read_text())
    assert distribution.version == ubukit.__version__ == manifest['version']
    assert len({entry['path'] for entry in manifest['files']}) == len(manifest['files'])
    for entry in manifest['files']:
        name = entry['path'].removeprefix('src/')
        assert name in owned, name
        path = Path(distribution.locate_file(name)).resolve()
        assert path.is_relative_to(purelib), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], name
    for module in (ubukit, som_module):
        assert Path(module.__file__).resolve().is_relative_to(purelib)
