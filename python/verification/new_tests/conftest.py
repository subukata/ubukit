from importlib import metadata
from pathlib import Path
import sysconfig
import pytest

def _guard():
    import ubukit, ubukit.optimization, ubukit._impl.fcm.core, ubukit._impl.fcm._robust, ubukit._impl.portable_accel.som_olp
    root = Path(sysconfig.get_paths()['purelib']).resolve()
    dist = metadata.distribution('ubukit')
    assert dist.version == ubukit.__version__ == '0.1.0a1'
    for mod in [ubukit,ubukit.optimization,ubukit._impl.fcm.core,ubukit._impl.fcm._robust,ubukit._impl.portable_accel.som_olp]:
        assert Path(mod.__file__).resolve().is_relative_to(root), mod.__file__

@pytest.fixture(autouse=True,scope='session')
def installed_only_guard():
    _guard()
    yield
    _guard()
