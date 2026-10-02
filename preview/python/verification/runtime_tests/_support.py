"""Frozen test oracles, loaded only under private names (never candidate roots)."""
import importlib.util
from pathlib import Path
import sys

_ORACLES = Path(__file__).resolve().parent / '_oracles'

def _load(name, relative, package=False):
    if name in sys.modules:
        return sys.modules[name]
    source = _ORACLES / relative
    spec = importlib.util.spec_from_file_location(
        name, source,
        submodule_search_locations=[str(source.parent)] if package else None,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

fcm_oracle = _load('_ubukit_oracle_fcm', 'fcm/__init__.py', True)
rough_oracle = _load('_ubukit_oracle_rough', 'rough_cmeans.py')
portable_oracle = _load('_ubukit_oracle_portable', 'portable/__init__.py', True)
rmcm_oracle = _load('_ubukit_oracle_rmcm', 'rmcm/__init__.py', True)
kmeans_reference = _load('_ubukit_oracle_numpy_kmeans', 'numpy_kmeans_reference.py')
