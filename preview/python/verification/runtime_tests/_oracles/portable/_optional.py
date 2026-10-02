import importlib.util
import os
def numba_available():
    return os.environ.get('NUMBA_DISABLE_JIT','0')!='1' and importlib.util.find_spec('numba') is not None
