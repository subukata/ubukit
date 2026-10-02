"""Restricted NumPy/OpenBLAS arithmetic scope shared with the reviewed rough path.

Unknown builds, layouts and unreviewed DGEMM dispatch shapes use direct math.
The immutable linked build identity is cached; this does not cache thread limits.
"""
import numpy as np
_BLAS_CERTIFICATE_BUILD = None

def _blas_certificate_supported(X, C):
    """Only the reviewed NumPy/OpenBLAS double-GEMM path may certify masks.

    Unknown providers/versions keep working through direct distances. The
    linked NumPy build identity is cached; thread limits/controllers are not.
    """
    if (type(X) is not np.ndarray or type(C) is not np.ndarray
            or X.dtype != np.float64 or C.dtype != np.float64
            or not X.flags.c_contiguous or not C.flags.c_contiguous
            or not X.flags.aligned or not C.flags.aligned
            or len(X) < 32 or len(C) < 4 or X.shape[1] < 128
            or len(X) * len(C) * X.shape[1] <= 1000000
            or np.may_share_memory(X, C)):
        return False
    global _BLAS_CERTIFICATE_BUILD
    if _BLAS_CERTIFICATE_BUILD is None:
        _BLAS_CERTIFICATE_BUILD = False
        try:
            from threadpoolctl import threadpool_info
            blas = np.__config__.CONFIG['Build Dependencies']['blas']
            config = blas.get('openblas configuration', '')
            if (np.__version__ == '2.3.5' and blas.get('name') == 'scipy-openblas'
                    and blas.get('version') == '0.3.30'
                    and 'USE64BITINT' in config and 'DYNAMIC_ARCH' in config):
                _BLAS_CERTIFICATE_BUILD = any(
                    info.get('internal_api') == 'openblas'
                    and info.get('version') == '0.3.30'
                    and info.get('architecture') == 'SkylakeX'
                    and info.get('threading_layer') == 'pthreads'
                    and 'numpy.libs' in info.get('filepath', '')
                    and 'libscipy_openblas64_' in info.get('filepath', '')
                    for info in threadpool_info())
        except (ImportError, AttributeError, KeyError, TypeError, ValueError, OSError):
            pass
    return _BLAS_CERTIFICATE_BUILD

