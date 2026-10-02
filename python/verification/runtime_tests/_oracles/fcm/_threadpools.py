"""Threadpool controllers cached only while the dynamic-library set is unchanged.

On glibc Linux, dl_iterate_phdr's add/remove generations detect even ctypes-loaded
libraries. Older runtimes and other platforms conservatively rediscover on each
activation. Limits and their previous values are never cached: every activation
still creates the ordinary threadpoolctl scoped limiter.
"""
import ctypes
import os
import sys
import threading
from threadpoolctl import ThreadpoolController

_controller = None
_generation = None
_lock = threading.RLock()
_probe = None
_callback_type = None
_info_type = None

if sys.platform.startswith('linux'):
    class _DlPhdrInfo(ctypes.Structure):
        _fields_ = [('addr', ctypes.c_void_p), ('name', ctypes.c_char_p),
                    ('phdr', ctypes.c_void_p), ('phnum', ctypes.c_ushort),
                    ('adds', ctypes.c_ulonglong), ('subs', ctypes.c_ulonglong)]
    try:
        _callback_type = ctypes.CFUNCTYPE(ctypes.c_int,
                ctypes.POINTER(_DlPhdrInfo), ctypes.c_size_t, ctypes.c_void_p)
        _libc = ctypes.CDLL(None)
        # Counter semantics are verified for glibc. Unknown/musl runtimes use
        # conservative rediscovery even if a similarly named function exists.
        getattr(_libc, 'gnu_get_libc_version')
        _probe = _libc.dl_iterate_phdr
        _probe.argtypes = [_callback_type, ctypes.c_void_p]
        _probe.restype = ctypes.c_int
        _info_type = _DlPhdrInfo
    except (AttributeError, OSError):
        _probe = None


def _library_generation():
    if _probe is None:
        return None
    generation = []

    @_callback_type
    def first(info, size, unused):
        if size >= ctypes.sizeof(_info_type):
            generation.extend((info.contents.adds, info.contents.subs))
        return 1

    _probe(first, None)
    return tuple(generation) if generation else None


def threadpool_context(limits, user_api=None):
    """Return a fresh restorable limiter for every currently loaded library."""
    global _controller, _generation
    with _lock:
        generation = _library_generation()
        if _controller is None or generation is None or generation != _generation:
            _controller = ThreadpoolController()
            # Keep the PRE-discovery generation. If another thread loads a
            # library after enumeration, the next activation must rediscover
            # rather than mark an incomplete controller as current forever.
            _generation = generation
        return _controller.limit(limits=limits, user_api=user_api)


def _reset_after_fork():
    # A fork can inherit a lock held by another thread. Rebuild rather than
    # sharing controller/lock state with the parent process.
    global _controller, _generation, _lock
    _controller = None
    _generation = None
    _lock = threading.RLock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_after_fork)
