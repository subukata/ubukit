"""Sample-major adapters; numerical work remains in untouched rough_cmeans.

This module is imported only when an adapted facade function is accessed.
No shape-based guessing is used: rough_cmeans always returns cluster-major
memberships, including square N == K cases.
"""
import numpy as np
import rough_cmeans as _legacy


def _sample_major(values):
    """Own a writable C-contiguous transpose, retaining dtype and exact values."""
    if values is None:
        return None
    return np.array(values.T, copy=True, order="C")


class _RoughMembershipResult:
    """Forward metadata and expose independent (N, K) membership buffers.

    All non-membership attributes retain the legacy values/array references.
    The legacy result is retained privately; neither it nor its arrays are
    mutated. The two public membership arrays own their storage and do not
    alias the legacy buffers or one another. This is a facade result adapter,
    not a rough_cmeans.RoughCMeansResult/dataclass. Use rough_cmeans directly
    if the legacy result type or cluster-major output is required.
    """
    __slots__ = ("_legacy_result", "memberships", "upper_memberships")

    def __init__(self, result):
        self._legacy_result = result
        self.memberships = _sample_major(result.memberships)
        self.upper_memberships = _sample_major(result.upper_memberships)

    def __getattr__(self, name):
        # object.__getattribute__ also keeps unpickling/missing-state lookups
        # from recursively trying to resolve _legacy_result.
        return getattr(object.__getattribute__(self, "_legacy_result"), name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._legacy_result)))

    def __repr__(self):
        shape = None if self.memberships is None else self.memberships.shape
        return (f"{type(self).__name__}(membership_shape={shape}, "
                f"n_iter={self.n_iter!r}, converged={self.converged!r}, "
                f"stop_reason={self.stop_reason!r})")


def fit_rcm(X, n_clusters, **kwargs):
    """Fit legacy RCM, exposing memberships/upper_memberships as (N, K).

    Keyword arguments and numerical behavior are those of rough_cmeans.fit_rcm.
    With return_memberships=False, both membership attributes remain None.
    Other result metadata is forwarded without changing values or ownership.
    """
    return _RoughMembershipResult(_legacy.fit_rcm(X, n_clusters, **kwargs))


def fit_exrcm(X, n_clusters, *, alpha=1.1, beta=0.0, p=1.0,
              init=None, seed=0, max_iter=300, backend="auto", block_size=4096,
              return_memberships=True, cycle_window=16):
    """Fit legacy ExRCM, exposing memberships/upper_memberships as (N, K).

    All parameters, defaults and numerical behavior match rough_cmeans.fit_exrcm.
    Normalized memberships retain float64 dtype and upper memberships retain
    bool dtype. Both are independent writable C-contiguous copies; other
    metadata is forwarded. return_memberships=False preserves None outputs.
    """
    return _RoughMembershipResult(_legacy.fit_exrcm(
        X, n_clusters, alpha=alpha, beta=beta, p=p, init=init, seed=seed,
        max_iter=max_iter, backend=backend, block_size=block_size,
        return_memberships=return_memberships, cycle_window=cycle_window,
    ))


def assign_rcm(X, centers, *, alpha=1.1, beta=0.0, p=1.0,
               backend="numpy", block_size=4096):
    """Return (normalized memberships, binary upper memberships), each (N, K).

    Uses rough_cmeans.assign unchanged, then returns independent writable
    C-contiguous copies. Dtypes remain float64 and bool, respectively.
    """
    memberships, upper_memberships = _legacy.assign(
        X, centers, alpha=alpha, beta=beta, p=p,
        backend=backend, block_size=block_size,
    )
    return _sample_major(memberships), _sample_major(upper_memberships)
