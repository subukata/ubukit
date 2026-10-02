"""Local UbuKit facade with sample-major membership axes.

Only fit_rcm, fit_exrcm and assign_rcm adapt memberships to (N, K). Other
exports remain direct aliases; result field names and container types are not
otherwise standardized. Implementation modules are private and live only under ubukit._impl.
"""
from importlib import import_module

__version__ = "0.1.0a1"
_EXPORTS = {
    "TPEOptimizer": (".optimization", "TPEOptimizer"),
    "optimize": (".optimization", "optimize"),
    "float_range": (".optimization", "float_range"),
    "int_range": (".optimization", "int_range"),
    "categorical": (".optimization", "categorical"),
    "Trial": (".optimization", "Trial"),
    "OptimizationResult": (".optimization", "OptimizationResult"),
    "SearchSpaceExhausted": (".optimization", "SearchSpaceExhausted"),
    "ProposalError": (".optimization", "ProposalError"),

    "adjusted_rand_score": ("._impl.external_metrics", "adjusted_rand_score"),
    "adjusted_mutual_info_score": ("._impl.external_metrics", "adjusted_mutual_info_score"),
    "adjusted_scores": ("._impl.external_metrics", "adjusted_scores"),
    "ExecutionPolicy": ("._impl.portable_accel", "ExecutionPolicy"),
    "PreparedData": ("._impl.portable_accel", "PreparedData"),
    "PreparedSOM": ("._impl.portable_accel", "PreparedSOM"),
    "prepare": ("._impl.portable_accel", "prepare"),
    "fit_kmeans": ("._impl.portable_accel", "fit_kmeans"),
    "fit_som_olp": ("._impl.portable_accel", "fit_som_olp"),
    "initialize_som_olp": ("._impl.portable_accel", "initialize_som_olp"),
    "run_som_olp": ("._impl.portable_accel", "run_som_olp"),
    "joint_quality": ("._impl.portable_accel", "joint_quality"),
    "fit_fcm": ("._impl.fcm", "fit_fcm"),
    "fit_fcm_numpy": ("._impl.fcm", "fit_fcm_numpy"),
    "fit_entropy_fcm": ("._impl.entropy_fcm", "fit_entropy_fcm"),
    "fit_rcm": ("._impl.rough_cmeans", "fit_rcm"),
    "fit_exrcm": ("._impl.rough_cmeans", "fit_exrcm"),
    "assign_rcm": ("._impl.rough_cmeans", "assign"),
    "fit_rmcm": ("._impl.rmcm", "fit_rmcm"),
    "fit_rmcm_numpy": ("._impl.rmcm", "fit_rmcm_numpy"),
    "prepare_rmcm": ("._impl.rmcm", "prepare_rmcm"),
    "PreparedRMCM": ("._impl.rmcm", "PreparedRMCM"),
    "RMCMResult": ("._impl.rmcm", "RMCMResult"),
}
_EXPORTS.update({name: ("._impl.portable_accel", name) for name in ("som", "som_batch", "fit_som", "fit_som_batch", "initialize_som", "initialize_som_batch", "SOMState")})
_ADAPTED_EXPORTS = {
    name: ("._rough_memberships", name)
    for name in ("fit_rcm", "fit_exrcm", "assign_rcm")
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _ADAPTED_EXPORTS.get(name, _EXPORTS[name])
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
