"""Local UbuKit facade with sample-major membership axes.

Only fit_rcm, fit_exrcm and assign_rcm adapt memberships to (N, K). Other
exports remain direct aliases; result field names and container types are not
otherwise standardized. All original runtime imports retain their contracts.
"""
from importlib import import_module

__version__ = "0.0.0.dev4"
_EXPORTS = {
    "TPEOptimizer": ("ubukit.optimization", "TPEOptimizer"),
    "optimize": ("ubukit.optimization", "optimize"),
    "float_range": ("ubukit.optimization", "float_range"),
    "int_range": ("ubukit.optimization", "int_range"),
    "categorical": ("ubukit.optimization", "categorical"),
    "Trial": ("ubukit.optimization", "Trial"),
    "OptimizationResult": ("ubukit.optimization", "OptimizationResult"),
    "SearchSpaceExhausted": ("ubukit.optimization", "SearchSpaceExhausted"),
    "ProposalError": ("ubukit.optimization", "ProposalError"),

    "adjusted_rand_score": ("external_metrics", "adjusted_rand_score"),
    "adjusted_mutual_info_score": ("external_metrics", "adjusted_mutual_info_score"),
    "adjusted_scores": ("external_metrics", "adjusted_scores"),
    "ExecutionPolicy": ("portable_accel", "ExecutionPolicy"),
    "PreparedData": ("portable_accel", "PreparedData"),
    "PreparedSOM": ("portable_accel", "PreparedSOM"),
    "prepare": ("portable_accel", "prepare"),
    "fit_kmeans": ("portable_accel", "fit_kmeans"),
    "fit_som_olp": ("portable_accel", "fit_som_olp"),
    "initialize_som_olp": ("portable_accel", "initialize_som_olp"),
    "run_som_olp": ("portable_accel", "run_som_olp"),
    "joint_quality": ("portable_accel", "joint_quality"),
    "fit_fcm": ("ubukit_fcm", "fit_fcm"),
    "fit_fcm_numpy": ("ubukit_fcm", "fit_fcm_numpy"),
    "fit_rcm": ("rough_cmeans", "fit_rcm"),
    "fit_exrcm": ("rough_cmeans", "fit_exrcm"),
    "assign_rcm": ("rough_cmeans", "assign"),
    "fit_rmcm": ("ubukit_rmcm", "fit_rmcm"),
    "fit_rmcm_numpy": ("ubukit_rmcm", "fit_rmcm_numpy"),
    "prepare_rmcm": ("ubukit_rmcm", "prepare_rmcm"),
    "PreparedRMCM": ("ubukit_rmcm", "PreparedRMCM"),
    "RMCMResult": ("ubukit_rmcm", "RMCMResult"),
}
_EXPORTS.update({name: ("portable_accel", name) for name in ("som", "som_batch", "fit_som", "fit_som_batch", "initialize_som", "initialize_som_batch", "SOMState")})
_ADAPTED_EXPORTS = {
    name: ("ubukit._rough_memberships", name)
    for name in ("fit_rcm", "fit_exrcm", "assign_rcm")
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _ADAPTED_EXPORTS.get(name, _EXPORTS[name])
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
