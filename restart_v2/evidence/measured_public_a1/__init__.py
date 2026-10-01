"""New restart-v2 research foundation; lazy public imports."""
from importlib import import_module
__version__='0.2.0a1'
_exports={'ExecutionPolicy':'policy','PreparedData':'prepared','prepare':'prepared','fit_kmeans':'kmeans','joint_quality':'metrics','initialize_som_olp':'som_olp','run_som_olp':'som_olp','fit_som_olp':'som_olp'}
__all__=list(_exports)
def __getattr__(name):
    if name not in _exports:raise AttributeError(name)
    obj=getattr(import_module('.'+_exports[name],__name__),name);globals()[name]=obj;return obj
