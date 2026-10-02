"""New restart-v2 research foundation; lazy public imports."""
from importlib import import_module
__version__='0.2.0a2'
_exports={'ExecutionPolicy':'policy','PreparedData':'prepared','PreparedSOM':'som_prepared','prepare':'prepared','fit_kmeans':'kmeans','joint_quality':'metrics','initialize_som_olp':'som_olp','run_som_olp':'som_olp','fit_som_olp':'som_olp'}
_exports.update({name: '_som_classic' for name in ('som', 'som_batch', 'fit_som', 'fit_som_batch', 'initialize_som', 'initialize_som_batch', 'SOMState')})
__all__=list(_exports)
def __getattr__(name):
    if name not in _exports:raise AttributeError(name)
    obj=getattr(import_module('.'+_exports[name],__name__),name);globals()[name]=obj;return obj
