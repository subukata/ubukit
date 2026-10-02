"""Verify JIT-free ordinary path and explicitly fail absent compatibility deps."""
import importlib.abc
import json
import os
import importlib.util
from pathlib import Path
import sys
import numpy as np

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('sklearn','scipy','numba'):
            raise ImportError('optional dependency intentionally blocked: '+fullname)
        return None

sys.meta_path.insert(0,BlockOptional())
import ubukit._impl.external_metrics as em
rng=np.random.default_rng(18)
x=rng.integers(20,size=1000);y=rng.integers(30,size=1000)
xb,yb=x.copy(),y.copy()
got=em.adjusted_scores(x,y)
assert np.array_equal(x,xb) and np.array_equal(y,yb)
assert got==em.adjusted_scores(x,y)
y[:]=x
assert em.adjusted_scores(x,y)=={'ari':1.,'ami':1.}
assert not any(k.split('.')[0] in ('sklearn','scipy','numba') for k in sys.modules)
try: em.adjusted_mutual_info_score(np.arange(4),[0,0,1,1])
except ImportError as exc:
    assert 'compatibility fallback' in str(exc)
else: raise AssertionError('required fallback dependency not enforced')
try: em.adjusted_mutual_info_score(x,yb,backend='numba')
except ImportError: pass
else: raise AssertionError('Numba must be explicitly available')
report={'status':'passed','ordinary_scores':got,'ordinary_optional_imports':False,'does_not_mutate_labels':True,'recomputes_after_input_mutation':True,'missing_sklearn_fallback':'explicit ImportError','missing_numba':'explicit ImportError'}
(Path(os.environ['UBUKIT_EXTERNAL_RESULTS'])/'portability_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
