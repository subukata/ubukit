"""Small numerical-limit fixture; disagreement is reported, not hidden."""
import json
from pathlib import Path
import numpy as np
from ubukit_rmcm import fit_rmcm

X = np.array([[1,-1],[4,5],[-3,-4],[1,3],[-5,3],[2,2],[-2,-2],[100,100]], float)
init = X[[0,1,7]].copy()
rows = []
for backend in ['numpy', 'csr', 'adjoint']:
    first = fit_rmcm(X, 3, 12, init=init, backend=backend, max_iter=1)
    r = fit_rmcm(X, 3, 12, init=init, backend=backend, max_iter=20)
    rows.append(dict(backend=backend, first_centers=first.centers.tolist(),
                     centers=r.centers.tolist(), labels=r.labels.tolist(),
                     n_iter=r.n_iter, stop_reason=r.stop_reason, cycle_length=r.cycle_length))
output = dict(note='The first seven points form a clique; the final point is isolated. Centers 0 and 1 coincide in exact arithmetic. Float64 sum reassociation changes last bits, which changes subsequent exact nearest-center tie decisions. These differences are a documented numerical limit, not claimed parity. Whole-graph common-mean shortcut does not apply to this partial graph.',
              X=X.tolist(), init=init.tolist(), delta=12, results=rows)
Path('results/roundoff_fixture.json').write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps(output, indent=2))
