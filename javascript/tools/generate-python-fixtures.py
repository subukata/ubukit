"""Small reproducible fixture pass, not a benchmark. Run from the candidate root."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
try:
    import ubukit_rmcm
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT.parent/'rmcm_challenge'))
    import ubukit_rmcm
from ubukit_rmcm import fit_rmcm, prepare_rmcm
core=Path(ubukit_rmcm.__file__).with_name('core.py')
cases=[]
def add(name, X, init, delta, max_iter=100, cycle_window=32):
    X=np.asarray(X,dtype=float); init=np.asarray(init,dtype=float)
    item=dict(name=name, X=X.tolist(), init=init.tolist(), delta=delta, maxIterations=max_iter, cycleWindow=cycle_window, backends={})
    for backend in ['numpy','adjoint']:
        p=prepare_rmcm(X,delta,backend=backend)
        r=p.fit(len(init),init=init,max_iter=max_iter,cycle_window=cycle_window)
        item['backends'][backend]=dict(centers=r.centers.tolist(),labels=r.labels.tolist(),membership=r.memberships.tolist(),iterations=r.n_iter,converged=r.converged,stopReason=r.stop_reason,cycleLength=r.cycle_length,nEdges=r.n_edges,emptyClusterUpdates=r.empty_cluster_updates,degrees=p.degrees.tolist())
    cases.append(item)
add('irregular_degrees',[[0],[1],[3]],[[0],[3]],2,1)
add('zero_delta_duplicates_ties',[[0],[0],[2],[4]],[[0],[4]],0,1)
add('full_graph_empty',[[1,-1],[4,5],[-3,-4],[1,3],[-5,3],[2,2],[-2,-2]],[[1,-1],[4,5],[4,5]],100,1)
cycle=[[-1,-3],[5,-3],[3,4],[-1,-2],[1,-3],[2,2]]
add('cycle_with_empty',cycle,[[5,-3],[1,-3],[2,2]],7.3)
add('cycle_without_empty',cycle,[[1,-3],[2,2]],7.3)
add('cycle_disabled',cycle,[[5,-3],[1,-3],[2,2]],7.3,9,0)
add('update_producing_output',[[0],[2],[3],[7],[8]],[[0],[3]],2.1,1)
add('delta_zero_kmeans',[[0],[1],[8],[9]],[[0],[9]],0)
add('single_point',[[4,2]],[[4,2]],0)
add('exact_boundary',[[0,0],[3,4],[6,8]],[[0,0],[6,8]],5,1)
add('inside_boundary',[[0,0],[3,4],[6,8]],[[0,0],[6,8]],float(np.nextafter(5,0)),1)
add('outside_boundary',[[0,0],[3,4],[6,8]],[[0,0],[6,8]],float(np.nextafter(5,np.inf)),1)
rng=np.random.default_rng(9831)
for seed in range(5):
    X=rng.normal(size=(19,3))
    for delta in [0,.7,1.8]: add(f'random_{seed}_{delta}',X,X[[0,5,12]],delta,8)
result=dict(python_core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(), cases=cases)
(ROOT/'fixtures/rmcm-python.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'{len(cases)} cases, two Python backends per case')
