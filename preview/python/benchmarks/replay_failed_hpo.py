"""Replay the original 53 failed SOM configurations, without rerunning search."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse,hashlib,inspect,json,signal,time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris,load_wine
from sklearn.preprocessing import StandardScaler
import ubukit as uk
from portable_accel.som_olp_localized import fit_som_olp_localized
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
source=Path(a.source);output=Path(a.out)
rows=[json.loads(line) for line in source.read_text().splitlines()];failed=[r for r in rows if r['task'].startswith('som_') and r['state']=='fail']
X={name:np.ascontiguousarray(StandardScaler().fit_transform(loader().data)) for name,loader in [('som_iris',load_iris),('som_wine',load_wine)]}
R=np.array([(u,v) for u in np.linspace(-1,1,8) for v in np.linspace(-1,1,8)])
policy=uk.ExecutionPolicy(threads=1)
result={'actual_package_version':uk.__version__,'candidate_source_sha256':hashlib.sha256(Path(inspect.getsourcefile(fit_som_olp_localized)).read_bytes()).hexdigest(),'baseline_som_olp_sha256':hashlib.sha256(Path(inspect.getsourcefile(uk.fit_som_olp)).read_bytes()).hexdigest(),'timing_protocol':'One serial replay per original failed configuration, 2s wall timer around fit, scientific imports preloaded, single BLAS thread, identical StandardScaler full-dataset inputs, no quality-metric time inside fit timer','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'historical_failed_cases':len(failed),'scope':'Same fixed parameter sets only, not rerun HPO; historical dev3 failures compared with opt-in candidate; actual package identity and numerical source hash are recorded above; paired preserved-dev4 numerical timings reported separately','replays':[]}
def timeout(*args):raise TimeoutError('2-second original HPO per-fit resource bound')
signal.signal(signal.SIGALRM,timeout)
for row in failed:
    record={k:row[k] for k in ('task','sampler','search_seed','trial','params')};start=time.perf_counter();signal.setitimer(signal.ITIMER_REAL,2.)
    try:
        out=fit_som_olp_localized(X[row['task']],R,**row['params'],max_iters=100,tol=1e-5,backend='cdist_optimized',policy=policy)
        record.update(state='completed',fit_seconds=time.perf_counter()-start,n_iter=out['n_iter'],variant=out.get('variant'),finite=all(bool(np.isfinite(out[k]).all()) for k in ('P','W','V','history')))
    except Exception as e:record.update(state='failed',fit_seconds=time.perf_counter()-start,error=repr(e))
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    result['replays'].append(record)
output.write_text(json.dumps(result,indent=2)+'\n');print('Replay completed',sum(r['state']=='completed' for r in result['replays']),'/',len(failed),'max seconds',max(r['fit_seconds'] for r in result['replays']),flush=True)
# One baseline is allowed to finish beyond the original search cutoff. The
# extended deadline is purely a validation resource limit, not a speed claim.
row=failed[0];base=None;record={'task':row['task'],'params':row['params']};start=time.perf_counter();signal.setitimer(signal.ITIMER_REAL,30.)
try:
    base=uk.fit_som_olp(X[row['task']],R,**row['params'],max_iters=100,tol=1e-5,backend='cdist_optimized',policy=policy)
    record.update(state='completed',dev4_seconds=time.perf_counter()-start,dev4_n_iter=base['n_iter'])
except Exception as e:record.update(state='bounded_stop',dev4_seconds=time.perf_counter()-start,error=repr(e))
finally:signal.setitimer(signal.ITIMER_REAL,0)
new=fit_som_olp_localized(X[row['task']],R,**row['params'],max_iters=100,tol=1e-5,backend='cdist_optimized',policy=policy)
record['candidate_n_iter']=new['n_iter']
if base is not None:
    record['agreement']={}
    for key in ('W','V','P','history'):
        record['agreement'][key]={'shape_match':base[key].shape==new[key].shape}
        if base[key].shape==new[key].shape:record['agreement'][key].update(max_absolute_error=float(np.max(np.abs(base[key]-new[key]),initial=0)),allclose_1e_10=bool(np.allclose(base[key],new[key],rtol=1e-10,atol=1e-12)))
    for version,r in [('dev4',base),('candidate',new)]:
        q=uk.joint_quality(X[row['task']],r['V'],ks=[10],backend='numpy',policy=policy)[0]
        record[version+'_quality']={'trustworthiness':float(q.trustworthiness),'continuity':float(q.continuity)}
result['full_baseline_comparison']=record;output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(record),flush=True)
