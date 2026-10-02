"""Fresh-process import/first-call/RSS probe; values are not scratch-cap promises."""
import os,time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
start=time.perf_counter()
import argparse,importlib.util,json,resource
import numpy as np
import ubukit as uk
p=argparse.ArgumentParser();p.add_argument('version',choices=['dev4','candidate']);p.add_argument('--tail',action='store_true');args=p.parse_args()
if args.version=='candidate':
    from ubukit._impl.portable_accel.som_olp_localized import run_som_olp_localized as run
else:run=uk.run_som_olp
imports=time.perf_counter()-start
rng=np.random.default_rng(20261002);X=rng.normal(size=(128,8));R=rng.normal(size=(16,2));W=rng.normal(size=(16,8));P=rng.random((128,16))
if args.tail:P[:,-1]*=1e-300
P/=P.sum(1,keepdims=True)
start=time.perf_counter();r=run(X,R,W,P,gamma=.2,lam=.4,max_iters=5,tol=0.,backend='cdist_optimized',policy=uk.ExecutionPolicy(threads=1,block_rows=128));seconds=time.perf_counter()-start
print(json.dumps(dict(version=args.version,tail=args.tail,import_seconds=imports,first_fit_seconds=seconds,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cold_total_seconds=imports+seconds,numba_available=importlib.util.find_spec('numba') is not None,variant=r.get('variant'),scratch_budgeted_bytes=r.get('primary_scratch_budgeted_bytes',r.get('primary_scratch_bytes')),scope='Fresh isolated process; cold_total_seconds includes imports and first fit; RSS includes interpreter/imports/runtime and is not the scratch cap')))
