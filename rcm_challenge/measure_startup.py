"""Fresh-process startup/JIT and total-process peak RSS, not allocator-only memory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def worker(backend):
    start=time.perf_counter()
    import resource
    import numpy as np
    from rough_cmeans import fit
    import_seconds=time.perf_counter()-start
    rng=np.random.default_rng(107)
    X=rng.normal(size=(2048,256)); init=X[rng.choice(len(X),16,replace=False)]
    kw=dict(init=init,alpha=1.15,beta=.5,p=2,backend=backend,max_iter=4,return_memberships=False)
    begin=time.perf_counter(); first=fit(X,16,**kw); first_seconds=time.perf_counter()-begin
    begin=time.perf_counter(); second=fit(X,16,**kw); second_seconds=time.perf_counter()-begin
    print(json.dumps(dict(backend=backend,import_seconds=import_seconds,first_fit_seconds=first_seconds,second_fit_seconds=second_seconds,total_seconds=time.perf_counter()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,n_iter=second.n_iter,n=2048,k=16,d=256,p=2,max_iter=4,return_memberships=False)))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--worker'); ap.add_argument('--output',default='results/startup.json'); args=ap.parse_args()
    if args.worker: return worker(args.worker)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='exrcm-cold-') as cache:
        for backend in ['naive','numpy','scipy','numba']:
            env=dict(os.environ, NUMBA_CACHE_DIR=cache,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            child=subprocess.run([sys.executable,__file__,'--worker',backend],env=env,text=True,capture_output=True,check=True)
            rows.append(json.loads(child.stdout))
    hashes={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['rough_cmeans.py','_numba_kernel.py','measure_startup.py']}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(source_sha256=hashes,note='One fresh process per backend; Numba uses initially empty isolated JIT cache. OS disk cache not flushed. Peak RSS is whole Linux process including interpreter, imported libraries and JIT compiler, not just arrays. max_iter=4 is a startup workload, not a convergence claim.',results=rows),indent=2))
    for row in rows:print(json.dumps(row))


if __name__=='__main__':main()
