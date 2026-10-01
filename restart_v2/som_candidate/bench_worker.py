"""One method, fixture, scope and thread setting in one fresh process.

Run as a script with PYTHONPATH pointing to restart_v2. The parent campaign must
serialize workers. No benchmark is started by importing som_candidate itself.
"""
import argparse,gc,hashlib,importlib,json,os,pathlib,resource,statistics,time
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixture',required=True);parser.add_argument('--method',required=True)
parser.add_argument('--scope',choices=['kernel','initialization','fit'],default='kernel')
parser.add_argument('--initialization',choices=['direct','same_svd_lowrank'],default='direct')
parser.add_argument('--threads',type=int,default=1);parser.add_argument('--max-iters',type=int,default=10)
parser.add_argument('--tol',type=float,default=-1.);parser.add_argument('--repeats',type=int,default=3)
parser.add_argument('--kwargs',default='{}');parser.add_argument('--output',required=True)
a=parser.parse_args()
available=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
if available:os.sched_setaffinity(0,available[:a.threads])
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]=str(a.threads)
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE');os.environ.setdefault('GOMP_SPINCOUNT','0')
import numpy as np
from threadpoolctl import threadpool_info,threadpool_limits
from som_candidate.initialization import initialize
from som_candidate.oracle import initialize as oracle_initialize,run as oracle_run

z=np.load(a.fixture,allow_pickle=False)
X,R,W0,P0=(z[key] for key in ['X','R','W0','P0'])
gamma,lam=float(z['gamma']),float(z['lam'])
metadata=json.loads(str(z['metadata'])) if 'metadata' in z else {}
kwargs=json.loads(a.kwargs)

def fingerprint():
    h=hashlib.sha256()
    for ar in [X,R,W0,P0]:h.update(memoryview(ar).cast('B'))
    return h.hexdigest()

before=fingerprint()
record=dict(method=a.method,scope=a.scope,initialization=a.initialization,threads=a.threads,
            kwargs=kwargs,max_iters=a.max_iters,tol=a.tol,repeats=a.repeats,
            fixture_metadata=metadata,input_sha256=before,
            affinity=sorted(os.sched_getaffinity(0)) if available else None,
            time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),numpy_version=np.__version__,
            timing_scope='Per-call setup/allocation included; imports and fixture loading excluded')
try:
    start=time.perf_counter();module,fn=a.method.split(':');function=getattr(importlib.import_module(module),fn)
    record['import_seconds']=time.perf_counter()-start
    threadpool_limits(a.threads)
    def execute():
        if a.scope=='initialization':
            w,p=initialize(X,R,lam,threads=a.threads,method=a.initialization)
            return dict(W0=w,P0=p)
        if a.scope=='fit':
            start=time.perf_counter()
            w,p=initialize(X,R,lam,threads=a.threads,method=a.initialization)
            setup=time.perf_counter()-start
            out=function(X,R,w,p,gamma,lam,a.max_iters,a.tol,a.threads,**kwargs)
            out['initialization_component_seconds']=setup
            return out
        return function(X,R,W0,P0,gamma,lam,a.max_iters,a.tol,a.threads,**kwargs)
    start=time.perf_counter();out=execute();record['first_call_seconds']=time.perf_counter()-start
    keys=['W0','P0'] if a.scope=='initialization' else ['W','P','V','history']
    first={key:np.asarray(out[key]).copy() for key in keys}
    timings=[];cpus=[];stable=True
    for _ in range(a.repeats):
        gc.collect();gc.disable();start=time.perf_counter();cpu=time.process_time()
        try:out=execute()
        finally:
            timings.append(time.perf_counter()-start);cpus.append(time.process_time()-cpu);gc.enable()
        stable &= all(np.array_equal(first[key],np.asarray(out[key]),equal_nan=True) for key in keys)
    record.update(seconds=timings,cpu_seconds=cpus,median_seconds=statistics.median(timings),
                  mad_seconds=statistics.median(abs(t-statistics.median(timings)) for t in timings),
                  repeat_bitwise_stable=stable,input_unchanged=before==fingerprint())
    if a.scope=='kernel' and all('ref_'+key in z for key in keys) and metadata.get('reference_iters')==a.max_iters and metadata.get('reference_tol')==a.tol:
        expected={key:z['ref_'+key] for key in keys};expected_n_iter=int(z['ref_n_iter'])
    elif a.scope=='initialization':
        wr,pr=oracle_initialize(X,R,lam,threads=a.threads)
        expected=dict(W0=wr,P0=pr);expected_n_iter=None
    else:
        if a.scope=='fit':wr,pr=oracle_initialize(X,R,lam,threads=a.threads)
        else:wr,pr=W0,P0
        expected=oracle_run(X,R,wr,pr,gamma,lam,a.max_iters,a.tol,a.threads)
        expected_n_iter=expected['n_iter']
    comparisons={}
    for key in keys:
        value=np.asarray(out[key]);reference=np.asarray(expected[key]);same_shape=value.shape==reference.shape
        comparisons[key]=dict(shape=list(value.shape),reference_shape=list(reference.shape),finite=bool(np.all(np.isfinite(value))),
            allclose_1e8=bool(same_shape and np.allclose(value,reference,rtol=1e-8,atol=1e-8)))
        if same_shape:
            comparisons[key].update(max_abs=float(np.max(np.abs(value-reference))) if value.size else 0.,
                relative_l2=float(np.linalg.norm(value-reference)/max(1.,np.linalg.norm(reference))))
    record['validation']=comparisons
    record['n_iter']=int(out['n_iter']) if 'n_iter' in out else None
    record['reference_n_iter']=expected_n_iter
    record['status']='ok' if all(v['finite'] and v['allclose_1e8'] for v in comparisons.values()) and record['input_unchanged'] and (expected_n_iter is None or record['n_iter']==expected_n_iter) else 'numerical_difference'
    for key in ['variant','setup_seconds','direct_fallback_rows','effective_threads','blas_threads','initialization_component_seconds']:
        if key in out:record[key]=out[key]
    if 'P' in out:record['row_sum_max_error']=float(np.max(np.abs(out['P'].sum(1)-1.)))
    record['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record['threadpools']=threadpool_info()
    folder=pathlib.Path(__file__).resolve().parent
    record['source_sha256']={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in ['.py','.pyx','.h','.so'] and '__pycache__' not in p.parts}
except Exception as error:
    import traceback
    record.update(status='error',error=repr(error),traceback=traceback.format_exc())
output=pathlib.Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({key:record.get(key) for key in ['method','scope','threads','median_seconds','status','error']}),flush=True)
