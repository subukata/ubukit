"""One true full-fit configuration per fresh process, with reusable reference.

Original mode executes the pinned SOMOLP.fit class. Its initializer is wrapped
only to apply the recorded init-thread policy and record its wall component.
Portable/native modes perform initialization and kernel inside every timer.
The first original fit supplies reference outputs; no extra oracle is run.
"""
import argparse,gc,hashlib,json,os,pathlib,resource,signal,time,traceback
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixture',required=True);p.add_argument('--mode',choices=['original','portable','native'],required=True)
p.add_argument('--threads',type=int,default=9);p.add_argument('--init-threads',type=int,default=1)
p.add_argument('--initialization',choices=['direct','same_svd_lowrank'],default='same_svd_lowrank')
p.add_argument('--max-iters',type=int,default=1000);p.add_argument('--tol',type=float,default=1e-4)
p.add_argument('--repeats',type=int,default=3);p.add_argument('--block-rows',type=int,default=256)
p.add_argument('--max-scratch-bytes',type=int,default=64<<20)
p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
p.add_argument('--budget-seconds',type=int,default=1000)
a=p.parse_args()
if min(a.threads,a.init_threads,a.max_iters,a.repeats,a.block_rows,a.budget_seconds)<1:p.error('positive thread/iteration/repeat/block/budget values required')
if hasattr(os,'sched_getaffinity'):
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:max(a.threads,a.init_threads)])
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]=str(a.threads)
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE');os.environ.setdefault('GOMP_SPINCOUNT','0')
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from som_candidate.initialization import initialize
from som_candidate.upstream.somolp import SOMOLP as OriginalSOMOLP
if a.mode=='portable':
    from portable_accel import fit_som_olp,ExecutionPolicy
    if a.init_threads!=a.threads:raise ValueError('public full fit uses the same policy thread count for initialization and kernel')
elif a.mode=='native':
    from som_candidate.native import run as kernel

folder=pathlib.Path(__file__).resolve().parent
fixture=np.load(a.fixture,allow_pickle=False);X=fixture['X'];R=fixture['R']
gamma,lam=float(fixture['gamma']),float(fixture['lam'])
metadata=json.loads(str(fixture['metadata'])) if 'metadata' in fixture else {}
output=pathlib.Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
reference_path=pathlib.Path(a.reference)

def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()

def sources():
    return {str(f.relative_to(folder)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(folder.rglob('*')) if f.is_file() and f.suffix in ['.py','.pyx','.h','.so'] and '__pycache__' not in f.parts}

input_hash={'X':digest(X),'R':digest(R)}
record=dict(mode=a.mode,scope='full_fit',initialization='original_direct' if a.mode=='original' else a.initialization,
            kernel_threads=a.threads,initialization_threads=a.init_threads,max_iters=a.max_iters,tol=a.tol,
            repeats_requested=a.repeats,repeats_completed=0,block_rows=a.block_rows,
            max_scratch_bytes=a.max_scratch_bytes if a.mode=='portable' else None,
            input_sha256=input_hash,fixture_metadata=metadata,source_sha256=sources(),
            original_source_sha256=hashlib.sha256((folder/'upstream/somolp.py').read_bytes()).hexdigest(),
            time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            timing_scope='Every timed fit includes initialization, kernel setup, iterations and returned-state construction; data loading/imports/validation/reference serialization excluded',
            seconds=[],cpu_seconds=[],initialization_seconds=[],kernel_plus_overhead_seconds=[],validations=[])
public_root=folder.parent/'portable_accel'
public_freeze_path=folder.parent/'evidence/som_public_fullfit_freeze.json'
public_freeze=json.loads(public_freeze_path.read_text())
def public_hashes():
    return {name:hashlib.sha256((public_root/name).read_bytes()).hexdigest() for name in public_freeze['files']}
record['public_source_freeze']=public_freeze
if a.mode=='portable':
    record['public_call']='portable_accel.fit_som_olp(...,backend=threadpool,initializer=svd_lowrank,policy=ExecutionPolicy(threads=9,block_rows=256,max_scratch_bytes=67108864))'
    record['scratch_scope']='named private/merged statistics and worker score buffers; centered coordinates, outputs, SVD, fallback temporaries and total RSS excluded'
    record['initialization_component_note']='not separately instrumented for public API; full initialization is inside total fit timer'

class TimedOriginal(OriginalSOMOLP):
    def _init_pca(self,values):
        started=time.perf_counter()
        with threadpool_limits(a.init_threads,user_api='blas'):
            super()._init_pca(values)
        self._measured_initialization_seconds=time.perf_counter()-started

def execute():
    if a.mode=='original':
        model=TimedOriginal(R,gamma=gamma,lam=lam,max_iters=a.max_iters,tol=a.tol,pca_scale=2.)
        model.fit(X)
        return dict(W=model.W,P=model.P,V=model.V,history=np.asarray(model.history),n_iter=model.n_iter),model._measured_initialization_seconds
    if a.mode=='portable':
        result=fit_som_olp(X,R,gamma=gamma,lam=lam,max_iters=a.max_iters,tol=a.tol,
                          backend='threadpool',initializer='svd_lowrank',
                          policy=ExecutionPolicy(threads=a.threads,block_rows=a.block_rows,max_scratch_bytes=a.max_scratch_bytes))
        return result,None
    started=time.perf_counter()
    W0,P0=initialize(X,R,lam,threads=a.init_threads,method=a.initialization)
    initialization_seconds=time.perf_counter()-started
    kwargs=dict(block_rows=a.block_rows,distance='guarded')
    if a.mode=='portable':kwargs['max_scratch_bytes']=a.max_scratch_bytes
    result=kernel(X,R,W0,P0,gamma,lam,a.max_iters,a.tol,a.threads,**kwargs)
    return result,initialization_seconds

def validate(result,reference):
    checks={}
    for key in ['W','P','V','history']:
        value=np.asarray(result[key]);expected=np.asarray(reference[key]);shape=value.shape==expected.shape
        checks[key]=dict(shape=list(value.shape),reference_shape=list(expected.shape),finite=bool(np.all(np.isfinite(value))),
                         allclose_1e8=bool(shape and np.allclose(value,expected,rtol=1e-8,atol=1e-8)))
        if shape:checks[key]['max_abs']=float(np.max(np.abs(value-expected))) if value.size else 0.
    checks['n_iter_match']=int(result['n_iter'])==int(reference['n_iter'])
    return checks

def save():
    record['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record['threadpools']=threadpool_info()
    if record['seconds']:
        values=np.asarray(record['seconds']);record['median_seconds']=float(np.median(values))
        record['mad_seconds']=float(np.median(np.abs(values-np.median(values))))
        if all(v is not None for v in record['initialization_seconds']):
            record['median_initialization_seconds']=float(np.median(record['initialization_seconds']))
            record['median_kernel_plus_overhead_seconds']=float(np.median(record['kernel_plus_overhead_seconds']))
    output.write_text(json.dumps(record,indent=2)+'\n')

class BudgetExpired(TimeoutError):pass
def timeout_handler(signum,frame):raise BudgetExpired('full-fit worker time budget expired')
signal.signal(signal.SIGALRM,timeout_handler)
signal.setitimer(signal.ITIMER_REAL,a.budget_seconds)
try:
    if public_hashes()!=public_freeze['files']:raise ValueError('public source differs from frozen manifest')
    threadpool_limits(a.threads,user_api='blas')
    reference=None
    if a.mode!='original':
        if not reference_path.exists():raise FileNotFoundError('original full-fit reference must exist first')
        refdata=np.load(reference_path,allow_pickle=False)
        refmeta=json.loads(str(refdata['metadata']))
        if refmeta['input_sha256']!=input_hash or refmeta['tol']!=a.tol or refmeta['max_iters']!=a.max_iters:
            raise ValueError('reference inputs or stopping rule differ')
        if refmeta['initialization_threads']!=a.init_threads or refmeta['kernel_threads']!=a.threads:
            raise ValueError('reference and candidate thread policies differ')
        reference={key:refdata[key] for key in ['W','P','V','history','n_iter']}
        record['reference_source']='first timed original full fit, saved at '+str(reference_path)
    gc.collect();gc.disable();started=time.perf_counter();cpu=time.process_time()
    try:result,initialization=execute()
    finally:
        first=time.perf_counter()-started;first_cpu=time.process_time()-cpu;gc.enable()
    record.update(first_call_seconds=first,first_call_cpu_seconds=first_cpu,
                  first_initialization_seconds=initialization,first_n_iter=int(result['n_iter']))
    hashes={key:digest(result[key]) for key in ['W','P','V','history']}
    if a.mode=='original':
        reference={key:result[key] for key in ['W','P','V','history','n_iter']}
        refmeta=dict(input_sha256=input_hash,tol=a.tol,max_iters=a.max_iters,initialization_threads=a.init_threads,
                     kernel_threads=a.threads,source='pinned original SOMOLP.fit first timed cold fit',
                     original_source_sha256=record['original_source_sha256'],initialization='original_direct')
        reference_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(reference_path,**reference,metadata=np.asarray(json.dumps(refmeta,sort_keys=True)))
        reference_path.with_suffix('.metadata.json').write_text(json.dumps(refmeta,indent=2)+'\n')
        record['reference_source']='this first timed original full fit'
    record['cold_validation']=validate(result,reference)
    record['status']='running';record['repeat_bitwise_stable']=True;save()
    print(json.dumps(dict(event='cold_fit_complete',mode=a.mode,seconds=first,initialization_seconds=initialization,n_iter=int(result['n_iter']))),flush=True)
    for repetition in range(a.repeats):
        gc.collect();gc.disable();started=time.perf_counter();cpu=time.process_time()
        try:result,initialization=execute()
        finally:
            elapsed=time.perf_counter()-started;cpu_elapsed=time.process_time()-cpu;gc.enable()
        record['seconds'].append(elapsed);record['cpu_seconds'].append(cpu_elapsed)
        record['initialization_seconds'].append(initialization)
        record['kernel_plus_overhead_seconds'].append(elapsed-initialization if initialization is not None else None)
        record['validations'].append(validate(result,reference))
        record['repeat_bitwise_stable'] &= all(digest(result[key])==hashes[key] for key in hashes)
        record['repeats_completed']=repetition+1;record['n_iter']=int(result['n_iter']);save()
        print(json.dumps(dict(event='warm_fit_complete',mode=a.mode,repetition=repetition+1,seconds=elapsed,
                              initialization_seconds=initialization,n_iter=int(result['n_iter']))),flush=True)
    record['input_unchanged']={key:digest(ar) for key,ar in [('X',X),('R',R)]}==input_hash
    record['source_unchanged_during_measurement']=sources()==record['source_sha256']
    record['public_freeze_verified_after']=public_hashes()==public_freeze['files']
    checks=[record['cold_validation'],*record['validations']]
    valid=all(c['n_iter_match'] and all(c[key]['finite'] and c[key]['allclose_1e8'] for key in ['W','P','V','history']) for c in checks)
    record['status']='ok' if valid and record['input_unchanged'] and record['source_unchanged_during_measurement'] and record['public_freeze_verified_after'] else 'numerical_difference'
except BaseException as error:
    record.update(status='timeout' if isinstance(error,BudgetExpired) else 'error',error=repr(error),traceback=traceback.format_exc())
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
    save()
print(json.dumps({key:record.get(key) for key in ['mode','status','median_seconds','median_initialization_seconds','n_iter','repeats_completed','error']}),flush=True)
