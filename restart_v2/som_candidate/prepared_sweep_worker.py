"""Fresh-public versus prepare+K full-fit workload, one mode per process.

Every warmed prepared workload creates a NEW snapshot/full SVD inside its timer,
then performs K fresh-state fits. The cached per-fit times are reported separately
but never substituted for the total preparation+K result. Reference IO and
validation are outside timing. No original-class long oracle is duplicated.
"""
import argparse,gc,hashlib,json,os,pathlib,resource,signal,time,traceback
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixture',required=True);p.add_argument('--mode',choices=['fresh','prepared'],required=True)
p.add_argument('--threads',type=int,default=9);p.add_argument('--repeats',type=int,default=3)
p.add_argument('--max-iters',type=int,default=1000);p.add_argument('--tol',type=float,default=1e-4)
p.add_argument('--reference-dir',required=True);p.add_argument('--output',required=True)
p.add_argument('--cases',default='[[2.5,2.2],[2.767,2.439],[3.0,2.7]]')
p.add_argument('--budget-seconds',type=int,default=420)
a=p.parse_args();cases=[dict(gamma=float(v[0]),lam=float(v[1])) for v in json.loads(a.cases)]
if min(a.threads,a.repeats,a.max_iters,a.budget_seconds)<1 or not cases:p.error('positive sizes and at least one case required')
if hasattr(os,'sched_getaffinity'):
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:a.threads])
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]=str(a.threads)
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE');os.environ.setdefault('GOMP_SPINCOUNT','0')
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from portable_accel import fit_som_olp,ExecutionPolicy
from som_candidate.prepared_svd import PreparedSOM
folder=pathlib.Path(__file__).resolve().parent
z=np.load(a.fixture,allow_pickle=False);X=z['X'];R=z['R']
metadata=json.loads(str(z['metadata'])) if 'metadata' in z else {}
policy=ExecutionPolicy(threads=a.threads,block_rows=256,max_scratch_bytes=64<<20)
references=pathlib.Path(a.reference_dir);output=pathlib.Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)

def digest(array):return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def public_sources():
    root=folder.parent/'portable_accel'
    return {str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob('*.py')) if '__pycache__' not in f.parts}
input_hash=dict(X=digest(X),R=digest(R));public_hashes=public_sources()
record=dict(mode=a.mode,scope='prepare_plus_K_full_fits' if a.mode=='prepared' else 'K_fresh_public_full_fits',
            cases=cases,K=len(cases),threads=a.threads,initialization='same full SVD; svd_lowrank scoring',
            repeats_requested=a.repeats,repeats_completed=0,max_iters=a.max_iters,tol=a.tol,
            policy=dict(threads=a.threads,block_rows=256,max_scratch_bytes=64<<20),
            input_sha256=input_hash,fixture_metadata=metadata,public_source_sha256=public_hashes,
            candidate_source_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['prepared_svd.py','prepared_sweep_worker.py']},
            started_utc=now(),workload_seconds=[],preparation_seconds=[],fit_seconds=[],cpu_seconds=[],validations=[],
            timing_scope='Owned snapshot/copy/full-SVD preparation (prepared mode), all K initializations, public kernels and output materialization inside timer; input loading/reference IO/validation excluded',
            accounting='A new preparation is charged in EACH warmed prepared workload; first-fit and steady-fit times are not substituted for prepare+K total')


def workload():
    outputs=[];fit_times=[];preparation=0.;prepared=None;info=None
    if a.mode=='prepared':
        start=time.perf_counter();prepared=PreparedSOM(X,max_rank=R.shape[1],threads=a.threads)
        preparation=time.perf_counter()-start;info=prepared.describe()
    for case in cases:
        start=time.perf_counter()
        if prepared is None:
            out=fit_som_olp(X,R,**case,max_iters=a.max_iters,tol=a.tol,
                           backend='threadpool',initializer='svd_lowrank',policy=policy)
        else:
            out=prepared.fit(R,**case,max_iters=a.max_iters,tol=a.tol,policy=policy)
        fit_times.append(time.perf_counter()-start);outputs.append(out)
    return outputs,prepared,dict(preparation_seconds=preparation,fit_seconds=fit_times,
                                  n_iter=[int(out['n_iter']) for out in outputs],prepared_info=info)


def save_reference(outputs):
    references.mkdir(parents=True,exist_ok=True)
    meta=dict(input_sha256=input_hash,cases=cases,threads=a.threads,max_iters=a.max_iters,tol=a.tol,
              source='first timed fresh-public K-fit workload',public_source_sha256=public_hashes)
    (references/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    for i,out in enumerate(outputs):
        np.savez(references/f'case_{i}.npz',**{k:out[k] for k in ['W','P','V','history','n_iter']})


def validate(outputs):
    checks=[]
    for i,out in enumerate(outputs):
        ref=np.load(references/f'case_{i}.npz',allow_pickle=False)
        row=dict(case=i,n_iter=int(out['n_iter']),reference_n_iter=int(ref['n_iter']),checks={})
        for key in ['W','P','V','history']:
            expected=ref[key];value=np.asarray(out[key]);shape=value.shape==expected.shape
            row['checks'][key]=dict(finite=bool(np.all(np.isfinite(value))),allclose_1e8=bool(shape and np.allclose(value,expected,rtol=1e-8,atol=1e-8)),
                                    max_abs=float(np.max(np.abs(value-expected))) if shape and value.size else (0. if shape else None))
        row['ok']=row['n_iter']==row['reference_n_iter'] and all(v['finite'] and v['allclose_1e8'] for v in row['checks'].values())
        ref.close();checks.append(row)
    return checks


def save():
    record['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record['threadpools']=threadpool_info()
    if record['workload_seconds']:
        values=np.asarray(record['workload_seconds']);record['median_workload_seconds']=float(np.median(values))
        record['mad_workload_seconds']=float(np.median(np.abs(values-np.median(values))))
        record['median_preparation_seconds']=float(np.median(record['preparation_seconds']))
        record['median_fit_seconds_by_case']=np.median(record['fit_seconds'],axis=0).tolist()
    output.write_text(json.dumps(record,indent=2)+'\n')

class BudgetExpired(TimeoutError):pass
def expire(signum,frame):raise BudgetExpired('prepared workload budget expired')
signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,a.budget_seconds)
try:
    threadpool_limits(a.threads,user_api='blas')
    if a.mode=='prepared':
        refmeta=json.loads((references/'metadata.json').read_text())
        for key,expected in [('input_sha256',input_hash),('cases',cases),('threads',a.threads),('max_iters',a.max_iters),('tol',a.tol)]:
            if refmeta[key]!=expected:raise ValueError('reference contract mismatch: '+key)
    gc.collect();gc.disable();start=time.perf_counter();cpu=time.process_time()
    try:outputs,prepared,details=workload()
    finally:first=time.perf_counter()-start;first_cpu=time.process_time()-cpu;gc.enable()
    record.update(first_workload_seconds=first,first_cpu_seconds=first_cpu,first_details=details)
    initial_hashes=[{key:digest(out[key]) for key in ['W','P','V','history']} for out in outputs]
    if a.mode=='fresh':save_reference(outputs)
    record['cold_validation']=validate(outputs);record['status']='running';record['repeat_bitwise_stable']=True
    if details['prepared_info'] is not None:record['prepared_info']=details['prepared_info']
    del outputs,prepared;gc.collect();save()
    print(json.dumps(dict(event='cold_workload_complete',mode=a.mode,seconds=first,details=details)),flush=True)
    for iteration in range(a.repeats):
        gc.collect();gc.disable();start=time.perf_counter();cpu=time.process_time()
        try:outputs,prepared,details=workload()
        finally:elapsed=time.perf_counter()-start;used_cpu=time.process_time()-cpu;gc.enable()
        record['workload_seconds'].append(elapsed);record['cpu_seconds'].append(used_cpu)
        record['preparation_seconds'].append(details['preparation_seconds']);record['fit_seconds'].append(details['fit_seconds'])
        record['validations'].append(validate(outputs))
        record['repeat_bitwise_stable'] &= all(digest(out[key])==initial_hashes[i][key] for i,out in enumerate(outputs) for key in initial_hashes[i])
        record['repeats_completed']=iteration+1
        del outputs,prepared;gc.collect();save()
        print(json.dumps(dict(event='warm_workload_complete',mode=a.mode,repetition=iteration+1,seconds=elapsed,details=details)),flush=True)
    record['input_unchanged']=dict(X=digest(X),R=digest(R))==input_hash
    record['public_source_unchanged']=public_sources()==public_hashes
    validations=[record['cold_validation'],*record['validations']]
    valid=all(row['ok'] for group in validations for row in group)
    record['status']='ok' if valid and record['input_unchanged'] and record['public_source_unchanged'] else 'numerical_difference'
except BaseException as error:
    record.update(status='timeout' if isinstance(error,BudgetExpired) else 'error',error=repr(error),traceback=traceback.format_exc())
finally:
    signal.setitimer(signal.ITIMER_REAL,0);record['completed_utc']=now();save()
print(json.dumps({key:record.get(key) for key in ['mode','status','median_workload_seconds','median_preparation_seconds','median_fit_seconds_by_case','repeats_completed','error']}),flush=True)
