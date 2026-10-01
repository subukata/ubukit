"""Actual restart-v2 public API versus two sklearn calls; no kernel edits."""
import os,time,json,hashlib,random,statistics,gc
from pathlib import Path
from datetime import datetime,timezone
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='9'
os.environ['OMP_WAIT_POLICY']='PASSIVE';os.environ['GOMP_SPINCOUNT']='0'
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:9])
import numpy as np
from sklearn.manifold import trustworthiness
from threadpoolctl import threadpool_limits,threadpool_info
from portable_accel import joint_quality,ExecutionPolicy
root=Path(__file__).resolve().parents[1];out=root/'evidence/public_metrics_digits/summary.json'
if out.exists():raise RuntimeError('Do not overwrite an existing campaign')
policy=ExecutionPolicy(threads=9,block_rows=256,max_scratch_bytes=32*2**20)
source={str(p.relative_to(root/'portable_accel')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'portable_accel').rglob('*.py'))}
def check_source():
 for n,h in source.items():assert hashlib.sha256((root/'portable_accel'/n).read_bytes()).hexdigest()==h
with threadpool_limits(limits=9):
 tiny=np.random.default_rng(0).normal(size=(17,5));t=time.perf_counter()
 for b in ('numpy','numba','sqrt_numba'):joint_quality(tiny,tiny[:,:2],2,backend=b,policy=policy)
 trustworthiness(tiny,tiny[:,:2],n_neighbors=2);warmup=time.perf_counter()-t
from sklearn.datasets import load_digits
rng=np.random.default_rng(20261001);X=np.ascontiguousarray(load_digits().data,dtype=np.float64);W=rng.normal(size=(64,2))
with threadpool_limits(limits=1):Y=X@W/np.sqrt(64)
def hashes():return {name:hashlib.sha256(x.tobytes()).hexdigest() for name,x in (('X',X),('Y',Y))}
before=hashes();reference_started=time.perf_counter()
reference=joint_quality(X,Y,[5,15],backend='numpy',policy=policy,rank_method='broadcast',max_distance_bytes=2*2**30)
expected={'scores':[[q.trustworthiness,q.continuity] for q in reference],'penalties':[[q.trustworthiness_penalty,q.continuity_penalty] for q in reference]}
reference_setup_seconds=time.perf_counter()-reference_started
methods={'sklearn_twice':lambda:[(float(trustworthiness(X,Y,n_neighbors=k)),float(trustworthiness(Y,X,n_neighbors=k))) for k in (5,15)]}
for backend in ('numpy','numba','sqrt_numba'):
 methods['public_'+backend]=lambda b=backend:joint_quality(X,Y,[5,15],backend=b,policy=policy,rank_method='broadcast' if b=='numpy' else 'scan',max_distance_bytes=2*2**30)
report={'scope':'actual portable_accel.joint_quality versus two actual sklearn calls per k (four total), tie-heavy Digits control','started_utc':datetime.now(timezone.utc).isoformat(),'status':'running','dataset':'sklearn bundled Digits; all1797 rows, original integer-valued features, seeded linear2D projection','reference_setup_seconds':reference_setup_seconds,'shape':[1797,64],'k':[5,15],'seed':20261001,'threads':9,'block_rows':256,'max_scratch_bytes':32*2**20,'rank_methods':{'numpy':'broadcast','numba':'scan','sqrt_numba':'scan'},'source_sha256':source,'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':before,'tiny_warmup_seconds':warmup,'methods':{n:{'warm_seconds':[],'validations':[]} for n in methods},'order':[],'first_call_is_after_tiny_runtime_warmup':True}
shuffle=random.Random(20261001)
with threadpool_limits(limits=9):
 for repeat in range(4):
  order=list(methods);shuffle.shuffle(order);report['order'].append(order)
  for name in order:
   check_source();gc.collect();gc.disable();t=time.perf_counter();tc=time.process_time()
   try:r=methods[name]();elapsed=time.perf_counter()-t;cpu=time.process_time()-tc
   finally:gc.enable()
   if name=='sklearn_twice':scores=[list(q) for q in r];penalties=None
   else:scores=[[q.trustworthiness,q.continuity] for q in r];penalties=[[q.trustworthiness_penalty,q.continuity_penalty] for q in r]
   valid={'scores':scores,'penalties':penalties,'exact_scores':scores==expected['scores'],'exact_penalties':penalties==expected['penalties'] if penalties is not None else None}
   assert valid['exact_scores'] and valid['exact_penalties'] is not False
   e=report['methods'][name];e['validations'].append(valid)
   if repeat==0:e['first_call_seconds']=elapsed
   else:
    e['warm_seconds'].append(elapsed);e.setdefault('warm_cpu_seconds',[]).append(cpu);med=statistics.median(e['warm_seconds']);e.update(median_seconds=med,mad_seconds=statistics.median(abs(v-med) for v in e['warm_seconds']))
   out.write_text(json.dumps(report,indent=2)+'\n');print(repeat,name,round(elapsed,6),'exact',flush=True)
check_source();assert hashes()==before
report.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),input_unchanged=True,source_unchanged=True,threadpools=threadpool_info())
base=report['methods']['sklearn_twice']['median_seconds'];report['ratios']={name:base/e['median_seconds'] for name,e in report['methods'].items() if name!='sklearn_twice'}
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'complete','ratios':report['ratios'],'path':str(out)}),flush=True)
