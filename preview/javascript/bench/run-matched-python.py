"""Same-input dev3 NumPy/Numba and sklearn measurements for the dev4 JS panel."""
import hashlib,importlib.metadata,json,os,platform,statistics,time,warnings
from pathlib import Path
import numpy as np
import sklearn,numba
import external_metrics as candidate
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
base=Path(__file__).resolve().parent.parent
raw=(base/'fixtures/matched-10000-k50.json').read_bytes();fixture=json.loads(raw)
a=np.asarray(fixture['a'],dtype=np.int32);b=np.asarray(fixture['b'],dtype=np.int32)
assert sklearn.__version__=='1.8.0'
assert importlib.metadata.version('ubukit-bundled-local-preview')=='0.0.0.dev3'
# Confirm a real installed distribution owns these exact module bytes.
dist=importlib.metadata.distribution('ubukit-bundled-local-preview');entry=next(p for p in dist.files if str(p)=='external_metrics.py')
assert Path(dist.locate_file(entry)).resolve()==Path(candidate.__file__).resolve()
import base64
assert base64.urlsafe_b64encode(hashlib.sha256(Path(candidate.__file__).read_bytes()).digest()).rstrip(b'=').decode()==entry.hash.value
warmups=20;repeats=21;batch_size=5
functions={
 'python_dev3_numpy':{'ari':candidate.adjusted_rand_score,'ami':candidate.adjusted_mutual_info_score,'joint':candidate.adjusted_scores},
 'python_dev3_numba_warm':{'ami':lambda a,b:candidate.adjusted_mutual_info_score(a,b,backend='numba'),'joint':lambda a,b:candidate.adjusted_scores(a,b,backend='numba')},
 'sklearn_1_8':{'ari':adjusted_rand_score,'ami':adjusted_mutual_info_score,'joint':lambda a,b:dict(ari=adjusted_rand_score(a,b),ami=adjusted_mutual_info_score(a,b))}}
results={};cold={}
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 for implementation,fns in functions.items():
  results[implementation]={}
  for name,fn in fns.items():
   if implementation=='python_dev3_numba_warm' and name=='ami':
    start=time.perf_counter();fn(a,b);cold['firstNumbaAMICallMs']=(time.perf_counter()-start)*1000
   for _ in range(warmups):fn(a,b)
   times=[];value=None
   for _ in range(repeats):
    start=time.perf_counter()
    for _ in range(batch_size):value=fn(a,b)
    times.append((time.perf_counter()-start)*1000/batch_size)
   results[implementation][name]=dict(medianMs=statistics.median(times),minMs=min(times),maxMs=max(times),rawMs=times,value=value)
result=dict(name=fixture['name'],n=len(a),kTrue=len(np.unique(a)),kPred=len(np.unique(b)),seed=fixture['seed'],fixtureSha256=hashlib.sha256(raw).hexdigest(),python=platform.python_version(),packageVersion=dist.version,numpy=np.__version__,numba=numba.__version__,sklearn=sklearn.__version__,moduleSha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),installedRecordVerified=True,threadSettings={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']},numbaThreads=numba.get_num_threads(),numbaARINote='ARI has no separate Numba implementation; use python_dev3_numpy ARI result',numbaCold=cold,warmups=warmups,repeats=repeats,batchSize=batch_size,timing='serial median wall milliseconds per invocation from batches of five; array construction excluded; validation/encoding/contingency included; arithmetic AMI; sklearn joint is two public calls; Numba warm excludes first-call compilation/cache-loading cost',results=results)
(base/'reports/matched-python.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
