"""Sequential scikit-learn timing on exactly the stored JS benchmark labels."""
import json,platform,statistics,time,warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
assert sklearn.__version__=='1.8.0',sklearn.__version__
base=Path(__file__).resolve().parent.parent
cases=json.loads((base/'fixtures/benchmark-inputs.json').read_text())['cases']
fns={'ari':adjusted_rand_score,'ami':adjusted_mutual_info_score,'joint':lambda a,b:dict(ari=adjusted_rand_score(a,b),ami=adjusted_mutual_info_score(a,b))}
rows=[]
for case in cases:
    a=np.asarray(case['a'],dtype=np.int32);b=np.asarray(case['b'],dtype=np.int32)
    times={};values={}
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      for name,fn in fns.items():
        for _ in range(3):fn(a,b)
        ts=[]
        for _ in range(9):
          start=time.perf_counter();values[name]=fn(a,b);ts.append((time.perf_counter()-start)*1000)
        times[name]=dict(medianMs=statistics.median(ts),minMs=min(ts),maxMs=max(ts),repeats=len(ts))
    rows.append(dict(name=case['name'],n=len(a),times=times,values=values))
    print(case['name'],times,flush=True)
result=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,timing='serial; 3 warmups, 9 measured calls; median wall milliseconds; array construction outside timing; joint means two separate sklearn public calls',rows=rows)
(base/'reports/benchmark-sklearn.json').write_text(json.dumps(result,indent=2)+'\n')
