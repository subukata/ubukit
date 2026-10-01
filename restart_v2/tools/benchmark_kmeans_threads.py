"""Bounded actual-public thread sensitivity; no algorithm changes.

Each thread budget uses fresh low/high-D processes and three interleaved warm
repetitions per method, with final labels/inertia included and exact-label
checks. Thread-budget order is fixed independently of the measurements.
"""
import json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1]
out=root/'evidence/kmeans_thread_sensitivity'
out.mkdir(parents=True,exist_ok=True)
summary=out/'summary.json'
if summary.exists():raise RuntimeError('Do not overwrite an existing campaign')
start=time.monotonic();record={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'budget_order':[4,1,9,2],'repeats':3,'scope':'Fresh process per fixture/budget; actual public finalization included; existing fixed-init fixtures; no scratch NumPy timing','groups':[]}
for threads in record['budget_order']:
    remaining=280-(time.monotonic()-start)
    if remaining<=0:raise TimeoutError('overall280-second budget exhausted')
    target=out/('t'+str(threads))
    cmd=[sys.executable,'-m','kmeans_candidate.benchmark_matched','--threads',str(threads),'--repeats','3','--budget-seconds',str(remaining),'--output',str(target)]
    with (out/('t'+str(threads)+'.log')).open('w') as log:
        proc=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
    group=json.loads((target/'summary.json').read_text());record['groups'].append({'threads':threads,**group})
    summary.write_text(json.dumps(record,indent=2)+'\n')
    if proc.returncode or group['status']!='matched':raise RuntimeError('thread control failed; retain record without ranking')
record.update(status='matched',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
summary.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'elapsed_seconds':record['elapsed_seconds']}))
