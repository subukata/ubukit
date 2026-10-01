"""Serial fresh-process SOM screen; invoke only in an authorized timing slot."""
import argparse,json,os,random,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--output',required=True)
p.add_argument('--threads',type=int,nargs='+',default=[9]);p.add_argument('--iters',type=int,default=30)
p.add_argument('--repeats',type=int,default=3);p.add_argument('--timeout',type=int,default=180)
a=p.parse_args();here=Path(__file__).resolve().parent;root=here.parent
methods=[('original','som_candidate.oracle:run',{}),('guarded_numpy','som_candidate.kernel:run',{'distance':'guarded'}),
         ('threadpool','som_candidate.threaded:run',{'distance':'guarded','block_rows':256}),
         ('native','som_candidate.native:run',{'distance':'guarded','block_rows':256})]
jobs=[(name,method,kw,t) for t in a.threads for name,method,kw in methods]
random.Random(13097).shuffle(jobs);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['PYTHONPATH']=str(root);env['OMP_WAIT_POLICY']='PASSIVE';env['GOMP_SPINCOUNT']='0'
for i,(name,method,kwargs,t) in enumerate(jobs):
    target=output/f'{name}_t{t}.json'
    cmd=[sys.executable,str(here/'bench_worker.py'),'--fixture',a.fixture,'--method',method,
         '--threads',str(t),'--max-iters',str(a.iters),'--repeats',str(a.repeats),
         '--kwargs',json.dumps(kwargs),'--output',str(target)]
    print(f'[{i+1}/{len(jobs)}] {name} t{t}',flush=True)
    try:
        result=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=a.timeout)
        print(result.stdout,flush=True)
        if result.returncode:
            target.with_suffix('.stderr').write_text(result.stderr);print(result.stderr[-1500:],flush=True)
    except subprocess.TimeoutExpired:
        target.write_text(json.dumps(dict(status='timeout',method=method,threads=t))+'\n');print('TIMEOUT',flush=True)
