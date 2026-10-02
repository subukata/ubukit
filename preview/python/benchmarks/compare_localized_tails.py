"""Serial paired dev4-versus-opt-in candidate timings. Run under shared flock."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse,hashlib,inspect,json,platform,signal,statistics,sys,time
from pathlib import Path
import numpy as np
import scipy
from sklearn.datasets import load_iris,load_wine,load_breast_cancer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info
import ubukit as uk
from portable_accel.som_olp_localized import run_som_olp_localized,fit_som_olp_localized

parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
def timeout(*args):raise TimeoutError('3-second bounded individual measurement')
signal.signal(signal.SIGALRM,timeout)
policy=uk.ExecutionPolicy(threads=1,block_rows=128)
rows=[]
def measure(fn):
    start=time.perf_counter();signal.setitimer(signal.ITIMER_REAL,3.)
    try:r=fn();return dict(seconds=time.perf_counter()-start,state='completed'),r
    except TimeoutError as e:return dict(seconds=time.perf_counter()-start,state='timeout',error=str(e)),None
    finally:signal.setitimer(signal.ITIMER_REAL,0)
def compare(name,X,R,W,P,params,repeats=3,fit=False):
    kw=dict(gamma=params.get('gamma',.2),lam=params.get('lam',.4),max_iters=params.get('max_iters',5),tol=params.get('tol',0.),backend='cdist_optimized',policy=policy)
    if fit:
        kw['pca_scale']=params.get('pca_scale',2.)
        funcs=dict(dev4=lambda:uk.fit_som_olp(X,R,**kw),candidate=lambda:fit_som_olp_localized(X,R,**kw))
    else:funcs=dict(dev4=lambda:uk.run_som_olp(X,R,W,P,**kw),candidate=lambda:run_som_olp_localized(X,R,W,P,**kw))
    record=dict(case=name,N=len(X),D=X.shape[1],M=len(R),fit_includes_initialization=fit,parameters={k:v for k,v in kw.items() if k!='policy'},measurements={},agreement={})
    outputs={}
    for version in funcs:
        first,r=measure(funcs[version]);record['measurements'][version]=dict(first_call=first,warm=[]);outputs[version]=r
    for repeat in range(repeats):
        for version in (('dev4','candidate') if repeat%2==0 else ('candidate','dev4')):
            if record['measurements'][version]['first_call']['state']=='timeout':continue
            timing,r=measure(funcs[version]);record['measurements'][version]['warm'].append(timing);outputs[version]=r
    for version,values in record['measurements'].items():
        times=[v['seconds'] for v in values['warm'] if v['state']=='completed']
        values['warm_median_seconds']=statistics.median(times) if times else None
        r=outputs[version]
        if r is not None:values['result_metadata']={k:r[k] for k in ('variant','n_iter','exact_prototype_repairs','exact_grid_repairs','direct_fallback_rows','primary_scratch_budgeted_bytes') if k in r}
    a=outputs['dev4'];b=outputs['candidate']
    if a is not None and b is not None:
        for key in ('W','V','P','history'):
            if a[key].shape==b[key].shape:
                record['agreement'][key]=dict(max_absolute_error=float(np.max(np.abs(a[key]-b[key]),initial=0)),allclose_1e_10=bool(np.allclose(a[key],b[key],rtol=1e-10,atol=1e-12)),bitwise=bool(np.array_equal(a[key],b[key])))
            else:record['agreement'][key]=dict(shape_mismatch=[list(a[key].shape),list(b[key].shape)])
    old=record['measurements']['dev4']['warm_median_seconds'];new=record['measurements']['candidate']['warm_median_seconds']
    if old and new:record['speedup_dev4_over_candidate']=old/new
    elif record['measurements']['dev4']['first_call']['state']=='timeout' and new:record['speedup_lower_bound']=3/new
    rows.append(record);save();print(name,record.get('speedup_dev4_over_candidate',record.get('speedup_lower_bound')),flush=True)
def save():
    data=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,ubukit=uk.__version__,threadpools=threadpool_info(),scope='Fixed dev4 direct-backend request versus separate opt-in localization; paired alternating warm samples, no historical naive comparator; 3-second timeout is censored, not completed baseline timing',candidate_path=inspect.getsourcefile(run_som_olp_localized),candidate_sha256=hashlib.sha256(Path(inspect.getsourcefile(run_som_olp_localized)).read_bytes()).hexdigest(),rows=rows)
    out.write_text(json.dumps(data,indent=2)+'\n')
rng=np.random.default_rng(20261002)
for name,n,d,m,tail,iterations in [('ordinary_small',16,2,4,False,5),('ordinary_medium',256,16,32,False,5),('tail_small',16,2,4,True,5),('tail_medium',128,8,16,True,5),('tail_skinny',512,4,32,True,3),('tail_wide',96,64,32,True,3),('tail_large',1024,16,64,True,3)]:
    X=rng.normal(size=(n,d));R=rng.normal(size=(m,2));W=rng.normal(size=(m,d));P=rng.random((n,m))
    if tail:P[:,-1]*=1e-300
    P/=P.sum(1,keepdims=True)
    compare(name,X,R,W,P,dict(max_iters=iterations))
for name,loader,params in [('iris_known_tail',load_iris,dict(gamma=3.6375361652737697,lam=.10334288639248834,pca_scale=2.59009286733778,max_iters=100,tol=1e-5)),('wine_low_lambda',load_wine,dict(gamma=.1,lam=.1,max_iters=100,tol=1e-5)),('breast_cancer_default',load_breast_cancer,dict(gamma=.1,lam=.5,max_iters=100,tol=1e-5))]:
    X=np.ascontiguousarray(StandardScaler().fit_transform(loader().data));R=np.array([(a,b) for a in np.linspace(-1,1,8) for b in np.linspace(-1,1,8)])
    compare(name,X,R,None,None,params,fit=True)
