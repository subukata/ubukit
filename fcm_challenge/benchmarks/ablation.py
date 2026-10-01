"""Isolated, paired membership-expression ablations; no concurrent fitting.

The private update function is swapped outside timed regions so every variant
uses exactly the same fit loop, validation, distance computation and outputs.
This is a research harness, not a thread-safe public extension mechanism.
"""
import hashlib,json,random,statistics,time
from pathlib import Path
import numpy as np
from benchmarks.benchmark import data, metadata
from ubukit_fcm import core
from ubukit_fcm.log_membership import (membership_logsoftmax,membership_logsumexp,
                                      membership_scipy_softmax,membership_naive_logsoftmax)
BASE=core._memberships
METHODS={'inverse_power':BASE,'logsoftmax_log1p':membership_logsoftmax,
         'logsumexp_log1p':membership_logsumexp,'scipy_softmax_log1p':membership_scipy_softmax,
         'plain_logsoftmax':membership_naive_logsoftmax}

def main():
    rng=np.random.default_rng(419)
    result={'metadata':metadata(),'notes':[
        'Kernel timing excludes distance computation; data are deterministic positive squared distances',
        'Full-fit timing uses exact same SciPy distance/fit driver; private membership kernel swapped outside timing',
        'All methods warm once, then seeded interleaving: kernel 5 repeats, fit 3 repeats; one BLAS thread',
        'plain_logsoftmax uses direct log(q), which loses near-tie information for extreme q,m; included for speed context, not recommended robust path',
        'Inverse power has m=2 reciprocal specialization and guarded log/log1p fallbacks only for extreme ranges or m near 1',
        'No O(NK²) pairwise-ratio baseline is used'], 'kernel':[],'fit':[]}
    order=random.Random(197)
    try:
        for n,k in [(2000,5),(100000,8),(10000,64)]:
            q=np.exp(rng.normal(size=(n,k))*2)
            for m in [1.3,2.,3.]:
                ref=BASE(q,m)
                row={'n':n,'k':k,'m':m,'methods':{}}
                for name,fn in METHODS.items():
                    out=fn(q,m)
                    row['methods'][name]={'seconds':[],'max_error':float(np.max(abs(out-ref)))}
                for rep in range(5):
                    names=list(METHODS);order.shuffle(names)
                    for name in names:
                        start=time.perf_counter(); METHODS[name](q,m);elapsed=time.perf_counter()-start
                        row['methods'][name]['seconds'].append(elapsed)
                for v in row['methods'].values():v['median_seconds']=statistics.median(v['seconds'])
                result['kernel'].append(row)
        for case in ['small_overlap','large_n','high_d']:
            x,u,truth=data(case)
            for m in [1.3,2.,3.]:
                row={'case':case,'m':m,'iterations':30,'methods':{}}
                core._memberships=BASE
                ref=core.fit_fcm(x,init=u,m=m,backend='scipy',max_iter=30,tol=0)
                for name,fn in METHODS.items():
                    core._memberships=fn
                    got=core.fit_fcm(x,init=u,m=m,backend='scipy',max_iter=30,tol=0)
                    row['methods'][name]={'seconds':[],'membership_max_error':float(np.max(abs(got['membership']-ref['membership']))),
                        'objective_relative_error':abs(got['objective']-ref['objective'])/max(abs(ref['objective']),1e-300)}
                for rep in range(3):
                    names=list(METHODS);order.shuffle(names)
                    for name in names:
                        core._memberships=METHODS[name]
                        start=time.perf_counter(); core.fit_fcm(x,init=u,m=m,backend='scipy',max_iter=30,tol=0); elapsed=time.perf_counter()-start
                        row['methods'][name]['seconds'].append(elapsed)
                for v in row['methods'].values():v['median_seconds']=statistics.median(v['seconds'])
                result['fit'].append(row)
                Path('results/ablation_logsoftmax.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({'case':case,'m':m,'times':{n:v['median_seconds'] for n,v in row['methods'].items()}}),flush=True)
    finally:
        core._memberships=BASE
    Path('results/ablation_logsoftmax.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
