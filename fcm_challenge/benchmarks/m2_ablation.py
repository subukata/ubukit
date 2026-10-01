"""Compare positive-only m=2 formulas; not a guard-matched power-only study."""
import json,random,statistics,time
from pathlib import Path
import numpy as np
from benchmarks.benchmark import metadata
from ubukit_fcm.core import _memberships

def generic(q,exponent):
    out=q.min(axis=1,keepdims=True)/q
    np.power(out,exponent,out=out)
    out/=out.sum(axis=1,keepdims=True)
    return out

def main():
    rng=np.random.default_rng(815)
    order=random.Random(71)
    methods={'m2_reciprocal':lambda q:_memberships(q,2.),
             'generic_power_one':lambda q:generic(q,1.),
             'sqrt_then_power_two':lambda q:generic(np.sqrt(q),2.)}
    result={'metadata':metadata(),'note':'Positive squared distances; generic/sqrt variants omit the robust zero-distance guard; formula-level context only, not a guard-matched power-only study or headline speedup baseline; 9 interleaved warm repeats','cases':[]}
    for n,k in [(2000,5),(100000,8),(10000,64)]:
        q=np.exp(rng.normal(size=(n,k))*2)
        ref=methods['m2_reciprocal'](q)
        row={'n':n,'k':k,'methods':{name:{'seconds':[],'max_error':float(np.max(abs(fn(q)-ref)))} for name,fn in methods.items()}}
        for rep in range(9):
            names=list(methods);order.shuffle(names)
            for name in names:
                s=time.perf_counter();methods[name](q);t=time.perf_counter()-s
                row['methods'][name]['seconds'].append(t)
        for v in row['methods'].values():v['median_seconds']=statistics.median(v['seconds'])
        result['cases'].append(row)
    Path('results/ablation_m2.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
