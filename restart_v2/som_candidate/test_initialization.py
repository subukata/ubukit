"""Tiny same-SVD initialization/fit comparisons, not a timing campaign."""
import json
from pathlib import Path
import numpy as np
from .initialization import initialize,fit
from .oracle import initialize as reference_initialize,run as reference_run

def main():
    rng=np.random.default_rng(819)
    errors=dict(W=0.,P0=0.,fit_W=0.,fit_P=0.,fit_V=0.,fit_history=0.)
    count=0
    for n,d,m,k in [(1,1,1,1),(29,5,16,2),(43,16,25,3),(19,7,13,2)]:
        for kind in ['normal','zero','offset']:
            X=rng.normal(size=(n,d));R=rng.normal(size=(m,k))
            if kind=='zero':X[:]=0
            if kind=='offset':X+=1e8
            for lam in [.001,.5,5.]:
                Wr,Pr=reference_initialize(X,R,lam,threads=1)
                for method in ['direct','same_svd_lowrank']:
                    W,P=initialize(X,R,lam,threads=1,method=method)
                    errors['W']=max(errors['W'],float(np.max(np.abs(W-Wr))))
                    errors['P0']=max(errors['P0'],float(np.max(np.abs(P-Pr))))
                    np.testing.assert_array_equal(W,Wr)
                    np.testing.assert_allclose(P,Pr,rtol=1e-9,atol=1e-10)
                    ref=reference_run(X,R,Wr,Pr,1.2,lam,5,-1,1)
                    out=fit(X,R,1.2,lam,5,-1,1,initialization=method)
                    for key in ['W','P','V','history']:
                        errors['fit_'+key]=max(errors['fit_'+key],float(np.max(np.abs(out[key]-ref[key]))))
                        np.testing.assert_allclose(out[key],ref[key],rtol=1e-8,atol=1e-8)
                    count+=1
    result=dict(status='pass',configurations=count,worst_absolute=errors,
                same_full_svd=True,no_randomized_or_truncated_svd=True)
    Path(__file__).with_name('initialization_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
