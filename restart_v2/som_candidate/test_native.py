"""Fresh tiny validation of the rebuilt optional native extension."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .native import run
from .oracle import run as reference

def main():
    rng=np.random.default_rng(914);worst={key:0. for key in ['W','P','V','history']};count=0
    for n,d,m,k in [(1,1,1,1),(23,5,13,2),(19,9,11,3)]:
        X=rng.normal(size=(n,d));R=rng.normal(size=(m,k));W=rng.normal(size=(m,d));P=softmax(rng.normal(size=(n,m)),axis=1)
        for zero in [False,True]:
            Q=P.copy()
            if zero:Q[:,max(1,m//2):]=0;Q/=Q.sum(1,keepdims=True)
            originals=[ar.copy() for ar in (X,R,W,Q)]
            for lam in [.001,.5,10.]:
                ref=reference(X,R,W,Q,1.2,lam,7,-1,1)
                for threads in [1,2,4]:
                    for distance in ['guarded','direct','gemm']:
                        out=run(X,R,W,Q,1.2,lam,7,-1,threads,distance=distance,block_rows=7)
                        for key in worst:
                            worst[key]=max(worst[key],float(np.max(np.abs(out[key]-ref[key]))))
                            np.testing.assert_allclose(out[key],ref[key],rtol=1e-9,atol=1e-9)
                        count+=1
            for a,b in zip((X,R,W,Q),originals):np.testing.assert_array_equal(a,b)
        if m>1:
            out=run(X,R,W,Q,1.2,.5,1,-1,2)
            np.testing.assert_array_equal(out['W'][max(1,m//2):],W[max(1,m//2):])
        ref=reference(X,R,W,P,1.2,.5,200,1e-4,1);out=run(X,R,W,P,1.2,.5,200,1e-4,2)
        assert out['n_iter']==ref['n_iter']
        for key in worst:np.testing.assert_allclose(out[key],ref[key],rtol=1e-8,atol=1e-8)
    result=dict(status='pass',tiny_configurations=count,worst_absolute=worst,inputs_unmodified=True,
                zero_mass_preserved=True,stopping_checked=True,new_build=True)
    Path(__file__).with_name('native_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
