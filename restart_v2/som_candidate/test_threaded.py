"""Tiny correctness and repeat-stability validation, no performance timing."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .threaded import run
from .oracle import run as oracle

def main():
    rng=np.random.default_rng(487);worst={k:0. for k in ['W','P','V','history']};count=0
    for n,d,m,k in [(1,1,1,1),(23,5,13,2),(19,8,11,3)]:
        X=rng.normal(size=(n,d));R=rng.normal(size=(m,k));W=rng.normal(size=(m,d));P=softmax(rng.normal(size=(n,m)),axis=1)
        for zero_mass in [False,True]:
            Q=P.copy()
            if zero_mass:Q[:,max(1,m//2):]=0;Q/=Q.sum(1,keepdims=True)
            original=[a.copy() for a in [X,R,W,Q]]
            for lam in [.001,.5,10.]:
                ref=oracle(X,R,W,Q,1.2,lam,7,-1,1)
                for threads in [1,2,4]:
                    for block in [3,64]:
                        out=run(X,R,W,Q,1.2,lam,7,-1,threads,block_rows=block)
                        for key in worst:
                            worst[key]=max(worst[key],float(np.max(np.abs(out[key]-ref[key]))))
                            np.testing.assert_allclose(out[key],ref[key],rtol=1e-9,atol=1e-9)
                        count+=1
            for a,b in zip([X,R,W,Q],original):np.testing.assert_array_equal(a,b)
        once=run(X,R,W,Q,1.2,.5,1,-1,4)
        if m>1:np.testing.assert_array_equal(once['W'][max(1,m//2):],W[max(1,m//2):])
        ref=oracle(X,R,W,P,1.2,.5,200,1e-4,1);a=run(X,R,W,P,1.2,.5,200,1e-4,4);b=run(X,R,W,P,1.2,.5,200,1e-4,4)
        assert a['n_iter']==ref['n_iter']
        for key in worst:
            np.testing.assert_allclose(a[key],ref[key],rtol=1e-8,atol=1e-8)
            np.testing.assert_array_equal(a[key],b[key])
        Xo=X+1e8;Wo=W+1e8
        ref=oracle(Xo,R,Wo,P,1.2,.5,5,-1,1);out=run(Xo,R,Wo,P,1.2,.5,5,-1,4)
        assert out['variant']=='threaded_translation_single_thread_fallback'
        for key in worst:np.testing.assert_allclose(out[key],ref[key],rtol=1e-8,atol=1e-8)
    result=dict(status='pass',tiny_configurations=count,worst_absolute=worst,repeat_stable=True,
                zero_mass_retention=True,input_unmodified=True,translation_fallback=True,stopping_checked=True)
    Path(__file__).with_name('threaded_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
