"""Tiny validation only; intentionally not a benchmark."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .kernel import run
from .oracle import run as reference


def main():
    rng=np.random.default_rng(1947)
    worst={k:0.0 for k in ('W','P','V','history')}
    count=0;offset_worst={k:0.0 for k in worst};fallback_cases=0
    for kind in ('normal','zero_mass','duplicates','allzero','offset','grid_offset'):
        X=rng.normal(size=(23,5));R=rng.normal(size=(13,2))
        W=rng.normal(size=(13,5));P=softmax(rng.normal(size=(23,13)),axis=1)
        if kind=='zero_mass':P[:,4:]=0;P/=P.sum(axis=1,keepdims=True)
        if kind=='duplicates':X[:]=X[0]
        if kind=='allzero':X[:]=0
        if kind=='offset':X+=1e8;W+=1e8
        if kind=='grid_offset':R+=1e8
        originals=[v.copy() for v in (X,R,W,P)]
        for lam in (0.001,0.2,10.0):
            for iters in (1,7):
                ref=reference(X,R,W,P,1.3,lam,iters,-1,1)
                for distance in ('direct','guarded'):
                    for block in (None,7):
                        out=run(X,R,W,P,1.3,lam,iters,-1,1,distance=distance,block_rows=block)
                        stress=kind in ('offset','grid_offset')
                        for key in worst:
                            error=float(np.max(np.abs(out[key]-ref[key])))
                            (offset_worst if stress else worst)[key]=max((offset_worst if stress else worst)[key],error)
                            np.testing.assert_allclose(out[key],ref[key],rtol=2e-5 if stress else 2e-10,
                                                       atol=2e-5 if stress else 2e-10,
                                                       err_msg=str((kind,lam,iters,distance,block,key)))
                        assert out['n_iter']==iters
                        if kind=='zero_mass' and iters==1:
                            np.testing.assert_array_equal(out['W'][4:],W[4:])
                        fallback_cases+=bool(out['direct_fallback_rows'])
                        count+=1
        for a,b in zip((X,R,W,P),originals):np.testing.assert_array_equal(a,b)
    for n,m,d,k in ((1,1,1,1),(17,11,3,3)):
        X=rng.normal(size=(n,d));R=rng.normal(size=(m,k));W=rng.normal(size=(m,d));P=softmax(rng.normal(size=(n,m)),axis=1)
        ref=reference(X,R,W,P,1.2,.5,200,1e-4,1)
        out=run(X,R,W,P,1.2,.5,200,1e-4,1)
        assert out['n_iter']==ref['n_iter']
        for key in worst:np.testing.assert_allclose(out[key],ref[key],rtol=1e-8,atol=1e-8)
        zero=run(X,R,W,P,1.2,.5,0,1e-4,1)
        assert zero['V'] is None and zero['n_iter']==0 and len(zero['history'])==0
        np.testing.assert_array_equal(zero['W'],W);np.testing.assert_array_equal(zero['P'],P)
    result=dict(status='pass',tiny_configurations=count,worst_normal=worst,worst_offset_stress=offset_worst,
                fallback_cases=fallback_cases,zero_mass_preserved=True,input_unmodified=True,
                note='Offset stress uses explicit 2e-5 tolerance; no universal trajectory claim')
    Path(__file__).with_name('tiny_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
