"""Tiny prepared-SVD equivalence/immutability checks; no large benchmark."""
import json
from pathlib import Path
import numpy as np
from portable_accel import ExecutionPolicy,fit_som_olp
from .initialization import initialize
from .prepared_svd import PreparedSOM

def main():
    rng=np.random.default_rng(924)
    for invalid in [np.ones((3,4),dtype=bool),np.ones((3,4),dtype=complex),np.full((3,4),'1')]:
        try:PreparedSOM(invalid)
        except TypeError:pass
        else:raise AssertionError('non-real source accepted')
    wide_int=np.full((3,3),2**60,dtype=np.int64);integer_snapshot=PreparedSOM(wide_int)
    changed_int=wide_int.copy();changed_int[0,0]+=1
    try:integer_snapshot.assert_same_input(changed_int)
    except ValueError:pass
    else:raise AssertionError('raw integer change hidden by float64 rounding')
    worst={key:0. for key in ['W0','P0','W','P','V','history']};count=0
    policy=ExecutionPolicy(threads=1,block_rows=8,max_scratch_bytes=64<<20)
    for kind in ['normal','duplicates','offset','zero','float32','uint8']:
        X=rng.normal(size=(23,5));R=rng.normal(size=(13,2))
        if kind=='duplicates':X[1:]=X[0]
        if kind=='offset':X+=1e8
        if kind=='zero':X[:]=0
        if kind=='float32':X=X.astype(np.float32)
        if kind=='uint8':X=np.clip(10*X+40,0,255).astype(np.uint8)
        original=X.copy();prepared=PreparedSOM(X,max_rank=2,threads=1)
        prepared.assert_same_input(original)
        invalids=[original.astype(np.float64 if original.dtype!=np.float64 else np.float32),original.reshape(5,23)]
        changed=original.copy();changed[0,0]+=1.;invalids.append(changed)
        for bad in invalids:
            try:prepared.assert_same_input(bad)
            except ValueError:pass
            else:raise AssertionError('changed input identity accepted')
        try:prepared._rank=1
        except AttributeError:pass
        else:raise AssertionError('prepared metadata mutation accepted')
        for gamma,lam,scale in [(1.2,.3,2.),(2.4,.6,1.5),(.2,.001,2.)]:
            W0,P0=prepared.initialize(R,lam,pca_scale=scale)
            Wr,Pr=initialize(original,R,lam,scale,1,method='same_svd_lowrank')
            for key,value,ref in [('W0',W0,Wr),('P0',P0,Pr)]:
                worst[key]=max(worst[key],float(np.max(np.abs(value-ref))))
                np.testing.assert_allclose(value,ref,rtol=1e-10,atol=1e-10)
            ref=fit_som_olp(original,R,gamma=gamma,lam=lam,max_iters=7,tol=0.,pca_scale=scale,
                           backend='threadpool',initializer='svd_lowrank',policy=policy)
            out=prepared.fit(R,gamma=gamma,lam=lam,max_iters=7,tol=0.,pca_scale=scale,policy=policy)
            assert out['n_iter']==ref['n_iter']
            for key in ['W','P','V','history']:
                worst[key]=max(worst[key],float(np.max(np.abs(out[key]-ref[key]))))
                np.testing.assert_allclose(out[key],ref[key],rtol=1e-8,atol=1e-8)
            count+=1
        # Mutating the original caller array must not change the owned snapshot.
        before=prepared.initialize(R,.6)
        X[:]=123.
        after=prepared.initialize(R,.6)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
        arrays=[prepared._X,prepared._mean,prepared._projection,prepared._basis,prepared._singular]
        assert all(not a.flags.writeable for a in arrays)
        for array in arrays:
            try:array.setflags(write=True)
            except ValueError:pass
            else:raise AssertionError('immutable array flag could be reopened')
    X=rng.normal(size=(31,6));R=rng.normal(size=(17,2));prepared=PreparedSOM(X,max_rank=2)
    ref=fit_som_olp(X,R,gamma=1.2,lam=.5,max_iters=200,tol=1e-4,backend='threadpool',initializer='svd_lowrank',policy=policy)
    out=prepared.fit(R,gamma=1.2,lam=.5,max_iters=200,tol=1e-4,policy=policy)
    assert out['n_iter']==ref['n_iter']
    for key in ['W','P','V','history']:np.testing.assert_allclose(out[key],ref[key],rtol=1e-8,atol=1e-8)
    try:prepared.initialize(rng.normal(size=(17,3)),.5)
    except ValueError:pass
    else:raise AssertionError('larger rank was silently accepted')
    result=dict(status='pass',parameterized_tiny_fits=count,worst_absolute=worst,
                original_mutation_isolated=True,owned_arrays_readonly=True,
                fresh_state_per_fit=True,rank_capacity_checked=True,changed_input_shape_dtype_checked=True,convergence_n_iter=out['n_iter'])
    Path(__file__).with_name('prepared_svd_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
