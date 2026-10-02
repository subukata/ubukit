"""Reproducible rational/Decimal oracle and preserved-baseline checks; no timings."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys, pathlib, importlib.util, json, hashlib, traceback, math
from fractions import Fraction
from decimal import Decimal, localcontext
import numpy as np
import inspect
from portable_accel.som_olp_localized import run_som_olp_localized as localized
from portable_accel.som_olp import run_som_olp as original
from portable_accel.som_olp import run_som_olp as frozen
from portable_accel.policy import ExecutionPolicy as Policy
FrozenPolicy = Policy
SOURCE = pathlib.Path(inspect.getsourcefile(localized))
SOURCE_HASH = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
assert SOURCE_HASH == '115887d87e6523cee1f4663cfd6c46cdb672a0a76028de4bd6b71f1da825785a'
BASELINE_SOURCE = pathlib.Path(inspect.getsourcefile(frozen))
assert hashlib.sha256(BASELINE_SOURCE.read_bytes()).hexdigest() == '175fb02ceefd7cdb6c026a285601f130c6a23d29d8a8543e384c13ee25d6af28'
# The distribution retains the original baseline implementation. Package-wide
# unchanged-module evidence is supplied by staging/SOURCE_MANIFEST.json.
report=dict(groups=[],failures=[],measurements=[],cases=0)

def check(name,fn):
    try:
        n=fn() or 1;report['cases']+=n;report['groups'].append(dict(name=name,status='passed',cases=n))
    except Exception as e:
        report['groups'].append(dict(name=name,status='failed'));report['failures'].append(dict(name=name,error=str(e),traceback=traceback.format_exc()))

def call(fn, X,R,W,P,**kwargs):
    opts=dict(gamma=.2,lam=.4,max_iters=1,tol=0.,backend='cdist_optimized',policy=FrozenPolicy(threads=1,block_rows=2) if fn is frozen else Policy(threads=1,block_rows=2));opts.update(kwargs)
    return fn(X,R,W,P,**opts)

def equal(a,b):
    for key in ('W','P','V','history','n_iter'):
        if a[key] is None: assert b[key] is None
        else: np.testing.assert_array_equal(a[key],b[key],err_msg=key)

def fixture(seed=2,n=7,d=3,m=4,q=2):
    rng=np.random.default_rng(seed);X=rng.uniform(-2,2,(n,d));R=rng.uniform(-1,1,(m,q));W=rng.uniform(-1,1,(m,d));P=rng.uniform(.1,1,(n,m));P[:,-1]*=1e-300;P/=P.sum(axis=1)[:,None]
    return X,R,W,P

def high_precision_means(X,R,W,P):
    w=W.copy();v=np.zeros((len(X),R.shape[1]));F=lambda v:Fraction(float(v))
    for j in range(len(R)):
        den=sum(map(F,P[:,j]),Fraction(0))
        if den:
            for f in range(X.shape[1]):w[j,f]=float(sum((F(x)*F(p) for x,p in zip(X[:,f],P[:,j])),Fraction(0))/den)
    for i in range(len(X)):
        for h in range(R.shape[1]):v[i,h]=float(sum((F(r)*F(p) for r,p in zip(R[:,h],P[i])),Fraction(0)))
    return w,v

def cost_oracle(X,R,W,V,gamma,lam):
    D=lambda v:Decimal.from_float(float(v))
    with localcontext() as c:
        c.prec=750;total=D(0);probs=[];scale=D(0)
        for i,x in enumerate(X):
            costs=[sum(((D(a)-D(b))**2 for a,b in zip(x,w)),D(0))+D(gamma)*sum(((D(a)-D(b))**2 for a,b in zip(V[i],r)),D(0)) for w,r in zip(W,R)]
            minimum=min(costs);zs=[-(cost-minimum)/D(lam) for cost in costs]
            weights=[z.exp() if z>-10000 else D(0) for z in zs];den=sum(weights,D(0));logden=den.ln()
            total+=minimum-D(lam)*logden;scale+=abs(minimum)+abs(D(lam)*logden)
            probs.append([float(w/den) for w in weights])
        return float(total),np.array(probs),float(scale)

def decimals():
    n=0
    for seed in range(12):
        X,R,W,P=fixture(seed,n=3+seed%5,d=1+seed%4,m=2+seed%5,q=1+seed%3)
        out=call(localized,X,R,W,P);assert out['variant']=='experimental_localized_probability_tails'
        ew,ev=high_precision_means(X,R,W,P)
        np.testing.assert_allclose(out['W'],ew,rtol=5e-14,atol=2e-15)
        np.testing.assert_allclose(out['V'],ev,rtol=5e-14,atol=2e-15)
        objective,prob,scale=cost_oracle(X,R,out['W'],out['V'],.2,.4)
        err=abs(out['history'][0]-objective);assert err<=8e-14*max(scale,1e-300)
        np.testing.assert_allclose(out['P'],prob,rtol=5e-13,atol=2e-15)
        report['measurements'].append(dict(seed=seed,objective_absolute_error=err,objective_error_over_scale=err/max(scale,1e-300),max_mean_absolute_error=float(np.max(abs(out['W']-ew)))))
        n+=1
    return n
check('Independent exact-rational means and 750-digit direct objective oracle',decimals)

def threshold():
    n=0
    for marker in [1e-140,np.nextafter(1e-140,0.),np.nextafter(1e-140,np.inf),1e-141,1e-139,0.]:
        X,R,W,P=fixture();P[:,-1]=marker;P[:,:-1]/=P[:,:-1].sum(1)[:,None]
        out=call(localized,X,R,W,P)
        if marker and marker<1e-140:assert out['variant']=='experimental_localized_probability_tails'
        else:equal(out,call(frozen,X,R,W,P))
        n+=1
    for value in [np.nextafter(1e140,np.inf),np.nextafter(1e-140,0.),1e200,1e-200]:
        for target in ('X','R','W'):
            X,R,W,P=fixture();dict(X=X,R=R,W=W)[target][0,0]=value
            try:
                out=call(localized,X,R,W,P,gamma=0. if target!='R' else .2)
            except ValueError as error:
                try:call(frozen,X,R,W,P,gamma=0. if target!='R' else .2)
                except ValueError as old_error:assert str(error)==str(old_error)
                else:raise AssertionError('candidate-only exception')
            else:
                equal(out,call(frozen,X,R,W,P,gamma=0. if target!='R' else .2))
                assert out['variant']=='extreme_float64_exponent_fallback'
            n+=1
    for gamma in [np.nextafter(1e-140,0.),np.nextafter(1e140,np.inf)]:
        X,R,W,P=fixture();equal(call(localized,X,R,W,P,gamma=gamma),call(frozen,X,R,W,P,gamma=gamma));n+=1
    return n
check('Dispatch thresholds and unchanged coordinate/gamma extreme fallback',threshold)

def cancellation():
    n=0
    tiny=np.nextafter(0.,1.)
    for mass in [tiny,2*tiny,1e-320,1e-300,1e-141,1e-139]:
        for residual in [tiny,1e-300,1e-16]:
            # Tiny class must update; a second class cancels leading signed terms.
            X=np.array([[1.],[-1.],[1.]]);R=np.array([[0.],[1.],[2.]])
            P=np.array([[1.,mass,0.],[1.,mass,0.],[residual,mass,1.]])
            W=np.array([[99.],[88.],[77.]])
            out=call(localized,X,R,W,P,gamma=0.,lam=1.)
            old=call(frozen,X,R,W,P,gamma=0.,lam=1.)
            np.testing.assert_array_equal(out['W'],old['W']);assert out['W'][1,0]!=88.;n+=1
    # Three half-subnormal products must aggregate before their only rounding.
    X=np.ones((1,1));R=np.array([[1.],[-1.],[.5],[.5],[.5],[2.]])
    P=np.array([[.5,.5,tiny,tiny,tiny,0.]]);W=np.array([[2.],[2.],[2.],[2.],[2.],[7.]])
    out=call(localized,X,R,W,P,gamma=0.,lam=1.);assert out['V'][0,0]==2*tiny;assert out['W'][-1,0]==7.;n+=1
    return n
check('Exact cancellation, subnormal product aggregation, tiny-mass model updates and empty columns',cancellation)

def ownership():
    n=0
    for steps in [0,1,2,5]:
        X,R,W,P=fixture(n=8,d=4,m=5,q=2)
        # Non-contiguous, read-only inputs.
        args=tuple(np.stack([a,a],axis=2)[:,:,0] for a in (X,R,W,P));snap=[a.copy() for a in args]
        for a in args:a.setflags(write=False)
        out=call(localized,*args,max_iters=steps);again=call(localized,*args,max_iters=steps)
        equal(out,again)
        for a,b in zip(args,snap):np.testing.assert_array_equal(a,b)
        for key in ('W','P','V','history'):
            if out[key] is not None:assert not any(np.shares_memory(out[key],a) for a in args)
        n+=1
    for backend in ['gemm_guarded','gemm_centered','cdist','numpy','auto','cdist_optimized']:
        X,R,W,P=fixture();out=call(localized,X,R,W,P,backend=backend,max_iters=0);equal(out,call(frozen,X,R,W,P,backend=backend,max_iters=0));n+=1
    for backend in ['gemm_guarded','gemm_centered']:
        X,R,W,P=fixture();equal(call(localized,X,R,W,P,backend=backend,max_iters=2),call(frozen,X,R,W,P,backend=backend,max_iters=2));n+=1
    return n
check('Data ownership, determinism, zero iterations and explicitly nonlocal backend retention',ownership)

def costs():
    n=0
    for lam in [np.nextafter(0.,1.),1e-300,1e-10,1.,1e140,1e300]:
        X=np.array([[0.],[1.],[2.]]);R=np.array([[0.],[1.]]);W=np.zeros((2,1));P=np.array([[1.,1e-300],[1.,1e-300],[1e-300,1.]])
        out=call(localized,X,R,W,P,gamma=0.,lam=lam)
        expected=call(frozen,X,R,W,P,gamma=0.,lam=lam)
        for key in ['W','V','P','history']:np.testing.assert_allclose(out[key],expected[key],rtol=5e-14,atol=1e-300)
        assert np.all(np.isfinite(out['history']));n+=1
    X=np.zeros((2,1));R=np.array([[0.],[1e100]]);W=np.zeros((2,1));P=np.array([[1.,1e-300],[1.,1e-300]])
    out=call(localized,X,R,W,P,gamma=1e140,lam=1.);equal(out,call(frozen,X,R,W,P,gamma=1e140,lam=1.));assert out['direct_fallback_rows']==2;n+=1
    return n
check('Extreme temperatures, cold cost rows and composite-overflow fallback',costs)

def budgets_and_trajectories():
    n=0
    X,R,W,P=fixture();nn,d=X.shape;m,q=R.shape
    minimum=8*(3*m*d+3*m+3*nn*q+d+q)+10*m+128
    for cap in [8*m-1,8*m,16*m-1,16*m,minimum-1,minimum,minimum+10*m+128,10000]:
        try:out=call(localized,X,R,W,P,policy=Policy(threads=1,block_rows=7,max_scratch_bytes=cap))
        except ValueError as error:
            try:call(frozen,X,R,W,P,policy=FrozenPolicy(threads=1,block_rows=7,max_scratch_bytes=cap))
            except ValueError as old_error:assert str(error)==str(old_error)
            else:raise AssertionError('candidate rejects valid cold scratch cap')
        else:
            if cap<minimum:equal(out,call(frozen,X,R,W,P,policy=FrozenPolicy(threads=1,block_rows=7,max_scratch_bytes=cap)))
            else:assert out['primary_scratch_budgeted_bytes']<=cap
        n+=1
    drifts=[]
    for seed in range(8):
        args=fixture(seed,n=8,d=3,m=5,q=2)
        a=call(localized,*args,max_iters=8,tol=0.);b=call(frozen,*args,max_iters=8,tol=0.)
        drift={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['W','P','V','history']}
        for k in ['W','P','V','history']:np.testing.assert_allclose(a[k],b[k],rtol=5e-10,atol=5e-12)
        drifts.append(dict(seed=seed,candidate_n_iter=a['n_iter'],frozen_n_iter=b['n_iter'],max_absolute_difference=drift,bitwise_equal=all(np.array_equal(a[k],b[k]) for k in ['W','P','V','history'])))
        n+=1
    report['bounded_trajectory_differences']=drifts
    return n
check('Scratch boundary fallback parity and bounded eight-step trajectory drift',budgets_and_trajectories)

def amplified_grid_underflow():
    n=0
    for q in [1,2,5]:
        for gamma in [1e100,1e120,1e140]:
            for tail in [1e-170,1e-180,1e-200]:
                X=np.array([[0.],[2e-125]]);R=np.array([[0.],[1.],[-1.]]).repeat(q,axis=1)
                P=np.array([[1.,tail,0.],[1.,0.,tail]]);W=np.zeros((3,1))
                a=call(localized,X,R,W,P,gamma=gamma,lam=1e-250);b=call(frozen,X,R,W,P,gamma=gamma,lam=1e-250)
                for k in ['W','V','P','history']:np.testing.assert_allclose(a[k],b[k],rtol=3e-14,atol=0.)
                n+=1
    return n
check('Gamma-amplified grid square underflow hidden by ordinary data costs',amplified_grid_underflow)

report['source_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
assert SOURCE_HASH==report['source_sha256'], 'source changed during test run'
report['status']='failed' if report['failures'] else 'passed';print(json.dumps(report,indent=2));sys.exit(bool(report['failures']))
