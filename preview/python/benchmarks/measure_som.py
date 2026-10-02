"""Reproducible bounded timing/accuracy smoke; not a performance guarantee."""
import json,math,platform,statistics,sys,time
from pathlib import Path
import numpy as np
import scipy
import ubukit


def reference(X,W,shape,batch,sigma=1.3,rate=.3):
    W=W.tolist();X=X.tolist();m=len(W);d=len(W[0]);r=[(j%shape[0],j//shape[0]) for j in range(m)]
    def bmu(x):return min(range(m),key=lambda j:sum((x[f]-W[j][f])**2 for f in range(d)))
    def h(j,b):return math.exp(-sum((r[j][k]-r[b][k])**2 for k in (0,1))/(2*sigma*sigma))
    if batch:
        labels=[bmu(x) for x in X];new=[w[:] for w in W]
        for j in range(m):
            weights=[h(j,b) for b in labels];den=math.fsum(weights)
            if den:
                for f in range(d):new[j][f]=math.fsum(weights[i]*X[i][f] for i in range(len(X)))/den
        return np.array(new)
    for x in X:
        b=bmu(x)
        for j in range(m):
            a=rate*h(j,b)
            for f in range(d):W[j][f]+=a*(x[f]-W[j][f])
    return np.array(W)


def timed(f,repeats=5):
    f();times=[];last=None
    for _ in range(repeats):
        start=time.perf_counter();last=f();times.append(time.perf_counter()-start)
    return last,statistics.median(times),times


rng=np.random.default_rng(2048)
X=rng.normal(size=(200,8));W=rng.normal(size=(64,8));out=[]
for name,fit,batch in [('som',ubukit.som,False),('som_batch',ubukit.som_batch,True)]:
    kwargs=dict(grid_shape=(8,8),initial_prototypes=W,epochs=1,sigma=1.3,sigma_end=1.3)
    if not batch:kwargs.update(learning_rate=.3,learning_rate_end=.3)
    actual,seconds,raw=timed(lambda:fit(X,**kwargs))
    oracle,base,base_raw=timed(lambda:reference(X,W,(8,8),batch),3)
    error=float(np.max(np.abs(actual['centers']-oracle)))
    np.testing.assert_allclose(actual['centers'],oracle,rtol=1e-12,atol=1e-12)
    out.append(dict(algorithm=name,N=200,D=8,M=64,epochs=1,optimized_median_seconds=seconds,
                    scalar_reference_median_seconds=base,reference_over_optimized=base/seconds,
                    maximum_absolute_error=error,optimized_samples_seconds=raw,
                    reference_samples_seconds=base_raw,primary_scratch_budgeted_bytes=actual['primary_scratch_budgeted_bytes']))
X=rng.normal(size=(2000,32));W=rng.normal(size=(256,32))
for name,fit,epochs in [('som',ubukit.som,2),('som_batch',ubukit.som_batch,20)]:
    actual,seconds,raw=timed(lambda:fit(X,grid_shape=(16,16),initial_prototypes=W,epochs=epochs),3)
    out.append(dict(algorithm=name,N=2000,D=32,M=256,epochs=epochs,median_seconds=seconds,
                    samples_seconds=raw,iterations=actual['iterations'],unit=actual['unit'],
                    primary_scratch_budgeted_bytes=actual['primary_scratch_budgeted_bytes']))
result=dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
            ubukit=ubukit.__version__,threads=1,scope='Local warm smoke; full API includes initialization and final assignment; scalar oracle is correctness reference, not a tuned competing package.',results=out)
print(json.dumps(result,indent=2))
