import numpy as np,json
from .._support import portable_oracle
old=portable_oracle.joint_quality;P0=portable_oracle.ExecutionPolicy
from ubukit._impl.portable_accel import joint_quality as new,ExecutionPolicy as P1

def test_public_metrics():
    rng=np.random.default_rng(248);count=0
    for n,d in [(7,2),(31,16),(64,16),(101,32),(180,8)]:
     for kind in ['random','ties','duplicates','translated']:
      X=rng.normal(size=(n,d));Y=rng.normal(size=(n,2))
      if kind=='ties':X=np.round(X);Y=np.round(Y)
      if kind=='duplicates':X[n//2:]=X[:n-n//2];Y[n//2:]=Y[:n-n//2]
      if kind=='translated':X+=1e8
      ks=[1,min(5,(n-1)//2)]
      for backend in ['numpy','sqrt_numpy']+(['numba','sqrt_numba'] if __import__('importlib.util',fromlist=['find_spec']).find_spec('numba') else []):
       a=old(X,Y,ks,backend=backend,policy=P0(threads=1));b=new(X,Y,ks,backend=backend,policy=P1(threads=1));assert a==b,(n,d,kind,backend,a,b);count+=1
    print(json.dumps({'public_exact_cases':count,'backends':4 if __import__('importlib.util',fromlist=['find_spec']).find_spec('numba') else 2,'same_scores':True}))
