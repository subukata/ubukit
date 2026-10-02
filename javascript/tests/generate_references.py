"""Rebuild the scikit-learn 1.8 reference panel; no production dependency."""
import json, warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
assert sklearn.__version__ == '1.8.0', sklearn.__version__
rng=np.random.default_rng(20261002)
cases=[]
def add(name,a,b):
    a=np.asarray(a); b=np.asarray(b)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cases.append(dict(name=name,a=a.tolist(),b=b.tolist(),ari=adjusted_rand_score(a,b),ami={m:adjusted_mutual_info_score(a,b,average_method=m) for m in ['arithmetic','geometric','min','max']}))
for n in [0,1,2,3,4,5,10,20,50,100,257,1000]:
    add(f'constant-{n}',np.zeros(n,int),np.zeros(n,int))
    add(f'identical-{n}',np.arange(n)%max(1,n//3),np.arange(n)%max(1,n//3))
for i in range(140):
    n=int(rng.integers(4,501)); ka=int(rng.integers(2,min(n,50))); kb=int(rng.integers(2,min(n,50)))
    a=rng.integers(ka,size=n); b=rng.integers(kb,size=n)
    add(f'random-{i}',a,b)
for n,k in [(1000,7),(1000,50),(10000,10),(10000,1000),(3000,1500)]:
    a=np.arange(n)%k; b=rng.permutation(a); add(f'balanced-{n}-{k}',a,b)
for dataset in [load_iris,load_wine,load_breast_cancer]:
    x,y=dataset(return_X_y=True)
    pred=KMeans(n_clusters=len(np.unique(y)),n_init=10,random_state=0).fit_predict(StandardScaler().fit_transform(x))
    add(dataset.__name__,y,pred)
add('strings',['a','a','b','b','c','c'],['green','green','green','red','red','red'])
add('different-label-types',['a','a','b','b','c','c'],[0,0,0,1,1,1])
path=Path(__file__).resolve().parent.parent/'fixtures/sklearn-1.8.json'
path.write_text(json.dumps({'sklearn':sklearn.__version__,'numpy':np.__version__,'seed':20261002,'cases':cases},separators=(',',':'))+'\n')
print('Wrote',len(cases),'cases to',path)
