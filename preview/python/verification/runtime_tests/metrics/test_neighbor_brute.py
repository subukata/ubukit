import numpy as np,json
from sklearn.neighbors import NearestNeighbors
from portable_accel._backends.metrics_portable._neighbor_queries import neighbor_queries

def test_neighbor_brute():
    rng=np.random.default_rng(812);count=0
    for n in [7,8,31,32,101,300]:
     for d in [1,2,8,16,32]:
      for dtype in [np.float32,np.float64]:
       for kind in ['random','ties','duplicates','tiny','translated','near']:
        X=rng.normal(size=(n,d)).astype(dtype)
        if kind=='ties':X=np.round(X)
        if kind=='duplicates':X[n//2:]=X[:n-n//2]
        if kind=='tiny':X*=dtype(1e-20)
        if kind=='translated':X+=dtype(1e8)
        if kind=='near':X=np.nextafter(np.ones_like(X),X+1)
        ks=list(dict.fromkeys([1,min(3,(n-1)//2),min(15,(n-1)//2),(n-1)//2]))[::-1]
        original=np.concatenate([NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False) for k in ks],axis=1)
        got=neighbor_queries(X,ks);assert np.array_equal(original,got),(n,d,dtype,kind,ks)
        count+=1
    print(json.dumps({'exact_neighbor_cases':count,'sklearn_version':'1.8.0','tie_policy':'unchanged','includes':'odd-N fit-threshold, reversed k, duplicates, float32/64, translated and tiny'}))
    
    # Verified nonzero-self-distance counterexample to a no-ties-only approach.
    rng=np.random.default_rng(27);Z=1e8+rng.normal(size=(64,16))*1e4;Z[:3]=1e8+rng.normal(size=(3,16))*.75
    for ks in [[1,2],[1,3],[2,3],[1,15],[62,63]]:
     ref=np.concatenate([NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False) for k in ks],axis=1)
     assert np.array_equal(neighbor_queries(Z,ks),ref),ks
    print('self-position counterexample retained exact behavior')
    
    # Raw small-heap boundary tie with self inside the candidate prefix.
    rng=np.random.default_rng(59);Z=1e8+rng.normal(size=(64,16))*1e4;Z[:3]=1e8+rng.normal(size=(3,16))*.75
    for ks in [[1,2],[1,3],[2,3],[1,15]]:
     ref=np.concatenate([NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False) for k in ks],axis=1)
     assert np.array_equal(neighbor_queries(Z,ks),ref),ks
    print("raw boundary counterexample retained exact behavior")
