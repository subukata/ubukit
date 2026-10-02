"""Independent combinatorial checks, differential cases, and API validation."""
import itertools
import json
import os
import importlib.util
import math
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score as sklearn_ari
from sklearn.metrics import adjusted_mutual_info_score as sklearn_ami
import ubukit._impl.external_metrics as em

warnings.simplefilter("ignore", UserWarning)
HERE = Path(os.environ["UBUKIT_EXTERNAL_RESULTS"]).resolve()
NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None


def partitions(n):
    if not n:
        yield ()
        return
    def go(prefix, mx):
        if len(prefix) == n:
            yield tuple(prefix)
        else:
            for k in range(mx + 2):
                yield from go(prefix + [k], max(mx, k))
    yield from go([0], 0)


def brute_ari(x, y):
    tp = fp = fn = tn = 0
    for i in range(len(x)):
        for j in range(i):
            a, b = x[i] == x[j], y[i] == y[j]
            if a and b: tp += 1
            elif b: fp += 1
            elif a: fn += 1
            else: tn += 1
    if fp == fn == 0:
        return 1.0
    return float(Fraction(2 * (tp * tn - fp * fn), (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)))


def independent_ami(x, y, average):
    n = len(x)
    if not n:
        return 1.0
    groups_x = [frozenset(i for i, v in enumerate(x) if v == c) for c in set(x)]
    groups_y = [frozenset(i for i, v in enumerate(y) if v == c) for c in set(y)]
    ka, kb = len(groups_x), len(groups_y)
    if ka == kb == 1 or set(groups_x) == set(groups_y):
        return 1.0
    if ka == 1 or kb == 1:
        return 0.0
    if ka == n or kb == n:
        return None if average == "min" else 0.0
    a, b = [len(z) for z in groups_x], [len(z) for z in groups_y]
    hx = math.fsum(-ai / n * math.log(ai / n) for ai in a)
    hy = math.fsum(-bi / n * math.log(bi / n) for bi in b)
    mi = math.fsum(len(u & v) / n * math.log(Fraction(n * len(u & v), len(u) * len(v)))
                   for u in groups_x for v in groups_y if u & v)
    terms = []
    for ai in a:
        for bi in b:
            denom = math.comb(n, bi)
            for k in range(max(1, ai + bi - n), min(ai, bi) + 1):
                probability = Fraction(math.comb(ai, k) * math.comb(n - ai, bi - k), denom)
                terms.append(float(probability) * k / n * math.log(Fraction(n * k, ai * bi)))
    emi = math.fsum(terms)
    normalizer = {"arithmetic": (hx + hy) / 2, "geometric": math.sqrt(hx * hy), "min": min(hx, hy), "max": max(hx, hy)}[average]
    return (mi - emi) / (normalizer - emi)


def run():
    started = time.perf_counter()
    counts = {"independent_ari": 0, "independent_ami": 0, "sklearn_ari": 0, "sklearn_ami_regular": 0, "sklearn_ami_singular": 0, "backend_parity": 0, "invariance": 0, "invalid": 0, "overflow": 0}
    maxerrs = {"independent_ami": 0.0, "sklearn_ami_regular": 0.0, "backend_parity": 0.0}
    singular = []
    methods = ("arithmetic", "geometric", "min", "max")
    # Compile optional backend before test loops; timing is not a benchmark.
    if NUMBA_AVAILABLE:
        em.adjusted_mutual_info_score([0,0,1,1],[0,1,0,1],backend="numba")
    for n in range(6):
        parts = list(partitions(n))
        for x in parts:
            for y in parts:
                ours = em.adjusted_rand_score(x, y)
                assert ours == brute_ari(x, y), (n, x, y, ours, brute_ari(x, y))
                counts["independent_ari"] += 1
                assert ours == sklearn_ari(x, y)
                counts["sklearn_ari"] += 1
                for m in methods:
                    ref = independent_ami(x, y, m)
                    got = em.adjusted_mutual_info_score(x, y, average_method=m)
                    if ref is not None:
                        err = abs(got - ref)
                        assert err < 3e-13, (n, x, y, m, got, ref)
                        maxerrs["independent_ami"] = max(maxerrs["independent_ami"], err)
                        counts["independent_ami"] += 1
                    if NUMBA_AVAILABLE:
                        nb = em.adjusted_mutual_info_score(x, y, average_method=m, backend="numba")
                        assert abs(nb - got) < 3e-13
                        counts["backend_parity"] += 1
                        maxerrs["backend_parity"] = max(maxerrs["backend_parity"],abs(nb-got))
                    sk = sklearn_ami(x, y, average_method=m)
                    is_singular = m == "min" and n > 1 and (len(set(x)) == n or len(set(y)) == n) and len(set(x)) != len(set(y)) and min(len(set(x)),len(set(y))) > 1
                    if is_singular:
                        counts["sklearn_ami_singular"] += 1
                        assert got == sk
                        if abs(got-1.0) > 1e-10 and len(singular)<40:
                            singular.append(dict(n=n,x=x,y=y,average=m,candidate=got,sklearn=sk))
                    else:
                        assert abs(got-sk) < 3e-12, (n,x,y,m,got,sk)
                        counts["sklearn_ami_regular"] += 1
                        maxerrs["sklearn_ami_regular"] = max(maxerrs["sklearn_ami_regular"],abs(got-sk))
    print("exhaustive complete",counts,flush=True)
    rng = np.random.default_rng(8102026)
    fixtures = []
    for n in [2,3,7,17,100,1000,10000]:
        for k,l in [(2,2),(3,7),(20,30),(max(2,n//2),max(2,n//3))]:
            for shape in ["random", "imbalanced", "correlated"]:
                if shape=="imbalanced":
                    x = np.minimum(rng.zipf(2.1,n)-1,k-1)
                    y = np.minimum(rng.zipf(1.6,n)-1,l-1)
                else:
                    x = rng.integers(k,size=n)
                    y = rng.integers(l,size=n)
                    if shape=="correlated":
                        mask = rng.random(n)<.92
                        y[mask]=x[mask]
                fixtures.append((shape,x,y))
    fixtures += [
      ("negative",np.array([-9,7,-9,999,7]),np.array([5,5,6,7,6])),
      ("uint64",np.array([2**64-1,2**64-2,2**64-1,2**64-3],dtype=np.uint64),np.array([0,1,1,2])),
      ("int64min",np.array([-2**63,-2**63+1,-2**63,-2**63+2]),np.array([0,1,1,2])),
      ("bool",np.array([True,False,True,False]),np.array([0,1,1,2])),
      ("bigobject",np.array([2**80,2**80,2**81,2**82],dtype=object),np.array([0,1,1,2])),
      ("strings",np.array(["cat","dog","cat","raccoon"]),np.array(["a","a","b","b"])),
      ("floatlabels",np.array([0.,1.,1.,2.,2.]),np.array([0.,0.,1.,1.,2.])),
      ("noncontiguous",np.arange(60)[::3]%5,np.arange(60)[::-3]%7),
    ]
    near_x=np.arange(1000);near_y=np.arange(1000);near_x[1]=0;near_y[3]=2
    fixtures.append(("almost_singletons",near_x,near_y))
    assert em._ami(*em._contingency(near_x,near_y),"arithmetic","numpy") is None
    random_errors=[]
    for label,x,y in fixtures:
        ari=em.adjusted_rand_score(x,y);skari=sklearn_ari(x,y)
        assert ari == skari,(label,ari,skari)
        counts["sklearn_ari"]+=1
        for m in methods:
            got = em.adjusted_scores(x,y,average_method=m)
            sk = sklearn_ami(x,y,average_method=m)
            assert got['ari'] == ari
            if NUMBA_AVAILABLE:
                nb = em.adjusted_scores(x,y,average_method=m,backend="numba")
                assert nb['ari'] == ari
                assert abs(got['ami']-nb['ami']) < 1e-11, (label,m,got,nb)
                counts['backend_parity']+=1
                maxerrs['backend_parity']=max(maxerrs['backend_parity'],abs(got['ami']-nb['ami']))
            singular_case=m=='min' and max(np.unique(x).size,np.unique(y).size)==len(x) and min(np.unique(x).size,np.unique(y).size)>1 and np.unique(x).size!=np.unique(y).size
            if not singular_case:
                err=abs(got['ami']-sk)
                assert err < 1e-8, (label,len(x),m,got,sk)
                maxerrs['sklearn_ami_regular']=max(maxerrs['sklearn_ami_regular'],err)
                counts['sklearn_ami_regular']+=1
                if err>1e-10: random_errors.append(dict(label=label,n=len(x),method=m,error=err,candidate=got['ami'],sklearn=sk))
            else: counts['sklearn_ami_singular']+=1
            swapped=em.adjusted_mutual_info_score(y,x,average_method=m)
            order=rng.permutation(len(x))
            permuted=em.adjusted_mutual_info_score(x[order],y[order],average_method=m)
            # Relabel to strings also covers generic encoding and gaps.
            relabelled=em.adjusted_mutual_info_score(np.array([f'x_{v}' for v in x]),np.array([f'y_{v}' for v in y]),average_method=m)
            assert max(abs(got['ami']-z) for z in [swapped,permuted,relabelled])<1e-8
            counts['invariance']+=3
    for x,y in [([1],[1,2]),([[1],[2]],[1,2]),([float('nan')],[0]),([float('inf')],[0]),([1+1j],[0])]:
        for fn in [em.adjusted_rand_score,em.adjusted_mutual_info_score,em.adjusted_scores]:
            try: fn(x,y)
            except ValueError: counts['invalid']+=1
            else: raise AssertionError((x,y,fn))
    for option in [{'average_method':'median'},{'backend':'automatic'}]:
        try: em.adjusted_mutual_info_score([],[],**option)
        except ValueError: counts['invalid']+=1
        else: raise AssertionError(option)
    # Synthetic tables exercise int64 overflow without allocating billions of labels.
    for scale in [1,10**6,10**10,10**12]:
        cells=np.array([2*scale,scale,scale,3*scale],dtype=np.int64)
        a=np.array([3*scale,4*scale],dtype=np.int64)
        b=np.array([3*scale,4*scale],dtype=np.int64);n=7*scale
        q=sum(int(z)**2 for z in cells);sa=sum(int(z)**2 for z in a);sb=sum(int(z)**2 for z in b)
        tp,fp,fn,tn=q-n,sb-q,sa-q,n*n-sa-sb+q
        ref=float(Fraction(2*(tp*tn-fp*fn),(tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)))
        assert abs(em._ari(n,a,b,cells)-ref)<2e-16
        counts['overflow']+=1
    report=dict(status='passed',optional_numba_tested=NUMBA_AVAILABLE,elapsed_seconds=time.perf_counter()-started,counts=counts,max_absolute_errors=maxerrs,
                singular_definition='For one all-singleton partition with nontrivial other partition, min normalization is mathematically0/0. Candidate lazily delegates singular/ill-conditioned cases to sklearn, preserving its rounding-dependent values.',
                singular_sklearn_compatibility_examples=singular,larger_regular_discrepancies=random_errors)
    (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': run()
