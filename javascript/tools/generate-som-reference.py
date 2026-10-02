"""Independent scalar sample-by-unit textbook reference (stdlib only)."""
import copy,json,math,pathlib

def schedule(a,b,t,T,kind):
    p=0 if T<=1 else t/(T-1)
    if p==0:return a
    if p==1:return b
    return (1-p)*a+p*b if kind=='linear' or a==0 or b==0 else math.exp((1-p)*math.log(a)+p*math.log(b))

def run(case):
    X=case['data'];o=case['options'];W=copy.deepcopy(case['initialPrototypes']);width,height=o['gridShape'];T=o['maxIterations'];history=[]
    bmu=lambda x,w:min(range(len(w)),key=lambda j:sum((a-b)**2 for a,b in zip(x,w[j])))
    for t in range(T):
        sigma=schedule(o['sigma'],o['sigmaEnd'],t,T,o['schedule'])
        def h(a,b):
            ds=(a%width-b%width)**2+(a//width-b//width)**2
            return (1 if ds==0 else 0) if sigma==0 else math.exp(-ds/(2*sigma*sigma))
        if case['algorithm']=='som':
            x=X[t%len(X)];best=bmu(x,W);eta=schedule(o['learningRate'],o['learningRateEnd'],t,T,o['schedule'])
            W=[[v+eta*h(j,best)*(x[f]-v) for f,v in enumerate(w)] for j,w in enumerate(W)]
        else:
            frozen=[bmu(x,W) for x in X];new=[]
            for j,w in enumerate(W):
                weights=[h(j,b) for b in frozen];den=sum(weights)
                new.append([sum(z*x[f] for z,x in zip(weights,X))/den for f in range(len(w))] if den else w[:])
            W=new
        history.append(copy.deepcopy(W))
    return {'centers':W,'labels':[bmu(x,W) for x in X],'history':history}
base={'data':[[0,0,0],[2,1,1],[0,2,-1],[3,3,2],[1.5,0,.5]],'initialPrototypes':[[0,0,0],[3,0,1],[0,3,-1],[3,3,2]]}
cases=[]
for algorithm in ['som','som_batch']:
    for kind in ['geometric','linear']:
        for end in [.2,0]:
            c={**copy.deepcopy(base),'algorithm':algorithm,'name':f'{algorithm}-{kind}-sigmaend{end}','options':{'gridShape':[2,2],'maxIterations':7 if algorithm=='som' else 3,'sigma':1.2,'sigmaEnd':end,'schedule':kind}}
            if algorithm=='som':c['options'].update(learningRate=.4,learningRateEnd=.1)
            c['expected']=run(c);cases.append(c)
for algorithm in ['som','som_batch']:
    c={'algorithm':algorithm,'name':algorithm+'-zero-radius-empty-and-tie','data':[[1,0],[1,0],[5,2]],'initialPrototypes':[[0,0],[2,0],[5,2],[100,100]],'options':{'gridShape':[2,2],'maxIterations':1,'sigma':0,'sigmaEnd':0,'schedule':'geometric'}}
    if algorithm=='som':c['options'].update(learningRate=.5,learningRateEnd=.5)
    c['expected']=run(c);cases.append(c)
out={'contract':'x-fast width,height lattice; population PCA separate; tie lowest index; sequential samples; frozen batch BMUs; stdlib scalar reference','cases':cases}
path=pathlib.Path(__file__).parent.parent/'fixtures'/'som-shared-reference.json';path.write_text(json.dumps(out,indent=2)+'\n')
print(path)
