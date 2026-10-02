"""Small equal-evaluation smoke only; no concurrent timing claims."""
import json, math, pathlib, statistics, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
(ROOT/'benchmark_replay_reports').mkdir(exist_ok=True)
from ubukit.optimization import optimize,float_range,int_range,categorical
cases={
 'sphere_3d':({'x':float_range(-3,3),'y':float_range(-3,3),'z':float_range(-3,3)},lambda p:(p['x']-.3)**2+(p['y']+.2)**2+(p['z']-.7)**2),
 'rotated_ridge_2d':({'x':float_range(-2,2),'y':float_range(-2,2)},lambda p:50*(p['x']-p['y'])**2+(p['x']+p['y']-.8)**2),
 'branin':({'x':float_range(-5,10),'y':float_range(0,15)},lambda p:(p['y']-5.1/(4*math.pi**2)*p['x']**2+5/math.pi*p['x']-6)**2+10*(1-1/(8*math.pi))*math.cos(p['x'])+10),
 'mixed_log':({'rate':float_range(1e-5,10,log=True),'k':int_range(1,15),'choice':categorical(['a','b','c'])},lambda p:(math.log(p['rate'])-math.log(.03))**2+.1*(p['k']-5)**2+(p['choice']!='b')*2)
}
results={}
for name,(space,objective) in cases.items():
 results[name]={}
 for mode,options in [('random',{'sampler':'random'}),('independent',{'multivariate':False}),('joint',{'multivariate':True})]:
  vals=[optimize(objective,space,50,seed=s,**options).best_value for s in range(5)]
  results[name][mode]={'best_values':vals,'median':statistics.median(vals),'mean':statistics.mean(vals),'n_seeds':5,'budget':50}
print(json.dumps(results,indent=2))
(ROOT/'benchmark_replay_reports'/'quality_smoke.json').write_text(json.dumps({'warning':'Small smoke suite; not SOTA evidence or timing benchmark. All settings are provisional; no held-out tuning claim.','results':results},indent=2)+'\n')
