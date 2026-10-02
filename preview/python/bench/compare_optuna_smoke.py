"""Run the existing cheap smoke suite against explicitly recorded Optuna variants."""
import importlib.util,json,pathlib,statistics,sys,time,warnings
ROOT=pathlib.Path(__file__).resolve().parents[1]
(ROOT/'benchmark_replay_reports').mkdir(exist_ok=True)
# Import problem definitions without executing the smoke runner.
source=(ROOT/'bench'/'smoke_quality.py').read_text()
namespace={"__file__":str(ROOT/'bench'/'smoke_quality.py')}
exec(source.split('results={}')[0],namespace)
cases=namespace['cases']
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings('ignore',category=optuna.exceptions.ExperimentalWarning)
base=json.loads((ROOT/'benchmark_replay_reports'/'quality_smoke.json').read_text())
report={'optuna_version':optuna.__version__,'python_version':sys.version.split()[0],
 'warning':'Exploratory quality smoke: 4 handpicked cheap tasks, 5 seeds, 50 evaluations. Not held-out, not evidence of universal superiority. Both libraries use native RNG/startup defaults; same integer seed does not match initial points. No parallel search. Timing includes framework overhead and concurrent CPU contention; not a publishable performance measurement.',
 'budget':50,'seeds':list(range(5)),
 'variants':{'ubukit_joint':{'n_startup_trials':12,'n_candidates':24,'gamma':.15,'multivariate':True,'weights':'ei','min_bandwidth':.03,'avoid_duplicates':True},
 'ubukit_independent':{'same_as':'ubukit_joint','multivariate':False},'ubukit_random':{'sampler':'random','avoid_duplicates':True},
 'optuna5_default':{'TPESampler':'seed only, all other Optuna5.0 defaults'},'optuna5_independent':{'TPESampler':'seed plus multivariate=False, other defaults'}},'results':{}}
for name,(space,objective) in cases.items():
 rows={}
 for label,key in [('ubukit_joint','joint'),('ubukit_independent','independent'),('ubukit_random','random')]:rows[label]=base['results'][name][key]
 for label,extra in [('optuna5_default',{}),('optuna5_independent',{'multivariate':False})]:
  vals=[];times=[]
  for seed in range(5):
   sampler=optuna.samplers.TPESampler(seed=seed,**extra)
   study=optuna.create_study(sampler=sampler,direction='minimize')
   def evaluate(trial):
    params={}
    for k,spec in space.items():
     if spec['type']=='float':params[k]=trial.suggest_float(k,spec['low'],spec['high'],log=spec.get('log',False))
     elif spec['type']=='int':params[k]=trial.suggest_int(k,spec['low'],spec['high'],log=spec.get('log',False))
     else:params[k]=trial.suggest_categorical(k,spec['choices'])
    return objective(params)
   start=time.perf_counter();study.optimize(evaluate,n_trials=50,show_progress_bar=False);times.append(time.perf_counter()-start)
   vals.append(study.best_value)
  rows[label]={'best_values':vals,'median':statistics.median(vals),'mean':statistics.mean(vals),'n_seeds':5,'budget':50,'exploratory_total_seconds':times}
  print(name,label,rows[label]['median'],flush=True)
 report['results'][name]=rows
(ROOT/'benchmark_replay_reports'/'optuna5_quality_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print('saved',ROOT/'benchmark_replay_reports'/'optuna5_quality_smoke.json')
