"""Public TPE driving public EFCM in one installed artifact, against alpha3."""
import unittest
raise unittest.SkipTest("Historical exact-EFCM cleanup qualification is superseded by the float64-only policy; use the focused EFCM tests instead.")
from pathlib import Path
import importlib.util, json, struct, sys, unittest
import numpy as np
import ubukit
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'baseline/python/ubukit'
spec=importlib.util.spec_from_file_location('_combined_alpha3',p/'__init__.py',submodule_search_locations=[str(p)])
baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
class CombinedIntegrationTests(unittest.TestCase):
 def test_tpe_efcm_seeded_search_matches_baseline_exactly(self):
  def run(lib,backend):
   opt=lib.TPEOptimizer({'tau':lib.float_range(.1,3.,log=True),'clusters':lib.int_range(2,3)},seed=42,n_startup_trials=3,n_candidates=5)
   trace=[]
   x=np.array([[-2.,.1],[-1.,1.],[0.,-.5],[1.,.4],[2.,1.]])
   for _ in range(12):
    trial=opt.ask();result=lib.fit_entropy_fcm(x,n_clusters=trial.params['clusters'],tau=trial.params['tau'],random_state=7,max_iter=3,tol=0,backend=backend,return_history=True)
    trace.append((dict(trial.params),result));opt.tell(trial.id,result['objective'])
   return trace,opt.result().to_dict(),opt._rng.state
  def encode(v):
   if isinstance(v,np.ndarray):return ('array',v.dtype.str,v.shape,v.tobytes())
   if isinstance(v,float):return ('float',struct.pack('>d',v))
   if isinstance(v,dict):return [(k,encode(i)) for k,i in v.items()]
   if isinstance(v,(list,tuple)):return [encode(i) for i in v]
   return v
  for backend in ['numpy','reference']:
   with self.subTest(backend=backend):self.assertEqual(encode(run(ubukit,backend)),encode(run(baseline,backend)))
