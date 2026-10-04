"""Run preserved regression tests against installed artifacts, with path proof."""
from pathlib import Path
import argparse, hashlib, importlib, importlib.metadata, importlib.util, json, os, platform, sys, unittest
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('consumer');p.add_argument('suite',choices=['efcm','tpe','api']);a=p.parse_args()
import ubukit
actual=Path(ubukit.__file__).resolve().parent
assert 'site-packages' in actual.parts, 'Use a clean installed candidate interpreter, not a source-path import'
identity=json.loads((ROOT/'evidence/INTEGRATION_IDENTITY.json').read_text())
hashes={str(f.relative_to(actual)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(actual.rglob('*.py'))}
assert hashes==identity['python_candidate'],'installed source mismatch'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
suite=unittest.TestSuite();modules=[]
if a.suite in ['efcm','tpe']:
 d=ROOT/'tests/python'/a.suite;link=d/'candidate'
 if link.is_symlink():link.unlink()
 assert not link.exists();link.symlink_to(actual.parent)
 names=['tests/public/python/tests/test_entropy_fcm.py','tests/test_redundancy.py'] if a.suite=='efcm' else ['tests/test_optimization_existing.py','tests/test_optimizer_import_capture.py','tests/test_tpe_invariant_cleanup.py']
 for i,f in enumerate(names):
  m=load(a.suite+str(i),d/f);modules.append(m);suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))
else:
 m=load('namespace_contract',ROOT/'source/python/tests/test_namespace.py');modules.append(m)
 for n in sorted(vars(m)):
  if n.startswith('test_'):suite.addTest(unittest.FunctionTestCase(getattr(m,n)))
 m=load('combined_integration',ROOT/'tests/python/test_combined_integration.py');modules.append(m);suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))
class Result(unittest.TextTestResult):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.ids=[];self.subtests=0
 def startTest(self,test):self.ids.append(test.id());super().startTest(test)
 def addSubTest(self,test,subtest,err):self.subtests+=1;super().addSubTest(test,subtest,err)
r=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
record={'consumer':a.consumer,'suite':a.suite,'passed':r.wasSuccessful(),'tests_run':r.testsRun,'subtests':r.subtests,'test_ids':r.ids,'failures':[(str(t),e) for t,e in r.failures],'errors':[(str(t),e) for t,e in r.errors],'skips':r.skipped,'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'runtime_import_path':str(actual),'installed_source_hashes':hashes,'numba_installed':importlib.util.find_spec('numba') is not None,'numba_imported':any(n.split('.')[0]=='numba' for n in sys.modules),'distribution_versions':{n:importlib.metadata.version(n) for n in ['ubukit','numpy','scipy','scikit-learn','threadpoolctl','joblib']},'focused_check_only':True,'performance_benchmark':False}
for m in modules:
 if hasattr(m,'EVIDENCE'):record[m.__name__+'_evidence']=m.EVIDENCE
(ROOT/'evidence'/f'{a.consumer}_{a.suite}_results.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:record[k] for k in ['consumer','suite','passed','tests_run','subtests','runtime_import_path','numba_installed']}));sys.exit(0 if r.wasSuccessful() else 1)
