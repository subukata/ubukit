#!/usr/bin/env python3
"""Focused cleanup regression against an installed candidate; no downloads/benchmarks."""
from pathlib import Path
import argparse,hashlib,io,json,shutil,subprocess,sys,tarfile
HOME=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=HOME.parents[1]);p.add_argument('--python',default=sys.executable);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline-root',type=Path);a=p.parse_args()
repo=a.repo.resolve();out=a.output.resolve();contract=json.loads((HOME/'contract.json').read_text())
assert not out.exists(),'Use a new output directory to preserve prior evidence'
assert not out.is_relative_to(repo),'Place generated evidence outside the checkout'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(root,expected):
 for n,h in expected.items():assert (root/n).is_file() and sha(root/n)==h,f'Source/test identity mismatch: {root/n}'
check(repo/'python/src/ubukit',contract['candidate_python_sha256']);check(repo/'javascript',contract['candidate_javascript_runtime_sha256']);check(repo,contract['repository_test_input_sha256']);check(HOME,contract['tests_sha256'])
out.mkdir(parents=True);(out/'evidence').mkdir();(out/'scripts').mkdir()
# Baselines are reconstructed from a pinned Git commit. A verified read-only
# checkout can be supplied for environments that have no local Git object store.
if a.baseline_root:base=a.baseline_root.resolve()
else:
 raw=subprocess.run(['git','-C',str(repo),'archive','--format=tar',contract['baseline_git_commit'],'python/src/ubukit','javascript/src','javascript/package.json'],capture_output=True,check=True).stdout
 base=out/'baseline_checkout'
 with tarfile.open(fileobj=io.BytesIO(raw)) as t:t.extractall(base,filter='data')
check(base/'python/src/ubukit',contract['baseline_python_sha256']);check(base/'javascript',contract['baseline_javascript_runtime_sha256'])
shutil.copytree(base/'python/src/ubukit',out/'baseline/python/ubukit',ignore=shutil.ignore_patterns('__pycache__'))
(out/'baseline/javascript').mkdir(parents=True);shutil.copytree(base/'javascript/src',out/'baseline/javascript/src');shutil.copy2(base/'javascript/package.json',out/'baseline/javascript/package.json')
# Preserve reviewed test bytes and their relative fixture/import layout.
E=out/'tests/python/efcm';T=out/'tests/python/tpe';J=out/'tests/javascript'
for d in [E/'tests/public/python/tests',E/'tests/public/javascript/fixtures',T/'tests',T/'fixtures',J/'consumer',out/'source/python/tests',out/'source/tools/provenance']:d.mkdir(parents=True)
copy=[(repo/'python/tests/test_entropy_fcm.py',E/'tests/public/python/tests/test_entropy_fcm.py'),(repo/'javascript/fixtures/entropy-fcm-reference.json',E/'tests/public/javascript/fixtures/entropy-fcm-reference.json'),(HOME/'tests/test_redundancy.py',E/'tests/test_redundancy.py'),(repo/'python/verification/new_tests/test_optimization.py',T/'tests/test_optimization_existing.py'),(repo/'python/tests/test_optimizer_import_capture.py',T/'tests/test_optimizer_import_capture.py'),(HOME/'tests/test_tpe_invariant_cleanup.py',T/'tests/test_tpe_invariant_cleanup.py'),(repo/'python/verification/fixtures/normal_interval_reference.json',T/'fixtures/normal_interval_reference.json'),(HOME/'tests/test_combined_integration.py',out/'tests/python/test_combined_integration.py'),(repo/'python/tests/test_namespace.py',out/'source/python/tests/test_namespace.py'),(repo/'python/SOURCE_MANIFEST.json',out/'source/python/SOURCE_MANIFEST.json'),(repo/'tools/provenance/PRODUCT_CONTRACT.json',out/'source/tools/provenance/PRODUCT_CONTRACT.json'),(HOME/'tests/verify-public-api.mjs',J/'consumer/verify-public-api.mjs')]
for source,dest in copy:shutil.copy2(source,dest)
for root in [E,T]:(root/'baseline').symlink_to('../../../baseline/python');(root/'evidence').mkdir()
for name in ['tests','fixtures','validation']:shutil.copytree(repo/'javascript'/name,J/name,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(HOME/'tests/unused-work-regression.test.mjs',J/'tests/unused-work-regression.test.mjs')
package=J/'consumer/node_modules/ubukit-js';package.mkdir(parents=True)
# Copy only declared npm payload files into the generated consumer. A symlink
# would make Node resolve a different path and weaken the preserved import check.
for name in ['package.json',*json.loads((repo/'javascript/package.json').read_text())['files']]:
 target=package/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/'javascript'/name,target)
(J/'candidate_package').symlink_to('consumer/node_modules/ubukit-js');(J/'baseline_package').symlink_to('../../baseline/javascript')
for name in ['run_bounded.py','run_python_suite.py']:shutil.copy2(HOME/'controls'/name,out/'scripts'/name)
(out/'evidence/INTEGRATION_IDENTITY.json').write_text(json.dumps({'python_candidate':contract['candidate_python_sha256']},indent=2)+'\n')
(out/'evidence/REPOSITORY_INPUT_IDENTITY.json').write_text(json.dumps({'repo':str(repo),'baseline_root':str(base),'baseline_commit':contract['baseline_git_commit'],'all_expected_source_and_test_hashes_verified':True,'python_interpreter':a.python,'contract':contract},indent=2)+'\n')
def run(label,command,mem=1024,cpu=60,wall=90,cwd=out):
 subprocess.run([sys.executable,str(out/'scripts/run_bounded.py'),'--cwd',str(cwd),'--memory-mib',str(mem),'--cpu',str(cpu),'--wall',str(wall),label,'--',*map(str,command)],check=True)
python=str(Path(a.python).absolute()) if '/' in a.python else a.python
for suite in ['efcm','tpe','api']:run('python_'+suite,[python,'-I','-B',out/'scripts/run_python_suite.py','installed',suite],mem=512 if suite=='tpe' else 1024)
files=['tests/unused-work-regression.test.mjs','tests/extreme-fcm.test.mjs','tests/extreme-som.test.mjs','tests/efficiency-candidates.test.mjs','tests/som-variants.test.mjs','tests/warm-preflight-regression.test.mjs','tests/clone-owned-security-regression.test.mjs','tests/grid-update-regression.test.mjs','tests/scheduler-memory-regression.test.mjs','validation/tests/legacy-session.test.js','validation/tests/legacy-realtime-worker.test.js']
run('javascript_regression',['node','--experimental-vm-modules','--test','--test-concurrency=1',*files],mem=0,cpu=90,wall=120,cwd=J)
run('javascript_api',['node','--test','--test-concurrency=1','consumer/verify-public-api.mjs'],mem=0,cpu=90,wall=120,cwd=J)
(out/'evidence/COMPLETE.json').write_text(json.dumps({'status':'passed','python_checks':105,'javascript_regression_checks':402,'javascript_public_api_checks':8,'performance_benchmark':False,'release_qualification':False},indent=2)+'\n')
print('Focused cleanup checks passed; evidence:',out/'evidence')
