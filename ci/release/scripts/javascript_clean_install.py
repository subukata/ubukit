"""Portable npm tarball replay. No runtime registry dependencies expected.
Copies the candidate test tree to a new temporary directory; never reuses consumer node_modules.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--javascript-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--pack-only',action='store_true')
p.add_argument('--frozen-artifacts',type=Path,help='Use one build job tarball and SHA256SUMS.json')
a=p.parse_args()
root,out=a.javascript_root.resolve(),a.output.resolve()
assert not out.exists(), 'Fresh output required'
for relative in ('tests/som-variants.test.mjs', 'tests/efficiency-candidates.test.mjs',
                 'tools/check-package-docs.mjs', 'tests/runtime-compatibility.mjs',
                 'tests/som-browser-compatibility.mjs',
                 'validation/efficiency/differential.mjs', 'validation/efficiency/workers.mjs'):
    assert (root/relative).is_file(), f'Missing dev6 candidate input: {relative}'
out.mkdir(parents=True)
work=out/'candidate'
shutil.copytree(root,work,ignore=shutil.ignore_patterns('node_modules','artifacts','reports','results','.npm-cache'))
artifacts=work/'artifacts'; artifacts.mkdir()
(work/'reports').mkdir(exist_ok=True)
consumer=work/'consumer'; consumer.mkdir(exist_ok=True)
# Clean copied lock file can refer to a prior tarball; do not trust it for this candidate.
for filename in ('package-lock.json','npm-shrinkwrap.json'):
    (consumer/filename).unlink(missing_ok=True)
(consumer/'package.json').write_text(json.dumps({'name':'ubukit-ci-consumer','private':True,'type':'module'}))
steps=[]
def run(label,argv,cwd):
    # npm.cmd is a system launcher on Windows. Arguments here are fixed plus trusted local paths.
    kwargs={'shell':os.name=='nt' and Path(str(argv[0])).name.lower()=='npm.cmd'}
    with (out/(label+'.log')).open('w',encoding='utf-8') as f:
        r=subprocess.run([str(x) for x in argv],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,
                         text=True,env={**os.environ,'npm_config_cache':str(out/'npm-cache')},**kwargs)
    steps.append({'name':label,'exit_code':r.returncode})
    (out/'stages.json').write_text(json.dumps(steps,indent=2))
    if r.returncode:
        print((out/(label+'.log')).read_text(encoding='utf-8',errors='replace'))
        raise SystemExit(r.returncode)
npm=shutil.which('npm.cmd' if os.name=='nt' else 'npm'); assert npm
node=shutil.which('node'); assert node
run('node-version',[node,'--version'],work)
run('npm-version',[npm,'--version'],work)
if a.frozen_artifacts:
    frozen=a.frozen_artifacts.resolve()
    checksums=json.loads((frozen/'SHA256SUMS.json').read_text(encoding='utf-8'))
    tarballs=list(frozen.glob('*.tgz')); assert len(tarballs)==1
    tarball=tarballs[0]
    assert hashlib.sha256(tarball.read_bytes()).hexdigest()==checksums[tarball.name]
    shutil.copy2(tarball,artifacts/tarball.name)
else:
    run('pack',[npm,'pack','--ignore-scripts','--json','--pack-destination',artifacts],work/'package')
packages=list(artifacts.glob('*.tgz')); assert len(packages)==1
run('install',[npm,'install','--offline','--ignore-scripts','--no-audit','--no-fund','--package-lock=false',
               '../artifacts/'+packages[0].name],consumer)
patterns=['tests/*.test.mjs','consumer/tests/*.test.mjs','validation/tests/*.js','validation/tests/*.mjs']
tests=sorted({str(p) for pattern in patterns for p in work.glob(pattern)})
assert tests
if not a.pack_only:
    run('tests',[node,'--test','--test-concurrency=1',*tests],work)
    run('package-docs',[node,'tools/check-package-docs.mjs'],work)
    run('external-vm-worker',[node,'--experimental-vm-modules','tests/runtime-compatibility.mjs'],work)
    run('som-vm-compatibility',[node,'--experimental-vm-modules','tests/som-browser-compatibility.mjs'],work)
    run('efficiency-differential',[node,'validation/efficiency/differential.mjs'],work)
    run('efficiency-workers',[node,'validation/efficiency/workers.mjs'],work)
meta={'status':'passed','tarball':packages[0].name,'sha256':hashlib.sha256(packages[0].read_bytes()).hexdigest(),
      'test_files':len(tests),'actual_browser_execution':False,'package_only':a.pack_only,'consumer':str(consumer)}
(out/'summary.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
