"""Build wheel, sdist and npm tarball once; write checksums and bound storage."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from verify_source import verify

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();source=a.source.resolve();out=a.output.resolve();assert not out.exists()
verify(source/'preview');out.mkdir(parents=True)
py=out/'python';js=out/'javascript';py.mkdir();js.mkdir()
subprocess.run([sys.executable,'-m','build','--no-isolation','--wheel','--sdist','--outdir',str(py),str(source/'preview/python/staging')],check=True)
npm=shutil.which('npm');assert npm
subprocess.run([npm,'--cache',str(out.parent/'candidate-npm-cache'),'pack','--ignore-scripts','--json','--pack-destination',str(js)],cwd=source/'preview/javascript/package',check=True)
expected={py:['*.whl','*.tar.gz'],js:['*.tgz']}
for folder,patterns in expected.items():
    files=[]
    for pattern in patterns:
        matches=list(folder.glob(pattern));assert len(matches)==1,(pattern,matches);files+=matches
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (folder/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
manifest={'source_sha':os.environ.get('CANDIDATE_SHA'),'harness_sha':os.environ.get('GITHUB_SHA'),
          'python':sys.version,'files':{f.relative_to(out).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}}
(out/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
size=sum(f.stat().st_size for f in out.rglob('*') if f.is_file());assert size<=4*1024*1024,size
print(json.dumps({'status':'passed','frozen_artifact_bytes':size,**manifest},indent=2))
