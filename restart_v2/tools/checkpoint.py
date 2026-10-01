"""Create an aggregate source checkpoint from explicitly ready directories."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import argparse,hashlib,json,datetime,ast
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--include',action='append',default=[]);p.add_argument('--preserve-prefix',action='append',default=[]);a=p.parse_args()
r=Path(__file__).resolve().parents[1];files=[]
for node in ast.parse((r/'portable_accel/__init__.py').read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_exports' for t in node.targets):
  for module in ast.literal_eval(node.value).values():
   assert (r/'portable_accel'/(module+'.py')).is_file(), 'Missing public module: '+module
for folder in ['portable_accel','tests','evidence','tools',*a.include]:
 if folder == 'som_candidate' and (r/folder/'SOURCE_MANIFEST.json').exists():
  for manifest_name in ('SOURCE_MANIFEST.json','PREPARED_SVD_MANIFEST.json'):
   source_manifest=r/folder/manifest_name
   if not source_manifest.exists():continue
   files.append(source_manifest)
   for name,want in json.loads(source_manifest.read_text())['files'].items():
    f=r/folder/name
    assert hashlib.sha256(f.read_bytes()).hexdigest()==want, 'Frozen SOM source changed: '+name
    files.append(f)
  continue
 for f in (r/folder).rglob('*'):
  if f.is_file() and not any(x in f.parts for x in ('__pycache__','.venv','build','.git','runtime','data')) and f.suffix not in ('.pyc','.nbc','.nbi','.so','.o','.npz','.npy'):
   files.append(f)
for f in r.iterdir():
 if f.is_file() and f.suffix in ('.md','.toml','.txt'):files.append(f)
files=sorted(set(files));manifest={'stage':a.stage,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_source_not_old_recovery':True,'files':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
out=r.parent/'restart_v2_checkpoint.zip'
preserved={}
if out.exists() and a.preserve_prefix:
 with ZipFile(out) as old:
  for name in old.namelist():
   relative=name.removeprefix('restart_v2/')
   if any(relative.startswith(prefix.rstrip('/')+'/') for prefix in a.preserve_prefix):
    preserved[relative]=old.read(name)
 for name,data in preserved.items():manifest['files'].setdefault(name,hashlib.sha256(data).hexdigest())
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for name,data in preserved.items():
  if name not in {str(f.relative_to(r)) for f in files}:z.writestr('restart_v2/'+name,data)
 for f in files:z.write(f,'restart_v2/'+str(f.relative_to(r)))
 z.writestr('restart_v2/CHECKPOINT_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
with ZipFile(out) as z:
 assert z.testzip() is None
 for name,h in manifest['files'].items():assert hashlib.sha256(z.read('restart_v2/'+name)).hexdigest()==h
print(json.dumps({'path':str(out),'stage':a.stage,'files':len(manifest['files'])+1,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
