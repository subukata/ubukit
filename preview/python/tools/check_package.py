"""Bind private dev6 source, wheel, sdist and installed runtime; enforce one root."""
import hashlib,importlib.metadata as md,json,pathlib,sys,sysconfig,tarfile,zipfile
import ubukit
ROOT=pathlib.Path(__file__).resolve().parents[1]
STAGE=ROOT/'staging';PURE=pathlib.Path(sysconfig.get_paths()['purelib']).resolve()
manifest=json.loads((STAGE/'SOURCE_MANIFEST.json').read_text())
assert ubukit.__version__==md.version('ubukit-bundled-local-preview')==manifest['version']=='0.0.0.dev6'
assert pathlib.Path(ubukit.__file__).resolve().is_relative_to(PURE)
assert not any(x in sys.modules for x in ('numpy','scipy','sklearn','numba'))
assert len(ubukit.__all__)==38
assert len(manifest['files'])==63
wheel=ROOT/'artifacts/ubukit_bundled_local_preview-0.0.0.dev6-py3-none-any.whl'
sdist=ROOT/'artifacts/ubukit_bundled_local_preview-0.0.0.dev6.tar.gz'
checked=[]
with zipfile.ZipFile(wheel) as wh,tarfile.open(sdist) as sd:
 metadata=next(n for n in wh.namelist() if n.endswith('.dist-info/METADATA'))
 assert '\nVersion: 0.0.0.dev6\n' in wh.read(metadata).decode()
 prefix=sd.getnames()[0].split('/')[0]
 assert '\nVersion: 0.0.0.dev6\n' in sd.extractfile(prefix+'/PKG-INFO').read().decode()
 expected={r['path'][4:] for r in manifest['files']}
 assert {n for n in wh.namelist() if n.endswith('.py')}==expected
 assert {n.removeprefix(prefix+'/src/') for n in sd.getnames() if n.startswith(prefix+'/src/') and n.endswith('.py')}==expected
 assert all(n.startswith('ubukit/') or '.dist-info/' in n for n in wh.namelist())
 assert not any('__pycache__' in n or n.endswith(('.pyc','.pyo','.nbc','.nbi')) for n in wh.namelist()+sd.getnames())
 for row in manifest['files']:
  name=row['path'].removeprefix('src/')
  streams={'source':(STAGE/row['path']).read_bytes(),'wheel':wh.read(name),'sdist':sd.extractfile(prefix+'/'+row['path']).read(),'consumer':(PURE/name).read_bytes()}
  for label,data in streams.items():assert hashlib.sha256(data).hexdigest()==row['sha256'],(label,name)
  if row['change']=='unchanged_bytes':assert row['sha256']==row['baseline_sha256']
  checked.append(name)
assert sum(r['change']=='unchanged_bytes' for r in manifest['files'])==58
print(json.dumps(dict(status='passed',package_version=ubukit.__version__,runtime_files_checked=len(checked),byte_identical_inherited_files=58,namespace_modified_files=4,new_private_package_markers=1,facade_exports=38,only_runtime_root='ubukit',artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel,sdist)}),indent=2))
