"""Verify private dev5 identity and source/wheel/sdist/consumer bindings."""
import hashlib,importlib.metadata as md,json,pathlib,sys,sysconfig,tarfile,zipfile
import ubukit
ROOT=pathlib.Path(__file__).resolve().parents[1]
STAGE=ROOT/'staging';PURE=pathlib.Path(sysconfig.get_paths()['purelib']).resolve()
manifest=json.loads((STAGE/'SOURCE_MANIFEST.json').read_text())
assert ubukit.__version__==md.version('ubukit-bundled-local-preview')==manifest['version']=='0.0.0.dev5'
assert pathlib.Path(ubukit.__file__).resolve().is_relative_to(PURE)
assert not any(x in sys.modules for x in ('numpy','scipy','sklearn','numba'))
assert len(ubukit.__all__)==38
assert 'version = "0.0.0.dev5"' in (STAGE/'pyproject.toml').read_text()
assert len(manifest['files'])==62
wheel=ROOT/'artifacts/ubukit_bundled_local_preview-0.0.0.dev5-py3-none-any.whl'
sdist=ROOT/'artifacts/ubukit_bundled_local_preview-0.0.0.dev5.tar.gz'
checked=[]
with zipfile.ZipFile(wheel) as wh,tarfile.open(sdist) as sd:
 metadata=next(n for n in wh.namelist() if n.endswith('.dist-info/METADATA'))
 assert '\nVersion: 0.0.0.dev5\n' in wh.read(metadata).decode()
 prefix=sd.getnames()[0].split('/')[0]
 assert '\nVersion: 0.0.0.dev5\n' in sd.extractfile(prefix+'/PKG-INFO').read().decode()
 for row in manifest['files']:
  name=row['path'].removeprefix('src/')
  streams={'source':(STAGE/row['path']).read_bytes(),'wheel':wh.read(name),'sdist':sd.extractfile(prefix+'/'+row['path']).read(),'consumer':(PURE/name).read_bytes()}
  for label,data in streams.items():assert hashlib.sha256(data).hexdigest()==row['sha256'],(label,name)
  if row['change']=='version_only':
   assert name=='ubukit/__init__.py'
   normalized=streams['source'].replace(b'__version__ = "0.0.0.dev5"',b'__version__ = "0.0.0.dev4"')
   assert hashlib.sha256(normalized).hexdigest()==row['baseline_sha256']
  elif row['change']=='unchanged':assert row['sha256']==row['baseline_sha256']
  checked.append(name)
assert sum(r['change']=='unchanged' for r in manifest['files'])==60
assert sum(r['change']=='version_only' for r in manifest['files'])==1
assert next(r['sha256'] for r in manifest['files'] if r['path'].endswith('/som_olp_localized.py'))=='115887d87e6523cee1f4663cfd6c46cdb672a0a76028de4bd6b71f1da825785a'
print(json.dumps(dict(status='passed',package_version=ubukit.__version__,runtime_files_checked=len(checked),byte_identical_baseline_modules=60,version_only_facade_changes=1,new_modules=1,numerical_candidate_unchanged=True,facade_exports=38,artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel,sdist)}),indent=2))
