from pathlib import Path
from importlib import metadata,util
import hashlib,json,sys,sysconfig
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'staging/SOURCE_MANIFEST.json').read_text())
d=metadata.distribution('ubukit-bundled-local-preview');purelib=Path(sysconfig.get_paths()['purelib']).resolve();owned={str(p) for p in d.files}
assert d.version==m['version']=='0.0.0.dev6'
expected={r['path'].removeprefix('src/'):r['sha256'] for r in m['files']};assert len(expected)==63
for n,h in expected.items():
 assert n in owned and n.startswith('ubukit/'),n
 p=Path(d.locate_file(n)).resolve();assert p.is_relative_to(purelib),p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
assert all(n.startswith('ubukit/') or '.dist-info/' in n for n in owned),owned
for name in ('portable_accel','ubukit_fcm','ubukit_rmcm','rough_cmeans','_numba_kernel','external_metrics','_external_metrics_numba'):
 assert util.find_spec(name) is None,(name,'old top-level module visible in isolated consumer')
numba=util.find_spec('numba') is not None;assert numba==('numba' in sys.argv[1:]),numba
print(json.dumps({'status':'passed','source_hashes_verified':len(expected),'all_import_roots_owned':True,'sole_runtime_root':'ubukit','old_top_level_modules_absent':True,'numba':numba,'python':sys.version,'platform':sys.platform,'dependencies':{x.metadata['Name']:x.version for x in metadata.distributions()}},indent=2))
