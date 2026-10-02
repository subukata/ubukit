from pathlib import Path
from importlib import metadata,util
import hashlib,json,sys,sysconfig
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'staging/SOURCE_MANIFEST.json').read_text())
d=metadata.distribution('ubukit-bundled-local-preview'); purelib=Path(sysconfig.get_paths()['purelib']).resolve()
owned={str(p) for p in d.files}
assert d.version == m['version'] == '0.0.0.dev5'
expected={r['path'].removeprefix('src/'):r['sha256'] for r in m['files']}
assert len(expected) == 62
for n,h in expected.items():
 assert n in owned,n
 p=Path(d.locate_file(n)).resolve(); assert p.is_relative_to(purelib),p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
for n in ['ubukit','portable_accel','ubukit_fcm','ubukit_rmcm','rough_cmeans','_numba_kernel','external_metrics','_external_metrics_numba']:
 p=Path(util.find_spec(n).origin).resolve(); assert p.is_relative_to(purelib),p
 assert str(p.relative_to(purelib)) in owned,n
numba=util.find_spec('numba') is not None
assert numba==('numba' in sys.argv[1]),numba
print(json.dumps({'status':'passed','source_hashes_verified':len(expected),'all_import_roots_owned':True,'numba':numba,'python':sys.version,'platform':sys.platform,'dependencies':{x.metadata['Name']:x.version for x in metadata.distributions()}},indent=2))
