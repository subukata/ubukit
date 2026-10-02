from pathlib import Path
from importlib import metadata
import hashlib,json,sys
R=Path(__file__).resolve().parents[1]
d=metadata.distribution('ubukit-bundled-local-preview'); info=json.loads(d.read_text('direct_url.json'))
archive=next((R/'artifacts').glob('*.tar.gz' if sys.argv[1]=='base-sdist' else '*.whl'))
expected=hashlib.sha256(archive.read_bytes()).hexdigest()
actual=info['archive_info'].get('hashes',{}).get('sha256') or info['archive_info']['hash'].removeprefix('sha256=')
assert actual==expected,(actual,expected)
assert d.version==json.loads((R/'SOURCE_MANIFEST.json').read_text())['version']
print(json.dumps({'status':'passed','environment':sys.argv[1],'installed_version':d.version,'artifact':archive.name,'installed_source_archive_sha256':actual},indent=2))
