"""Verify then restore the exact private pre-layout main archive to a new directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
manifest=json.loads(a.manifest.read_text());expected={e['path']:e for e in manifest['files']}
assert len(expected)==len(manifest['files']) and expected, 'Duplicate or empty manifest'
assert not a.output.exists(), 'Output must be a new directory'
with tarfile.open(a.archive) as tf:
    members=tf.getmembers();assert {m.name for m in members}==set(expected) and len(members)==len(expected)
    for m in members:
        path=PurePosixPath(m.name);assert m.isfile() and not path.is_absolute() and '..' not in path.parts
        data=tf.extractfile(m).read();e=expected[m.name]
        assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],m.name
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==e['git_blob_sha1'],m.name
        assert m.mode & 0o777 == int(e['mode'],8) & 0o777,m.name
    a.output.mkdir(parents=True);tf.extractall(a.output,filter='data')
for name,e in expected.items():
    path=a.output/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256']
    assert path.stat().st_mode & 0o777 == int(e['mode'],8) & 0o777
print(json.dumps({'status':'passed','source_commit':manifest['source_commit'],'restored_files':len(expected)}))
