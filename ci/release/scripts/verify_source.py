"""Portable read-only verifier for the namespace-dev6 source and committed test bytes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

def verify(root):
    root=Path(root).resolve()
    data=json.loads((root/'SOURCE_SNAPSHOT.json').read_text(encoding='utf-8'))
    assert data['versions']=={'python':'0.0.0.dev6','javascript':'0.1.0-dev.6'}
    counts={'python':0,'javascript':0}
    for label,entries in [('runtime',data['runtime_files']),('verification',json.loads((root/'VERIFICATION_SNAPSHOT.json').read_text(encoding='utf-8'))['files'])]:
        seen=set()
        for entry in entries:
            name=entry['path'];path=PurePosixPath(name)
            assert name not in seen and not path.is_absolute() and '..' not in path.parts,name
            seen.add(name)
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==entry['sha256'],name
            if label=='runtime':counts[entry['language']]+=1
        if label=='verification':verification_count=len(seen)
    assert counts=={'python':63,'javascript':27},counts
    assert all(e['path'].startswith('python/staging/src/ubukit/') for e in data['runtime_files'] if e['language']=='python'),'Python runtime must be ubukit-only'
    for language,folder,pattern in [('python','python/staging/src','**/*.py'),('javascript','javascript/package/src','*.js')]:
        actual={p.relative_to(root).as_posix() for p in (root/folder).glob(pattern) if '__pycache__' not in p.parts}
        assert actual=={e['path'] for e in data['runtime_files'] if e['language']==language},language
    assert (root/'python/verification/runtime_tests/python_api.py').read_bytes()==(root/'python/verification/python_api.py').read_bytes(),'PR8 harness repair is required'
    return {'status':'passed','versions':data['versions'],'runtime_files':counts,'verification_files':verification_count}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('preview',type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.preview),indent=2))
