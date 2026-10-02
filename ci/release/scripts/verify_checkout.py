"""Fail closed if a workflow selects anything except the reviewed candidate."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
from verify_source import verify
APPROVED='934b506cd6250e1e9b9ea5ce2a419330f29de487'
sha=os.environ['CANDIDATE_SHA'];assert re.fullmatch(r'[0-9a-f]{40}',sha) and sha==APPROVED,'Unreviewed candidate SHA'
root=Path(sys.argv[1]).resolve()
actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
assert actual==sha,(actual,sha)
print(verify(root/'preview'))

if len(sys.argv)>2:
    bundle=Path(sys.argv[2]).resolve()
    manifest=json.loads((bundle/'build-manifest.json').read_text(encoding='utf-8'))
    assert manifest['source_sha']==sha,'Frozen artifacts target a different source commit'
    expected=manifest['files']
    assert set(expected)=={f.relative_to(bundle).as_posix() for f in bundle.rglob('*') if f.is_file() and f.name!='build-manifest.json'},'Unexpected frozen artifact files'
    for name,digest in expected.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==digest,name
    print('Frozen artifact manifest and source binding passed')
