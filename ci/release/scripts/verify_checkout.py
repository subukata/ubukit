"""Fail closed if a workflow selects anything except the reviewed candidate."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
from verify_source import verify
# The reviewed source commit precedes this CI-only change; GITHUB_SHA binds the harness.
APPROVED='86321a9f1bc405ddb3251facfe1d876064c9c9dd'
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

if len(sys.argv)>3:
    verification=Path(sys.argv[3]).resolve()
    harness_sha=os.environ['GITHUB_SHA']
    assert subprocess.check_output(['git','-C',str(verification),'rev-parse','HEAD'],text=True).strip()==harness_sha
    verify(verification/'preview')
    assert (verification/'preview/SOURCE_SNAPSHOT.json').read_bytes()==(root/'preview/SOURCE_SNAPSHOT.json').read_bytes(), 'Verification checkout changes reviewed runtime bytes'
    print('Workflow-bound verification snapshot and identical runtime manifest passed')
