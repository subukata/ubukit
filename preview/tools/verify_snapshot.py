"""Verify the current runtime and committed verification bytes without importing them."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'SOURCE_SNAPSHOT.json').read_text())
assert data['versions'] == {'python': '0.0.0.dev5', 'javascript': '0.1.0-dev.6'}
counts = {'python': 0, 'javascript': 0}
seen = set()
for entry in data['runtime_files']:
    assert entry['path'] not in seen, entry['path']
    seen.add(entry['path'])
    path = root / entry['path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry['sha256']:
        raise SystemExit('Source hash mismatch: ' + entry['path'])
    counts[entry['language']] += 1
assert counts == {'python': 62, 'javascript': 27}, counts
javascript_paths = {str(path.relative_to(root)) for path in (root / 'javascript/package/src').glob('*.js')}
assert javascript_paths == {entry['path'] for entry in data['runtime_files'] if entry['language'] == 'javascript'}
python_paths = {str(path.relative_to(root)) for path in (root / 'python/staging/src').rglob('*.py') if '__pycache__' not in path.parts}
assert python_paths == {entry['path'] for entry in data['runtime_files'] if entry['language'] == 'python'}

verification = json.loads((root / 'VERIFICATION_SNAPSHOT.json').read_text())
seen = set()
for entry in verification['files']:
    assert entry['path'] not in seen, entry['path']
    seen.add(entry['path'])
    digest = hashlib.sha256((root / entry['path']).read_bytes()).hexdigest()
    if digest != entry['sha256']:
        raise SystemExit('Verification hash mismatch: ' + entry['path'])
print(json.dumps({'status': 'passed', 'runtime_files': counts, 'verification_files': len(seen)}))
