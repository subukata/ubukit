"""Verify the current Python and JavaScript runtime snapshot without importing it."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'SOURCE_SNAPSHOT.json').read_text())
counts = {'python': 0, 'javascript': 0}
for entry in data['runtime_files']:
    path = root / entry['path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry['sha256']:
        raise SystemExit('Source hash mismatch: ' + entry['path'])
    counts[entry['language']] += 1
assert counts == {'python': 60, 'javascript': 24}, counts
print(json.dumps({'status': 'passed', 'runtime_files': counts}))
