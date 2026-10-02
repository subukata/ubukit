"""Cross-language fixed fixture integrity; no source imports or timing work."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_shared_fixture_contracts():
    for contract in json.loads((ROOT / 'tests/contracts/shared-fixtures.json').read_text())['contracts']:
        for name in contract['consumers']:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == contract['sha256'], name
