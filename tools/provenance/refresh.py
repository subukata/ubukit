"""Refresh layout snapshots after review. Never approves a candidate or changes runtime hashes."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ci/scripts'))
from verify_source import SOURCE, VERIFICATION, product_snapshot, verification_paths, file_rows, verify


def refresh(root):
    root = Path(root).resolve()
    (root / SOURCE).write_text(json.dumps(product_snapshot(root), indent=2) + '\n', encoding='utf-8')
    snapshot = {'schema_version': 1, 'files': file_rows(root, verification_paths(root))}
    (root / VERIFICATION).write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    return verify(root)


if __name__ == '__main__':
    print(json.dumps(refresh(Path(__file__).resolve().parents[2]), indent=2))
