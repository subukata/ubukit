#!/usr/bin/env python3
"""Check the deliberate legacy/active metric source pair without importing it.

The two import locations deliberately own independent Quality classes, globals,
and Numba dispatchers. Keep numerical source in lockstep while that direct-import
compatibility contract remains supported. This check is not a numerical test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PAIRED_PATHS = (
    Path('ubukit/_impl/portable_accel/_backends/metrics_numba.py'),
    Path('ubukit/_impl/portable_accel/_backends/metrics_portable/_numba_core.py'),
)


def check_pair(source_root: Path) -> dict:
    payloads = [(source_root / path).read_bytes() for path in PAIRED_PATHS]
    if payloads[0] != payloads[1]:
        raise ValueError('Intentional legacy/active metric sources diverged; review '
                         'and update both implementations together or explicitly '
                         'approve a compatibility migration.')
    return {
        'status': 'identical',
        'bytes_per_copy': len(payloads[0]),
        'sha256': hashlib.sha256(payloads[0]).hexdigest(),
        'paths': [path.as_posix() for path in PAIRED_PATHS],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'src')
    args = parser.parse_args()
    try:
        result = check_pair(args.source_root)
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
