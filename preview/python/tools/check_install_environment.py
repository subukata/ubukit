"""Read-only preflight for the local bundle; exit 2 if imports/owners conflict.

This is deliberately not an install hook. pip has no reliable standard metadata
field to make independently named distributions mutually exclusive. Run with
the TARGET environment's Python before installation. Never removes packages.
"""
import argparse
from importlib import metadata, util
import json
from pathlib import Path
import re
import sys

OWN = 'ubukit-bundled-local-preview'
LEGACY_ROOTS = {'portable_accel', 'ubukit_fcm', 'ubukit_rmcm', 'rough_cmeans',
                '_numba_kernel', 'external_metrics', '_external_metrics_numba'}
TOP = {'ubukit'}

def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()

def inspect_environment():
    ownership = {}
    blocked_distributions = []
    for dist in metadata.distributions():
        name = canonical(dist.metadata.get('Name', ''))
        for f in dist.files or ():
            p = Path(f)
            top = p.parts[0] if p.parts else ''
            key = top[:-3] if top.endswith('.py') else top
            if name == OWN and key in LEGACY_ROOTS:
                item = {'name': name, 'version': dist.version}
                if item not in blocked_distributions:
                    blocked_distributions.append(item)
            if key in TOP:
                resolved = str(Path(dist.locate_file(f)).resolve())
                ownership.setdefault(resolved, set()).add(name)
    conflicts = []
    for path, owners in sorted(ownership.items()):
        if owners - {OWN}:
            conflicts.append({'kind': 'foreign_file_owner', 'path': path,
                              'owners': sorted(owners)})
    modules = {}
    for name in sorted(TOP):
        spec = util.find_spec(name)
        if spec is None:
            modules[name] = {'present': False}
            continue
        origins = list(spec.submodule_search_locations or [])
        if spec.origin and spec.origin not in ('built-in', 'frozen'):
            path = str(Path(spec.origin).resolve())
            owners = ownership.get(path, set())
            modules[name] = {'present': True, 'origin': path, 'owners': sorted(owners)}
            if owners != {OWN}:
                conflicts.append({'kind': 'existing_import_path', 'module': name,
                                  'origin': path, 'owners': sorted(owners)})
        else:
            modules[name] = {'present': True, 'origin': spec.origin, 'locations': origins}
            conflicts.append({'kind': 'existing_namespace_or_builtin', 'module': name,
                              'origin': spec.origin, 'locations': origins})
    ok = not conflicts and not blocked_distributions
    return {'status': 'clear' if ok else 'blocked', 'python': sys.executable,
            'read_only': True, 'legacy_distributions': blocked_distributions,
            'conflicts': conflicts, 'modules': modules,
            'advice': ('Use a new environment; an old bundled layout or ubukit owner needs manual review. '
                       'never uninstall one overlapping distribution after installing the bundle.') if not ok else
                      'No current ownership/import collision found. Recheck if environment contents change.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = inspect_environment()
    rendered = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')
    raise SystemExit(0 if result['status'] == 'clear' else 2)
