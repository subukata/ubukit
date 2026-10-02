"""Read-only, manifest-driven verification of product and harness bytes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tomllib

CONTRACT = 'tools/provenance/PRODUCT_CONTRACT.json'
SOURCE = 'tools/provenance/SOURCE_SNAPSHOT.json'
VERIFICATION = 'tools/provenance/VERIFICATION_SNAPSHOT.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root, name):
    require(isinstance(name, str) and bool(name) and '\\' not in name, f'Invalid path: {name!r}')
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(p not in ('..', '.', '') for p in name.split('/')),
            f'Unsafe path: {name}')
    target = Path(root) / name
    require(target.resolve().is_relative_to(Path(root).resolve()), f'Path escapes root: {name}')
    require(not any(p.is_symlink() for p in [target, *target.parents] if p != Path(root).parent),
            f'Symlink not permitted: {name}')
    return target


def read_json(root, name):
    return json.loads(safe_path(root, name).read_text(encoding='utf-8'))


def file_rows(root, names):
    return [{'path': name, 'sha256': digest(safe_path(root, name))} for name in sorted(set(names))]


def product_snapshot(root):
    """Derive current layout from explicit contracts; do not update baseline hashes."""
    root = Path(root).resolve()
    contract = read_json(root, CONTRACT)
    require(contract['schema_version'] == 1, 'Unsupported product contract')
    runtime, package_names, versions = [], set(), {}
    for language, spec in contract['languages'].items():
        package_root = safe_path(root, spec['root'])
        manifest = read_json(root, spec['source_manifest'])
        versions[language] = manifest['version']
        if language == 'python':
            metadata = tomllib.loads(safe_path(root, spec['metadata']).read_text(encoding='utf-8'))['project']
            require(metadata['name'] == spec['distribution'], 'Python distribution mismatch')
            entries = {e['path']: e['sha256'] for e in manifest['files']}
            require(len(entries) == len(manifest['files']), 'Duplicate Python runtime path')
            require(all(n.startswith('src/' + spec['namespace'] + '/') for n in entries), 'Python runtime namespace mismatch')
            package_files = spec['package_metadata']
            manifest_in = safe_path(package_root, 'MANIFEST.in')
            if manifest_in.is_file():
                included = {'pyproject.toml', 'MANIFEST.in'}
                for line in manifest_in.read_text(encoding='utf-8').splitlines():
                    words = line.split()
                    if words and words[0] == 'include':
                        require(not any('*' in name or '?' in name for name in words[1:]), 'Package metadata includes must be explicit files')
                        included.update(words[1:])
                require(set(package_files) == included, 'Python package metadata allowlist differs from MANIFEST.in')
        elif language == 'javascript':
            metadata = read_json(root, spec['metadata'])
            entries = {n: e['sha256'] for n, e in manifest['runtime_files'].items()}
            require(set(metadata['exports']) == set(spec['entrypoints']), 'JavaScript entrypoint mismatch')
            package_files = ['package.json', *metadata['files']]
            require(set(entries) == {n for n in metadata['files'] if n.startswith('src/')}, 'JavaScript package runtime allowlist mismatch')
        else:
            raise ValueError(f'Unknown language: {language}')
        require(metadata['version'] == versions[language], f'{language} package version mismatch')
        declared = {f"{spec['root']}/{name}" for name in entries}
        actual = {p.relative_to(root).as_posix() for p in safe_path(root, spec['runtime_root']).glob(spec['runtime_glob']) if p.is_file() and '__pycache__' not in p.parts}
        require(actual == declared, f'{language} runtime path set mismatch')
        for name, expected in entries.items():
            full = f"{spec['root']}/{name}"
            require(re.fullmatch('[0-9a-f]{64}', expected) is not None, f'Invalid digest: {full}')
            require(digest(safe_path(root, full)) == expected, f'Runtime hash mismatch: {full}')
            runtime.append({'language': language, 'path': full, 'sha256': expected})
        package_names.update(declared)
        package_names.update(f"{spec['root']}/{name}" for name in package_files)
    package_names.add(CONTRACT)
    return {'schema_version': 1, 'versions': versions,
            'runtime_files': sorted(runtime, key=lambda e: e['path']),
            'package_files': file_rows(root, package_names)}


def verification_paths(root):
    root = Path(root).resolve()
    spec = read_json(root, CONTRACT)['verification']
    names = set()
    for pattern in spec['include']:
        for path in root.glob(pattern):
            if path.is_file():
                relative = path.relative_to(root)
                if not any(part in spec['exclude_parts'] or part.endswith('.egg-info') for part in relative.parts) and relative.as_posix() not in spec['exclude_files']:
                    names.add(relative.as_posix())
    return names


def verify(root):
    root = Path(root).resolve()
    expected = read_json(root, SOURCE)
    actual = product_snapshot(root)
    require(actual == expected, 'Product snapshot differs from reviewed runtime/package bytes')
    verification = read_json(root, VERIFICATION)
    rows = file_rows(root, verification_paths(root))
    require(verification.get('schema_version') == 1 and verification['files'] == rows,
            'Verification snapshot differs from complete harness input set')
    first, second = read_json(root, CONTRACT)['facade_parity']
    require(safe_path(root, first).read_bytes() == safe_path(root, second).read_bytes(), 'Facade harness parity repair is required')
    counts = {language: sum(e['language'] == language for e in actual['runtime_files']) for language in actual['versions']}
    return {'status': 'passed', 'versions': actual['versions'], 'runtime_files': counts,
            'package_files': len(actual['package_files']), 'verification_files': len(rows),
            'source_snapshot_sha256': digest(root / SOURCE)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(verify(parser.parse_args().root), indent=2))
