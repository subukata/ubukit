"""Read-only archive inventory and source-byte checks; no extraction or installation."""
import argparse
from email.parser import BytesParser
import json
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile
from verify_source import CONTRACT, product_snapshot, read_json, require, safe_path


def tar_members(path):
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        require(all(m.isfile() or m.isdir() for m in members), f'Unsupported archive member type: {path}')
        names = [m.name for m in members]
        require(len(names) == len(set(names)), f'Duplicate archive member: {path}')
        for name in names:
            p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in name, f'Unsafe archive member: {name}')
        return {m.name: archive.extractfile(m).read() for m in members if m.isfile()}


def one(folder, pattern):
    matches = list(folder.glob(pattern))
    require(len(matches) == 1, f'Expected exactly one {folder}/{pattern}')
    return matches[0]


def verify_artifacts(source, artifacts):
    source, artifacts = Path(source).resolve(), Path(artifacts).resolve()
    snapshot = product_snapshot(source)
    contract = read_json(source, CONTRACT)
    py = contract['languages']['python']
    project = tomllib.loads((source / py['metadata']).read_text())['project']
    normalized = py['distribution'].replace('-', '_')
    version = snapshot['versions']['python']
    pyroot = source / py['root']
    runtime = {e['path'][len(py['root']) + 1:] for e in snapshot['runtime_files'] if e['language'] == 'python'}
    sdist = tar_members(one(artifacts / 'python', '*.tar.gz'))
    prefix = f'{normalized}-{version}/'
    require(all(n.startswith(prefix) for n in sdist), 'Unexpected sdist root')
    sdist = {n.removeprefix(prefix): b for n, b in sdist.items()}
    egg = f'src/{normalized}.egg-info/'
    expected_sdist = runtime | set(py['package_metadata']) | set(py['sdist_generated_files']) | {egg + n for n in py['egg_info_files']}
    require(set(sdist) == expected_sdist, f'Sdist inventory differs: {sorted(set(sdist) ^ expected_sdist)}')
    for name in runtime | set(py['package_metadata']):
        require(sdist[name] == safe_path(pyroot, name).read_bytes(), f'Sdist source-byte mismatch: {name}')
    require(BytesParser().parsebytes(sdist['PKG-INFO'])['Version'] == version, 'Sdist version mismatch')
    wheel_path = one(artifacts / 'python', '*.whl')
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate wheel member')
        wheel = {n: archive.read(n) for n in names}
    info = f'{normalized}-{version}.dist-info/'
    licenses = {info + 'licenses/' + n for n in project['license-files']}
    expected_wheel = {n.removeprefix('src/') for n in runtime} | {info + n for n in py['wheel_metadata_files']} | licenses
    require(set(wheel) == expected_wheel, f'Wheel inventory differs: {sorted(set(wheel) ^ expected_wheel)}')
    for name in runtime:
        require(wheel[name.removeprefix('src/')] == safe_path(pyroot, name).read_bytes(), f'Wheel source-byte mismatch: {name}')
    for name in project['license-files']:
        require(wheel[info + 'licenses/' + name] == safe_path(pyroot, name).read_bytes(), f'Wheel notice mismatch: {name}')
    require(BytesParser().parsebytes(wheel[info + 'METADATA'])['Version'] == version, 'Wheel version mismatch')
    js = contract['languages']['javascript']
    metadata = read_json(source, js['metadata'])
    npm = tar_members(one(artifacts / 'javascript', '*.tgz'))
    allowlist = {'package/' + n for n in ['package.json', *metadata['files']]}
    require(set(npm) == allowlist, f'npm inventory differs: {sorted(set(npm) ^ allowlist)}')
    for name, data in npm.items():
        require(data == safe_path(source / js['root'], name.removeprefix('package/')).read_bytes(), f'npm source-byte mismatch: {name}')
    return {'status': 'passed', 'wheel_files': len(wheel), 'sdist_files': len(sdist), 'npm_files': len(npm)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_artifacts(args.source, args.artifacts), indent=2))
