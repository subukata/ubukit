"""Build once from a verified candidate; freeze artifact and independent harness provenance."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from verify_source import digest, require
from verify_checkout import verify_checkout
from verify_artifact_contents import verify_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    require(not out.exists(), 'Fresh output directory required')
    context = verify_checkout(source)
    out.mkdir(parents=True)
    py, js = out / 'python', out / 'javascript'
    py.mkdir(); js.mkdir()
    subprocess.run([sys.executable, '-m', 'build', '--no-isolation', '--wheel', '--sdist', '--outdir', str(py), str(source / 'python')], check=True)
    npm = shutil.which('npm')
    require(npm is not None, 'npm is unavailable')
    subprocess.run([npm, '--cache', str(out.parent / 'candidate-npm-cache'), 'pack', '--ignore-scripts', '--json', '--pack-destination', str(js)], cwd=source / 'javascript', check=True)
    for folder, patterns in {py: ['*.whl', '*.tar.gz'], js: ['*.tgz']}.items():
        files = []
        for pattern in patterns:
            matches = list(folder.glob(pattern))
            require(len(matches) == 1, f'Expected exactly one {pattern}')
            files.extend(matches)
        (folder / 'SHA256SUMS.json').write_text(json.dumps({f.name: digest(f) for f in files}, indent=2) + '\n', encoding='utf-8')
    archive_inventory = verify_artifacts(source, out)
    manifest = {'schema_version': 1, 'archive_inventory': archive_inventory, **{name: context[name] for name in ('source_sha', 'harness_sha', 'source_snapshot_sha256', 'harness_snapshot_sha256')},
                'build_run_id': os.environ['GITHUB_RUN_ID'], 'repository': os.environ['GITHUB_REPOSITORY'], 'python': sys.version,
                'files': {f.relative_to(out).as_posix(): digest(f) for f in sorted(out.rglob('*')) if f.is_file()}}
    (out / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    require(size <= 4 * 1024 * 1024, f'Frozen artifacts exceed 4 MiB: {size}')
    print(json.dumps({'status': 'passed', 'frozen_artifact_bytes': size, **manifest}, indent=2))


if __name__ == '__main__':
    main()
