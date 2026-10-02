"""Fail closed unless source, verification harness and artifacts match reviewed pins."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from verify_source import SOURCE, VERIFICATION, digest, read_json, require, safe_path, verify

POLICY = 'tools/ci/candidate-policy.json'


def full_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}', value) is not None


def validate_selection(policy, selected_sha, source_snapshot_sha256):
    approved = policy.get('approved_source_sha')
    require(policy.get('schema_version') == 1, 'Unsupported candidate policy')
    require(full_sha(approved), 'No reviewed product-layout candidate selected; dispatch is blocked')
    require(full_sha(selected_sha) and selected_sha == approved, 'Unreviewed candidate SHA')
    expected = policy.get('approved_source_snapshot_sha256')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None,
            'No reviewed source snapshot selected')
    require(source_snapshot_sha256 == expected, 'Candidate source snapshot digest differs from reviewed pin')


def verify_git(root, expected):
    require(full_sha(expected), 'Expected immutable full Git SHA')
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(actual == expected, f'Checkout SHA mismatch: {actual} != {expected}')
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True)
    require(not dirty, f'Checkout must be clean: {root}')


def verify_bundle(bundle, policy, source_sha, source_digest, harness_sha, harness_digest, expected_run_id):
    bundle = Path(bundle).resolve()
    manifest = read_json(bundle, 'build-manifest.json')
    require(manifest.get('schema_version') == 1, 'Unsupported artifact manifest')
    require(manifest['source_sha'] == source_sha, 'Frozen artifacts target a different source commit')
    require(manifest['source_snapshot_sha256'] == source_digest, 'Frozen artifacts target different source/package bytes')
    build_harness = policy.get('approved_build_harness_sha') or harness_sha
    require(full_sha(build_harness) and manifest['harness_sha'] == build_harness,
            'Frozen artifacts target an unreviewed build harness')
    build_snapshot = policy.get('approved_build_harness_snapshot_sha256') if policy.get('approved_build_harness_sha') else harness_digest
    require(isinstance(build_snapshot, str) and re.fullmatch('[0-9a-f]{64}', build_snapshot) is not None,
            'Missing reviewed build harness snapshot provenance')
    require(manifest['harness_snapshot_sha256'] == build_snapshot, 'Frozen artifacts target different build harness bytes')
    require(isinstance(expected_run_id, str) and re.fullmatch('[0-9]+', expected_run_id) is not None, 'Expected immutable artifact run ID')
    require(manifest['build_run_id'] == expected_run_id, 'Frozen artifacts target a different build run')
    require(manifest['repository'] == policy['repository'], 'Frozen artifact repository mismatch')
    expected = manifest['files']
    actual = {f.relative_to(bundle).as_posix() for f in bundle.rglob('*') if f.is_file() and f.relative_to(bundle).as_posix() != 'build-manifest.json'}
    require(bool(expected) and set(expected) == actual, 'Unexpected frozen artifact files')
    require(sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file()) <= 4 * 1024 * 1024, 'Frozen artifacts exceed 4 MiB')
    for name, checksum in expected.items():
        require(re.fullmatch('[0-9a-f]{64}', checksum) is not None, f'Invalid artifact digest: {name}')
        require(digest(safe_path(bundle, name)) == checksum, f'Artifact hash mismatch: {name}')
    return manifest


def verify_checkout(root, bundle=None, harness=None, environ=None):
    env = os.environ if environ is None else environ
    root = Path(root).resolve()
    harness = Path(harness or Path(__file__).resolve().parents[3]).resolve()
    policy = read_json(harness, POLICY)
    selected = env.get('CANDIDATE_SHA')
    source_digest = digest(root / SOURCE)
    validate_selection(policy, selected, source_digest)
    require(env.get('GITHUB_REPOSITORY') == policy['repository'], 'Unexpected repository')
    harness_sha = env.get('GITHUB_SHA')
    verify_git(root, selected)
    verify_git(harness, harness_sha)
    source_result = verify(root)
    harness_result = verify(harness)
    require((harness / SOURCE).read_bytes() == (root / SOURCE).read_bytes(),
            'Verification checkout changes reviewed runtime/package bytes')
    if bundle is not None:
        verify_bundle(bundle, policy, selected, source_digest, harness_sha, digest(harness / VERIFICATION),
                      env.get('FROZEN_ARTIFACT_RUN_ID') or env.get('GITHUB_RUN_ID'))
        from verify_artifact_contents import verify_artifacts
        verify_artifacts(root, bundle)
    return {'status': 'passed', 'source_sha': selected, 'harness_sha': harness_sha,
            'source_snapshot_sha256': source_digest, 'harness_snapshot_sha256': digest(harness / VERIFICATION),
            'source': source_result, 'harness': harness_result, 'frozen_artifacts_verified': bundle is not None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('bundle', nargs='?', type=Path)
    parser.add_argument('harness', nargs='?', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_checkout(args.candidate, args.bundle, args.harness), indent=2))
