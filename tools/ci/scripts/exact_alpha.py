"""Bind the frozen alpha tree to a real source commit; never dispatch or publish."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib

from verify_checkout import POLICY, full_sha, verify_checkout, verify_git
from verify_source import SOURCE, VERIFICATION, digest, read_json, require, verify

IDENTITY = 'tools/ci/exact-alpha.json'
PINS = ('approved_source_sha', 'approved_source_snapshot_sha256',
        'approved_build_harness_sha', 'approved_build_harness_snapshot_sha256')


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode('utf-8')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def verify_identity(source, harness, source_sha):
    """Read-only verification, including all 618 frozen source files and modes."""
    source, harness = Path(source).resolve(), Path(harness).resolve()
    identity = read_json(harness, IDENTITY)
    require(identity.get('schema_version') == 1, 'Unsupported exact-alpha identity')
    require(full_sha(identity.get('source_tree_sha1')), 'Expected full frozen source tree SHA')
    verify_git(source, source_sha)
    tree_sha = git(source, 'rev-parse', 'HEAD^{tree}')
    require(tree_sha == identity['source_tree_sha1'], 'Source tree differs from frozen alpha')
    paths = subprocess.check_output(['git', '-C', str(source), 'ls-tree', '-rz', '--name-only', 'HEAD']).split(b'\0')
    require(len([p for p in paths if p]) == identity['source_file_count'], 'Frozen alpha file count differs')
    result = verify(source)
    verify(harness)
    require(digest(source / SOURCE) == identity['source_snapshot_sha256'], 'Not the exact alpha source snapshot')
    require((source / SOURCE).read_bytes() == (harness / SOURCE).read_bytes(), 'Harness changes alpha package inputs')
    python = tomllib.loads((source / 'python/pyproject.toml').read_text(encoding='utf-8'))['project']
    javascript = read_json(source, 'javascript/package.json')
    for language, metadata in [('python', python), ('javascript', javascript)]:
        require(metadata['name'] == identity[language]['distribution'] and
                metadata['version'] == identity[language]['version'], f'{language} alpha identity differs')
    return {'status': 'passed', 'source_sha': source_sha, 'source_tree_sha1': tree_sha,
            'source_snapshot_sha256': identity['source_snapshot_sha256'], 'source': result}


def prepare_selection(source, harness, source_sha, output):
    """Emit a two-file overlay for review; never change either input checkout."""
    source, harness, output = (Path(p).resolve() for p in (source, harness, output))
    require(not output.exists(), 'Fresh output directory required')
    require(not output.is_relative_to(source) and not output.is_relative_to(harness),
            'Selection output must be outside both input checkouts')
    context = verify_identity(source, harness, source_sha)
    identity, policy = read_json(harness, IDENTITY), read_json(harness, POLICY)
    require(policy.get('schema_version') == 1 and policy.get('repository') == identity['repository'],
            'Candidate policy repository or schema differs')
    require(all(policy.get(name) is None for name in PINS), 'Expected unselected fresh-build policy')
    policy = copy.deepcopy(policy)
    policy.update(approved_source_sha=source_sha,
                  approved_source_snapshot_sha256=context['source_snapshot_sha256'],
                  approved_build_harness_sha=None,
                  approved_build_harness_snapshot_sha256=None,
                  status='selected_exact_alpha_pending_ci',
                  note='Exact frozen alpha source selected by immutable commit and tree. '
                       'Fresh Linux build required; this selection is not a CI result or publication approval. '
                       'Any different build-harness reuse requires separately reviewed pins.')
    snapshot = read_json(harness, VERIFICATION)
    matches = [row for row in snapshot['files'] if row['path'] == POLICY]
    require(len(matches) == 1, 'Policy must be bound exactly once in harness snapshot')
    matches[0]['sha256'] = hashlib.sha256(encoded(policy)).hexdigest()
    output.mkdir(parents=True)
    for name, value in [(POLICY, policy), (VERIFICATION, snapshot)]:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded(value))
    receipt = {**context, 'status': 'selection_overlay_prepared_for_review',
               'repository': identity['repository'], 'input_harness_snapshot_sha256': digest(harness / VERIFICATION),
               'output_harness_snapshot_sha256': digest(output / VERIFICATION),
               'overlay_files': {name: digest(output / name) for name in (POLICY, VERIFICATION)},
               'first_dispatch_inputs': {'stage': 'linux', 'candidate_sha': source_sha,
                                         'artifact_run_id': '', 'free_budget_rechecked': False},
               'remote_source_existence_verified': False, 'dispatch_performed': False,
               'publication_approved': False}
    (output / 'selection-receipt.json').write_bytes(encoded(receipt))
    return receipt


def verify_exact_checkout(source, bundle=None, harness=None, environ=None):
    env = os.environ if environ is None else environ
    harness = Path(harness or Path(__file__).resolve().parents[3]).resolve()
    identity = read_json(harness, IDENTITY)
    require(env.get('GITHUB_REPOSITORY') == identity['repository'], 'Unexpected exact-alpha repository')
    context = verify_checkout(source, bundle, harness, env)
    alpha = verify_identity(source, harness, env.get('CANDIDATE_SHA'))
    return {**context, 'exact_alpha': alpha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    selection = commands.add_parser('select', help='Prepare reviewed-policy overlay locally; no remote actions')
    selection.add_argument('--source', required=True, type=Path)
    selection.add_argument('--harness', required=True, type=Path)
    selection.add_argument('--source-sha', required=True)
    selection.add_argument('--output', required=True, type=Path)
    check = commands.add_parser('verify', help='Read-only exact-alpha checkout and optional artifact check')
    check.add_argument('source', type=Path)
    check.add_argument('bundle', nargs='?', type=Path)
    check.add_argument('harness', nargs='?', type=Path)
    args = parser.parse_args()
    if args.command == 'select':
        result = prepare_selection(args.source, args.harness, args.source_sha, args.output)
    else:
        result = verify_exact_checkout(args.source, args.bundle, args.harness)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
