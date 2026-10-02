"""Bounded, offline release-binding fixtures. No builds, installation or CI dispatch."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import tarfile
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tools/ci/scripts'))
from verify_checkout import validate_selection, verify_bundle, verify_checkout
from verify_artifact_contents import verify_artifacts
from verify_source import CONTRACT, SOURCE, VERIFICATION, digest, file_rows, product_snapshot, verification_paths, verify


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + '\n' if isinstance(content, (dict, list)) else content, encoding='utf-8')


def snapshots(root):
    write(root, SOURCE, product_snapshot(root))
    write(root, VERIFICATION, {'schema_version': 1, 'files': file_rows(root, verification_paths(root))})


def fixture(root):
    write(root, 'python/src/ubukit/__init__.py', '__version__ = "1.2.3"\n')
    write(root, 'javascript/src/index.js', 'export const value = 1;\n')
    write(root, 'python/pyproject.toml', '[project]\nname = "ubukit-bundled-local-preview"\nversion = "1.2.3"\nlicense-files = []\n')
    write(root, 'python/SOURCE_MANIFEST.json', {'version': '1.2.3', 'files': [{'path': 'src/ubukit/__init__.py', 'sha256': digest(root / 'python/src/ubukit/__init__.py')}]})
    write(root, 'javascript/package.json', {'name': 'ubukit-js', 'version': '2.3.4', 'exports': {'.': './src/index.js'}, 'files': ['src/index.js', 'SOURCE_MANIFEST.json']})
    write(root, 'javascript/SOURCE_MANIFEST.json', {'version': '2.3.4', 'runtime_files': {'src/index.js': {'sha256': digest(root / 'javascript/src/index.js')}}})
    write(root, 'python/verification/python_api.py', '# exact facade fixture\n')
    write(root, 'python/verification/runtime_tests/python_api.py', '# exact facade fixture\n')
    contract = copy.deepcopy(json.loads((REPO / CONTRACT).read_text()))
    contract['languages']['python']['package_metadata'] = ['pyproject.toml', 'SOURCE_MANIFEST.json']
    contract['languages']['javascript']['entrypoints'] = ['.']
    write(root, CONTRACT, contract)
    write(root, 'tools/ci/candidate-policy.json', {'schema_version': 1, 'repository': 'subukata/ubukit', 'approved_source_sha': None})
    snapshots(root)


def policy(source='a' * 40, snapshot='b' * 64, build_harness=None):
    return {'schema_version': 1, 'repository': 'subukata/ubukit', 'approved_source_sha': source,
            'approved_source_snapshot_sha256': snapshot, 'approved_build_harness_sha': build_harness,
            'approved_build_harness_snapshot_sha256': 'd' * 64 if build_harness else None}


class ProductSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)

    def test_correct_source_uses_manifest_versions_and_counts(self):
        result = verify(self.root)
        self.assertEqual(result['runtime_files'], {'python': 1, 'javascript': 1})
        self.assertEqual(result['versions'], {'python': '1.2.3', 'javascript': '2.3.4'})

    def test_runtime_byte_drift_cannot_be_blessed_by_refresh(self):
        write(self.root, 'javascript/src/index.js', 'changed runtime\n')
        with self.assertRaisesRegex(ValueError, 'Runtime hash mismatch'):
            product_snapshot(self.root)

    def test_extra_runtime_fails(self):
        write(self.root, 'python/src/ubukit/unreviewed.py', '# extra\n')
        with self.assertRaisesRegex(ValueError, 'runtime path set mismatch'):
            verify(self.root)

    def test_package_metadata_drift_fails(self):
        path = self.root / 'python/pyproject.toml'
        path.write_text(path.read_text() + '# changed build metadata\n')
        with self.assertRaisesRegex(ValueError, 'Product snapshot differs'):
            verify(self.root)

    def test_extra_harness_input_fails(self):
        write(self.root, 'tools/ci/scripts/unreviewed.py', '# extra\n')
        with self.assertRaisesRegex(ValueError, 'complete harness input set'):
            verify(self.root)

    def test_generated_outputs_do_not_expand_harness_inputs(self):
        for name in ['python/build/output.py', 'python/src/example.egg-info/PKG-INFO', 'javascript/node_modules/x/index.js', 'python/verification/results/a.json', 'javascript/wasm/.generated/out.wasm']:
            write(self.root, name, 'generated\n')
        self.assertEqual(verify(self.root)['status'], 'passed')

    def test_unsafe_package_path_fails(self):
        metadata = json.loads((self.root / 'javascript/package.json').read_text())
        metadata['files'].append('../secret')
        write(self.root, 'javascript/package.json', metadata)
        with self.assertRaisesRegex(ValueError, 'Unsafe path'):
            product_snapshot(self.root)

    def test_duplicate_runtime_manifest_fails(self):
        path = self.root / 'python/SOURCE_MANIFEST.json'
        manifest = json.loads(path.read_text())
        manifest['files'].append(manifest['files'][0])
        write(self.root, 'python/SOURCE_MANIFEST.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Duplicate Python'):
            product_snapshot(self.root)

    def test_symlinked_runtime_fails(self):
        original = self.root / 'python/src/ubukit/__init__.py'
        target = self.root / 'copy.py'
        target.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            product_snapshot(self.root)


class SelectionTests(unittest.TestCase):
    def test_reviewed_sha_and_snapshot_pass(self):
        validate_selection(policy(), 'a' * 40, 'b' * 64)

    def test_wrong_full_sha_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unreviewed candidate SHA'):
            validate_selection(policy(), 'c' * 40, 'b' * 64)

    def test_arbitrary_refs_short_sha_and_uppercase_rejected(self):
        for selected in ['HEAD', 'main', 'a' * 8, 'A' * 40, None]:
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, 'Unreviewed candidate SHA'):
                validate_selection(policy(), selected, 'b' * 64)

    def test_unselected_policy_stays_closed(self):
        current = policy(source=None)
        with self.assertRaisesRegex(ValueError, 'No reviewed product-layout candidate selected'):
            validate_selection(current, 'a' * 40, 'b' * 64)

    def test_wrong_snapshot_rejected(self):
        with self.assertRaisesRegex(ValueError, 'snapshot digest differs'):
            validate_selection(policy(), 'a' * 40, 'd' * 64)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write(self.root, 'python/frozen.whl', 'immutable artifact\n')
        self.manifest = {'schema_version': 1, 'source_sha': 'a' * 40, 'source_snapshot_sha256': 'b' * 64,
                         'harness_sha': 'c' * 40, 'harness_snapshot_sha256': 'd' * 64,
                         'build_run_id': '12345', 'repository': 'subukata/ubukit',
                         'files': {'python/frozen.whl': digest(self.root / 'python/frozen.whl')}}
        self.save()

    def save(self):
        write(self.root, 'build-manifest.json', self.manifest)

    def verify(self, selected_policy=None):
        return verify_bundle(self.root, selected_policy or policy(), 'a' * 40, 'b' * 64, 'c' * 40, 'd' * 64, '12345')

    def test_exact_artifact_provenance_passes(self):
        self.verify()

    def test_wrong_source_sha_rejected(self):
        self.manifest['source_sha'] = 'e' * 40
        self.save()
        with self.assertRaisesRegex(ValueError, 'different source commit'):
            self.verify()

    def test_wrong_artifact_hash_rejected(self):
        write(self.root, 'python/frozen.whl', 'changed artifact\n')
        with self.assertRaisesRegex(ValueError, 'Artifact hash mismatch'):
            self.verify()

    def test_extra_artifact_rejected(self):
        write(self.root, 'unbound.tgz', 'extra\n')
        with self.assertRaisesRegex(ValueError, 'Unexpected frozen artifact'):
            self.verify()

    def test_wrong_build_run_rejected(self):
        self.manifest['build_run_id'] = '99999'
        self.save()
        with self.assertRaisesRegex(ValueError, 'different build run'):
            self.verify()

    def test_wrong_harness_snapshot_rejected(self):
        self.manifest['harness_snapshot_sha256'] = 'e' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'different build harness bytes'):
            self.verify()

    def test_unreviewed_build_harness_rejected(self):
        self.manifest['harness_sha'] = 'e' * 40
        self.save()
        with self.assertRaisesRegex(ValueError, 'unreviewed build harness'):
            self.verify()

    def test_explicit_prior_build_harness_can_be_reused(self):
        self.manifest['harness_sha'] = 'e' * 40
        self.save()
        self.verify(policy(build_harness='e' * 40))


class CheckoutTests(unittest.TestCase):
    def test_separate_source_and_harness_commit_with_dirty_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            candidate, harness = base / 'candidate', base / 'harness'
            candidate.mkdir(); harness.mkdir()
            fixture(candidate); fixture(harness)
            def git(root, *args):
                return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
            def commit(root):
                git(root, 'init', '-q')
                git(root, 'add', '.')
                git(root, '-c', 'user.name=Release Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'offline fixture')
                return git(root, 'rev-parse', 'HEAD')
            source_sha = commit(candidate)
            write(harness, 'tools/ci/candidate-policy.json', policy(source_sha, digest(candidate / SOURCE)))
            snapshots(harness)
            harness_sha = commit(harness)
            self.assertNotEqual(source_sha, harness_sha)
            env = {'CANDIDATE_SHA': source_sha, 'GITHUB_SHA': harness_sha, 'GITHUB_REPOSITORY': 'subukata/ubukit'}
            self.assertEqual(verify_checkout(candidate, harness=harness, environ=env)['status'], 'passed')
            with self.assertRaisesRegex(ValueError, 'Unreviewed candidate SHA'):
                verify_checkout(candidate, harness=harness, environ={**env, 'CANDIDATE_SHA': harness_sha})
            with self.assertRaisesRegex(ValueError, 'Checkout SHA mismatch'):
                verify_checkout(candidate, harness=harness, environ={**env, 'GITHUB_SHA': 'e' * 40})
            write(candidate, 'untracked.txt', 'dirty\n')
            with self.assertRaisesRegex(ValueError, 'Checkout must be clean'):
                verify_checkout(candidate, harness=harness, environ=env)


class ArchiveInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.artifacts = self.root / 'source', self.root / 'artifacts'
        self.source.mkdir(); fixture(self.source)
        (self.artifacts / 'python').mkdir(parents=True)
        (self.artifacts / 'javascript').mkdir()
        contract = json.loads((self.source / CONTRACT).read_text())
        spec = contract['languages']['python']
        runtime = ['src/ubukit/__init__.py']
        self.sdist = {name: (self.source / 'python' / name).read_bytes() for name in runtime + spec['package_metadata']}
        self.sdist.update({name: b'generated\n' for name in spec['sdist_generated_files']})
        self.sdist.update({'src/ubukit_bundled_local_preview.egg-info/' + name: b'generated\n' for name in spec['egg_info_files']})
        self.sdist['PKG-INFO'] = b'Version: 1.2.3\n'
        self.wheel = {'ubukit/__init__.py': (self.source / 'python/src/ubukit/__init__.py').read_bytes()}
        self.wheel.update({'ubukit_bundled_local_preview-1.2.3.dist-info/' + name: b'generated\n' for name in spec['wheel_metadata_files']})
        self.wheel['ubukit_bundled_local_preview-1.2.3.dist-info/METADATA'] = b'Version: 1.2.3\n'
        metadata = json.loads((self.source / 'javascript/package.json').read_text())
        self.npm = {'package/' + name: (self.source / 'javascript' / name).read_bytes() for name in ['package.json', *metadata['files']]}

    def save(self):
        def tar(path, files):
            with tarfile.open(path, 'w:gz') as archive:
                for name, data in files.items():
                    item = tarfile.TarInfo(name); item.size = len(data)
                    archive.addfile(item, io.BytesIO(data))
        tar(self.artifacts / 'python/candidate.tar.gz', {'ubukit_bundled_local_preview-1.2.3/' + n: data for n, data in self.sdist.items()})
        tar(self.artifacts / 'javascript/candidate.tgz', self.npm)
        with zipfile.ZipFile(self.artifacts / 'python/candidate.whl', 'w') as archive:
            for name, data in self.wheel.items():
                archive.writestr(name, data)

    def check(self):
        self.save()
        return verify_artifacts(self.source, self.artifacts)

    def test_exact_archive_inventories_pass(self):
        self.assertEqual(self.check()['status'], 'passed')

    def test_extra_sdist_harness_file_fails(self):
        self.sdist['tools/unwanted.py'] = b'extra\n'
        with self.assertRaisesRegex(ValueError, 'Sdist inventory differs'):
            self.check()

    def test_changed_sdist_metadata_fails(self):
        self.sdist['pyproject.toml'] += b'# changed\n'
        with self.assertRaisesRegex(ValueError, 'Sdist source-byte mismatch'):
            self.check()

    def test_extra_wheel_runtime_fails(self):
        self.wheel['ubukit/unwanted.py'] = b'extra\n'
        with self.assertRaisesRegex(ValueError, 'Wheel inventory differs'):
            self.check()

    def test_changed_npm_runtime_fails(self):
        self.npm['package/src/index.js'] = b'changed\n'
        with self.assertRaisesRegex(ValueError, 'npm source-byte mismatch'):
            self.check()

    def test_unsafe_archive_member_fails(self):
        self.npm['../outside'] = b'unsafe\n'
        with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
            self.check()


class WorkflowStaticTests(unittest.TestCase):
    def test_manual_budget_and_immutable_actions_are_preserved(self):
        workflow = (REPO / '.github/workflows/private-candidate.yml').read_text()
        self.assertIn('workflow_dispatch:', workflow)
        for forbidden in ['  push:', '  pull_request:', '  schedule:', 'strategy:', 'preview/', 'ci/release/', 'actions/cache', 'publish']:
            self.assertNotIn(forbidden, workflow)
        self.assertIn('default: false', workflow)
        self.assertEqual(workflow.count('inputs.free_budget_rechecked'), 2)
        self.assertIn('Actions $0 Stop usage budget', workflow)
        self.assertIn('retention-days: 1', workflow)
        self.assertIn('timeout-minutes: 10', workflow)
        self.assertIn('timeout-minutes: 45', workflow)
        self.assertIn('persist-credentials: false', workflow)
        import re
        for action in re.findall(r'uses: (\S+)', workflow):
            self.assertRegex(action, r'^[^@]+@[0-9a-f]{40}$')
        self.assertIn('--javascript-root harness/javascript', workflow)
        self.assertIn('--python-root harness/python', workflow)


if __name__ == '__main__':
    unittest.main(verbosity=2)
