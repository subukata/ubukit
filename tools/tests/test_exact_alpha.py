"""Offline source-selection fixtures. Local Git commits are never remote identities."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tools/ci/scripts'))
from exact_alpha import IDENTITY, PINS, prepare_selection, verify_exact_checkout, verify_identity
from verify_checkout import POLICY
from verify_source import SOURCE, VERIFICATION, digest, verify
from test_release_contract import fixture, snapshots, write


class ExactAlphaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source, self.harness = self.base / 'source', self.base / 'harness'
        self.source.mkdir(); self.harness.mkdir()
        fixture(self.source)
        self.source_sha = self.commit(self.source)
        shutil.copytree(self.source, self.harness, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git'))
        self.identity = {'schema_version': 1, 'repository': 'subukata/ubukit',
                         'source_tree_sha1': self.git(self.source, 'rev-parse', 'HEAD^{tree}'),
                         'source_file_count': len(self.git(self.source, 'ls-tree', '-r', '--name-only', 'HEAD').splitlines()),
                         'source_snapshot_sha256': digest(self.source / SOURCE),
                         'python': {'distribution': 'ubukit', 'version': '1.2.3'},
                         'javascript': {'distribution': 'ubukit-js', 'version': '2.3.4'}}
        self.save_identity()
        self.output = self.base / 'selection'

    def git(self, root, *args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()

    def commit(self, root):
        self.git(root, 'init', '-q')
        self.git(root, 'add', '.')
        self.git(root, '-c', 'user.name=Offline Alpha Fixture', '-c', 'user.email=fixture@example.invalid',
                 'commit', '--allow-empty', '-qm', 'local fixture, not a remote release identity')
        return self.git(root, 'rev-parse', 'HEAD')

    def save_identity(self):
        write(self.harness, IDENTITY, self.identity)
        snapshots(self.harness)

    def select(self, **changes):
        args = dict(source=self.source, harness=self.harness, source_sha=self.source_sha, output=self.output)
        args.update(changes)
        return prepare_selection(**args)

    def test_exact_selection_does_not_mutate_either_input(self):
        before = {str(p): digest(p) for root in [self.source, self.harness]
                  for p in root.rglob('*') if p.is_file() and '.git' not in p.parts}
        result = self.select()
        self.assertEqual(result['source_sha'], self.source_sha)
        self.assertEqual(result['first_dispatch_inputs']['artifact_run_id'], '')
        self.assertFalse(result['first_dispatch_inputs']['free_budget_rechecked'])
        self.assertFalse(result['remote_source_existence_verified'])
        self.assertEqual(before, {str(p): digest(p) for root in [self.source, self.harness]
                                  for p in root.rglob('*') if p.is_file() and '.git' not in p.parts})
        self.assertEqual(set(result['overlay_files']), {POLICY, VERIFICATION})

    def test_two_file_selection_overlay_produces_valid_harness(self):
        self.select()
        for name in (POLICY, VERIFICATION):
            shutil.copy2(self.output / name, self.harness / name)
        self.assertEqual(verify(self.harness)['status'], 'passed')
        policy = json.loads((self.harness / POLICY).read_text())
        self.assertIsNone(policy['approved_build_harness_sha'])
        self.assertIsNone(policy['approved_build_harness_snapshot_sha256'])
        harness_sha = self.commit(self.harness)
        env = {'GITHUB_REPOSITORY': 'subukata/ubukit', 'GITHUB_SHA': harness_sha, 'CANDIDATE_SHA': self.source_sha}
        self.assertEqual(verify_exact_checkout(self.source, harness=self.harness, environ=env)['status'], 'passed')

    def test_unselected_ci_still_fails_closed(self):
        harness_sha = self.commit(self.harness)
        env = {'GITHUB_REPOSITORY': 'subukata/ubukit', 'GITHUB_SHA': harness_sha, 'CANDIDATE_SHA': self.source_sha}
        with self.assertRaisesRegex(ValueError, 'No reviewed product-layout candidate selected'):
            verify_exact_checkout(self.source, harness=self.harness, environ=env)

    def test_mutable_or_wrong_source_sha_is_rejected(self):
        for value in ['HEAD', 'main', self.source_sha[:7], 'A' * 40, 'a' * 40]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.select(source_sha=value)

    def test_dirty_source_is_rejected(self):
        (self.source / 'untracked.txt').write_text('unreviewed')
        with self.assertRaisesRegex(ValueError, 'Checkout must be clean'):
            self.select()

    def test_same_product_but_different_committed_harness_tree_is_rejected(self):
        (self.source / 'README.md').write_text('not in package snapshot but changes the frozen tree')
        self.source_sha = self.commit(self.source)
        with self.assertRaisesRegex(ValueError, 'Source tree differs from frozen alpha'):
            self.select()

    def test_wrong_source_file_count_is_rejected(self):
        self.identity['source_file_count'] += 1
        self.save_identity()
        with self.assertRaisesRegex(ValueError, 'file count differs'):
            self.select()

    def test_preview_snapshot_is_rejected(self):
        self.identity['source_snapshot_sha256'] = 'b' * 64
        self.save_identity()
        with self.assertRaisesRegex(ValueError, 'Not the exact alpha source snapshot'):
            self.select()

    def test_wrong_package_names_and_versions_are_rejected(self):
        original = copy.deepcopy(self.identity)
        for language in ('python', 'javascript'):
            for key in ('distribution', 'version'):
                self.identity = copy.deepcopy(original)
                self.identity[language][key] = 'preview'
                self.save_identity()
                with self.subTest(language=language, key=key), self.assertRaisesRegex(ValueError, 'alpha identity differs'):
                    self.select()

    def test_existing_source_or_reuse_pins_are_rejected(self):
        initial = json.loads((self.harness / POLICY).read_text())
        for name in PINS:
            current = {**initial, name: 'a' * 40}
            write(self.harness, POLICY, current)
            snapshots(self.harness)
            with self.subTest(pin=name), self.assertRaisesRegex(ValueError, 'Expected unselected fresh-build policy'):
                self.select()

    def test_wrong_policy_repository_is_rejected(self):
        current = json.loads((self.harness / POLICY).read_text())
        current['repository'] = 'other/repository'
        write(self.harness, POLICY, current); snapshots(self.harness)
        with self.assertRaisesRegex(ValueError, 'repository or schema differs'):
            self.select()

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir(); sentinel = self.output / 'keep.txt'; sentinel.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'Fresh output directory required'):
            self.select()
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_output_inside_input_is_rejected(self):
        for root in (self.source, self.harness):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, 'outside both input checkouts'):
                self.select(output=root / 'generated-selection')

    def test_stale_harness_snapshot_is_rejected(self):
        (self.harness / 'tools/ci/new-unreviewed.txt').write_text('unreviewed')
        with self.assertRaisesRegex(ValueError, 'complete harness input set'):
            self.select()

    def test_ci_wrong_repository_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected exact-alpha repository'):
            verify_exact_checkout(self.source, harness=self.harness, environ={'GITHUB_REPOSITORY': 'other/repo'})


if __name__ == '__main__':
    unittest.main()
