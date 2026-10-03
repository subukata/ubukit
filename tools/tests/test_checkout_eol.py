"""Offline regression: Git CRLF conversion must preserve reviewed source bytes."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('git'), 'Git is required for checkout conversion regression')
class CheckoutEolTests(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.check_output(
            ['git', '-C', str(root), *args], stderr=subprocess.STDOUT,
        )

    def test_autocrlf_checkout_preserves_manifest_inputs_and_binary_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, checkout = base / 'source', base / 'checkout'
            source.mkdir()
            # Real preserved inputs, including the 10,297-byte WAT that previously
            # changed to 10,423 bytes. Copies never modify the tracked originals.
            records = json.loads((ROOT / 'javascript/wasm/SOURCE_FILES.json').read_text())['files']
            names = ['.gitattributes', 'javascript/wasm/SOURCE_FILES.json']
            names += ['javascript/wasm/' + record['path'] for record in records]
            names += ['javascript/src/fcm-stable.js', 'docs/assets/ubukit-logo-b.png',
                      'docs/assets/ubukit-mascot.png']
            expected = {}
            for name in names:
                data = (ROOT / name).read_bytes()
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                expected[name] = data
            # Deliberately text-looking binary data tests explicit binary rules,
            # independently of Git's NUL-based binary heuristic.
            for suffix in ['wasm', 'png', 'zip', 'npy']:
                name = 'fixtures/text-looking.' + suffix
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'first\r\nsecond\n')
                expected[name] = target.read_bytes()
            self.git(source, 'init', '-q')
            self.git(source, '-c', 'core.autocrlf=false', 'add', '.')
            self.git(source, '-c', 'user.name=Offline EOL Fixture',
                     '-c', 'user.email=fixture@example.invalid', 'commit', '-qm',
                     'local line-ending fixture, not a remote release identity')
            self.git(base, '-c', 'core.autocrlf=true', 'clone', '--no-local',
                     '--no-checkout', str(source), str(checkout))
            self.git(checkout, '-c', 'core.autocrlf=true', 'checkout', '--force', 'HEAD')
            for name, data in expected.items():
                with self.subTest(path=name):
                    self.assertEqual((checkout / name).read_bytes(), data)
            self.assertEqual(len((checkout / 'javascript/wasm/original-wat/rmcm/f32_candidates.wat').read_bytes()), 10297)
            self.assertEqual(self.git(checkout, '-c', 'core.autocrlf=true', 'status', '--porcelain'), b'')
            for name in ['fixtures/text-looking.wasm', 'fixtures/text-looking.png']:
                attributes = self.git(checkout, 'check-attr', 'text', '--', name).decode()
                self.assertIn(': text: unset', attributes)


if __name__ == '__main__':
    unittest.main()
