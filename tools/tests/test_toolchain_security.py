"""Offline guards for the reviewed CPython 3.12 verification dependency locks."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
LOCKS = ROOT / 'tools/ci/constraints'


class ToolchainSecurityTests(unittest.TestCase):
    def test_security_fixed_versions_and_official_wheel_provenance(self):
        data = json.loads((LOCKS / 'toolchain-provenance.json').read_text())
        packages = {p['name'].lower(): p for p in data['packages']}
        for name, version in [('pip', '26.2'), ('pytest', '9.0.3'), ('setuptools', '83.0.0')]:
            self.assertEqual(packages[name]['version'], version)
        for name, package in packages.items():
            self.assertTrue(package['metadata_url'].startswith('https://pypi.org/pypi/'))
            self.assertTrue(package['files'])
            for wheel in package['files']:
                self.assertTrue(wheel['filename'].endswith('.whl'))
                self.assertTrue(wheel['url'].startswith('https://files.pythonhosted.org/packages/'))
                self.assertRegex(wheel['sha256'], r'^[0-9a-f]{64}$')
                self.assertGreater(wheel['size'], 0)

    def test_active_locks_are_complete_and_match_provenance(self):
        data = json.loads((LOCKS / 'toolchain-provenance.json').read_text())
        expected = {p['name'].lower(): p for p in data['packages']}
        seen = set()
        for filename in ['installer.txt', 'build-py312-locked.txt', 'test-py312-locked.txt', 'base-py312-locked.txt']:
            text = (LOCKS / filename).read_text().replace('\\\n', '')
            for line in text.splitlines():
                if not line or line.startswith(('#', '-r ')):
                    continue
                match = re.fullmatch(r'([A-Za-z0-9_-]+)==([0-9.]+)(?:; sys_platform == "win32")?\s+(.+)', line)
                self.assertIsNotNone(match, line)
                name, version, hashes = match.groups()
                package = expected[name.lower()]
                self.assertEqual(version, package['version'])
                self.assertEqual(set(hashes.split()), {'--hash=sha256:' + f['sha256'] for f in package['files']})
                seen.add(name.lower())
        self.assertEqual(seen, set(expected))
        self.assertTrue((LOCKS / 'test-py312-locked.txt').read_text().startswith('-r base-py312-locked.txt\n'))

    def test_active_driver_uses_hashes_and_offline_builds(self):
        driver = (ROOT / 'tools/ci/scripts/python_clean_install.py').read_text()
        self.assertNotIn('pytest==8.4.2', driver)
        self.assertIn("'--require-hashes'", driver)
        self.assertIn("'PIP_NO_INDEX':'1'", driver)
        self.assertIn("'PIP_FIND_LINKS':wheelhouse.as_uri()", driver)
        self.assertIn("'--build-constraint',a.build_constraints.resolve()", driver)
        workflow = (ROOT / '.github/workflows/private-candidate.yml').read_text()
        self.assertIn('--require-hashes --no-deps -r harness/tools/ci/constraints/installer.txt', workflow)
        self.assertIn('--require-hashes -r harness/tools/ci/constraints/build-py312-locked.txt', workflow)


if __name__ == '__main__':
    unittest.main()
