"""Read-only namespace collision detection using synthetic metadata fixtures."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('install_preflight', ROOT / 'tools/check_install_environment.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class D:
    def __init__(self, name, root, files=()):
        self.metadata = {'Name': name}
        self.version = '0.0'
        self.root = root
        self.files = files

    def locate_file(self, file):
        return self.root / file


class Tests(unittest.TestCase):
    def inspect(self, distributions=(), specs=None):
        specs = specs or {}
        with patch.object(p.metadata, 'distributions', return_value=distributions), \
                patch.object(p.util, 'find_spec', side_effect=specs.get):
            return p.inspect_environment()

    def test_unrelated_legacy_distribution_is_allowed_readonly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'rough_cmeans.py'
            target.write_text('sentinel=True\n')
            before = target.read_bytes()
            dist = D('ubukit-exrcm-research', root, ['rough_cmeans.py'])
            legacy_spec = importlib.util.spec_from_file_location('rough_cmeans', target)
            result = self.inspect([dist], {'rough_cmeans': legacy_spec})
            self.assertEqual(result['status'], 'clear')
            self.assertEqual(result['legacy_distributions'], [])
            self.assertEqual(target.read_bytes(), before)

    def test_old_bundled_layout_requires_fresh_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'rough_cmeans.py'
            target.write_text('sentinel=True\n')
            before = target.read_bytes()
            dist = D('ubukit-bundled-local-preview', root,
                     ['ubukit/__init__.py', 'rough_cmeans.py'])
            result = self.inspect([dist])
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(result['legacy_distributions'][0]['name'], 'ubukit-bundled-local-preview')
            self.assertEqual(target.read_bytes(), before)

    def test_unowned_ubukit_shadow_is_blocked(self):
        shadow = importlib.util.spec_from_file_location('ubukit', '/synthetic/ubukit/__init__.py')
        result = self.inspect(specs={'ubukit': shadow})
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['conflicts'][0]['kind'], 'existing_import_path')

    def test_previous_preview_namespace_requires_fresh_environment(self):
        root = Path('/synthetic')
        dist = D('ubukit-bundled-local-preview', root, ['ubukit/__init__.py'])
        old_spec = importlib.util.spec_from_file_location('ubukit', root / 'ubukit/__init__.py')
        result = self.inspect([dist], {'ubukit': old_spec})
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['legacy_distributions'][0]['name'], 'ubukit-bundled-local-preview')
        self.assertTrue(result['read_only'])

    def test_foreign_ubukit_owner_is_blocked(self):
        dist = D('another-owner', Path('/synthetic'), ['ubukit/__init__.py'])
        result = self.inspect([dist])
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['conflicts'][0]['kind'], 'foreign_file_owner')

    def test_current_owned_layout_is_clear(self):
        root = Path('/synthetic')
        dist = D('ubukit', root,
                 ['ubukit/__init__.py', 'ubukit/_impl/rough_cmeans.py'])
        own_spec = importlib.util.spec_from_file_location('ubukit', root / 'ubukit/__init__.py')
        result = self.inspect([dist], {'ubukit': own_spec})
        self.assertEqual(result['status'], 'clear')
        self.assertTrue(result['read_only'])

    def test_empty_is_clear(self):
        result = self.inspect()
        self.assertEqual(result['status'], 'clear')
        self.assertTrue(result['read_only'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
