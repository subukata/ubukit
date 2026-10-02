"""Read-only conflict detection in synthetic metadata fixtures, no co-install."""
from pathlib import Path
import importlib.util,json,tempfile,sys,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('install_preflight',ROOT/'tools/check_install_environment.py')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
class D:
 def __init__(self,name,root,files=()): self.metadata={'Name':name}; self.version='0.0'; self.root=root; self.files=files
 def locate_file(self,f): return self.root/f
class Tests(unittest.TestCase):
 def test_legacy_owner_blocked_readonly(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); target=root/'rough_cmeans.py'; target.write_text('sentinel=True\n'); before=target.read_bytes()
   d=D('ubukit-exrcm-research',root,['rough_cmeans.py'])
   with patch.object(p.metadata,'distributions',return_value=[d]),patch.object(p.util,'find_spec',return_value=None): r=p.inspect_environment()
   self.assertEqual(r['status'],'blocked'); self.assertEqual(r['legacy_distributions'][0]['name'],'ubukit-exrcm-research'); self.assertEqual(target.read_bytes(),before)
 def test_unowned_shadow_blocked(self):
  for name in ['rough_cmeans','external_metrics','_external_metrics_numba']:
   spec=importlib.util.spec_from_file_location(name,'/synthetic/'+name+'.py')
   with patch.object(p.metadata,'distributions',return_value=[]),patch.object(p.util,'find_spec',side_effect=lambda n:spec if n==name else None): r=p.inspect_environment()
   self.assertEqual(r['status'],'blocked'); self.assertEqual(r['conflicts'][0]['kind'],'existing_import_path')
 def test_empty_is_clear(self):
  with patch.object(p.metadata,'distributions',return_value=[]),patch.object(p.util,'find_spec',return_value=None): r=p.inspect_environment()
  self.assertEqual(r['status'],'clear'); self.assertTrue(r['read_only'])
if __name__=='__main__': unittest.main(verbosity=2)
