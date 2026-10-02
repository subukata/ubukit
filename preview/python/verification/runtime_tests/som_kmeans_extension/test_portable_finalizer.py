import importlib.abc,json,os,subprocess,sys,unittest
from pathlib import Path
import ubukit._impl.portable_accel as candidate
from .._support import portable_oracle as checkpoint, kmeans_reference as reference
import numpy as np
class PortableFinalizer(unittest.TestCase):
 def test_direct_reference_all_layouts(self):
  rng=np.random.default_rng(128311)
  for n,d,k in [(1,1,1),(7,2,8),(37,8,16),(129,32,17),(131,128,32),(64,784,16)]:
   for scale,offset in [(1.,0.),(1e-160,0.),(1.,1e12)]:
    X=rng.normal(size=(n,d))*scale+offset;C=rng.normal(size=(k,d))*scale+offset
    for layout in [X,X[::-1,::-1]]:
     expected=reference.fit_kmeans(layout,C,max_iter=5,block_rows=13)
     for cap in [8*k,32*2**20]:
      actual=candidate.fit_kmeans(layout,C,max_iter=5,backend='scipy',finalizer='scipy',policy=candidate.ExecutionPolicy(threads=1,block_rows=13,max_scratch_bytes=cap))
      for key in ['centers','labels','core_labels','inertia','n_iter']:np.testing.assert_array_equal(actual[key],expected[key])
 def test_no_sklearn_or_numba_import_required(self):
  code='''import importlib.abc,sys\nclass Block(importlib.abc.MetaPathFinder):\n def find_spec(self,name,path=None,target=None):\n  if name.split('.')[0] in ('sklearn','numba'):raise ImportError('blocked for portable-path test')\nsys.meta_path.insert(0,Block())\nimport numpy as np\nfrom ubukit._impl.portable_accel import fit_kmeans,ExecutionPolicy\nx=np.arange(120,dtype=np.float64).reshape(30,4)/10\nr=fit_kmeans(x,x[:3].copy(),backend='scipy',finalizer='scipy',max_iter=3,policy=ExecutionPolicy(threads=1))\nassert len(r['labels'])==30 and np.isfinite(r['inertia'])\nassert not any(k=='sklearn' or k.startswith('sklearn.') or k=='numba' or k.startswith('numba.') for k in sys.modules)\nprint('portable scipy path succeeded without sklearn/Numba imports')'''
  env={**os.environ,'NUMBA_DISABLE_JIT':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
  result=subprocess.run([sys.executable,'-I','-B','-c',code],env=env,text=True,capture_output=True,check=True);self.assertIn('succeeded',result.stdout)
if __name__=='__main__':unittest.main(verbosity=2)
