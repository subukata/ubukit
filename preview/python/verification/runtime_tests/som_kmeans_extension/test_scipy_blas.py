import sys,unittest,os
from pathlib import Path
import portable_accel as candidate
from .._support import portable_oracle as checkpoint, kmeans_reference as reference
import numpy as np
from unittest.mock import patch

_REVIEWED_OVERFLOW_CASES = [
 (1,3,4,'0x1.279a74590331cp+510'),
 (32,129,256,'0x1.68a1f80d71812p+507'),
 (32,784,64,'0x1.2492492492486p+506'),
]

class ScipyBlas(unittest.TestCase):
 def compare(self,X,C,iters=4,cap=32*2**20):
  funcs=[lambda:checkpoint.fit_kmeans(X,C,max_iter=iters,backend='numpy',finalize=False,policy=checkpoint.ExecutionPolicy(threads=1,max_scratch_bytes=max(cap,16*len(C)),block_rows=13)),lambda:candidate.fit_kmeans(X,C,max_iter=iters,backend='scipy_blas',finalize=False,policy=candidate.ExecutionPolicy(threads=1,max_scratch_bytes=cap,block_rows=13))]
  values=[]
  for fn in funcs:
   try:values.append(fn())
   except Exception as e:values.append(e)
  if isinstance(values[0],Exception):self.assertIsInstance(values[1],type(values[0]));return
  self.assertIsInstance(values[1],dict)
  for field in ['centers','labels','core_labels','n_iter']:np.testing.assert_array_equal(values[1][field],values[0][field],err_msg=field)
  np.testing.assert_array_equal(values[1]['centers'].view(np.uint64),values[0]['centers'].view(np.uint64))
  self.assertLessEqual(values[1]['primary_scratch_budgeted_bytes'],cap)
 def test_random_shapes_scales_offsets(self):
  rng=np.random.default_rng(971)
  for trial in range(120):
   n=int(rng.integers(1,80));d=[1,2,8,17,32,128][trial%6];k=int(rng.integers(1,20))
   for scale,offset in [(1.,0.),(1e-160,0.),(1e-8,1e8),(1.,1e12),(1e136,1e150)]:
    X=np.ascontiguousarray(rng.normal(size=(n,d))*scale+offset);C=np.ascontiguousarray(rng.normal(size=(k,d))*scale+offset)
    if trial%5==0:C[1:]=C[0]
    self.compare(X,C)
 def test_near_ties_and_degenerate_inputs(self):
  rng=np.random.default_rng(973)
  for d in [1,2,8,32,128,784]:
   for offset in [0.,1e8,1e12]:
    a=rng.normal(size=d)+offset;b=np.nextafter(a,np.inf);mid=(a+b)/2
    X=np.tile(mid,(31,1));X[::2]=np.nextafter(X[::2],np.inf);X[1::2]=np.nextafter(X[1::2],-np.inf);self.compare(X,np.stack([a,b,mid]))
   for value in [0.,-0.,1.,1e150,1e200]:self.compare(np.full((17,d),value),np.full((9,d),value))
 def test_layout_readonly_and_input_updates(self):
  rng=np.random.default_rng(975)
  for dtype in [np.float32,np.float64,np.int32]:
   X=(rng.normal(size=(37,64))*10).astype(dtype)[::-1,::-1];C=X[:7].copy();self.compare(X,C)
   X[0,0]+=1;self.compare(X,C);X.setflags(write=False);self.compare(X,C)
  raw=np.zeros(1+37*32*8,dtype=np.uint8);X=np.ndarray((37,32),dtype=np.float64,buffer=raw,offset=1);X[:]=rng.normal(size=X.shape);self.compare(X,X[:7].copy())
 def test_budget_edges_and_no_jit(self):
  rng=np.random.default_rng(977);X=rng.normal(size=(129,32));C=X[:7].copy()
  for cap in [8*7,16*7,1000,10000,100000,32*2**20]:self.compare(X,C,cap=cap)
  with patch.dict(os.environ,{'NUMBA_DISABLE_JIT':'1'}):
   expected=checkpoint.fit_kmeans(X,C,max_iter=5,backend='numpy')
   actual=candidate.fit_kmeans(X,C,max_iter=5,backend='scipy_blas')
   for key in ['centers','labels','core_labels','inertia','n_iter']:np.testing.assert_array_equal(actual[key],expected[key])
 def test_prepared_snapshots_and_dtype(self):
  rng=np.random.default_rng(979);X=rng.normal(size=(129,32)).astype(np.float32);C=X[:7].copy()
  a=candidate.prepare(X);before=candidate.fit_kmeans(a,C,max_iter=5,backend='scipy_blas',finalize=False);X[:]=0
  after=candidate.fit_kmeans(a,C,max_iter=5,backend='scipy_blas',finalize=False)
  self.assertEqual(after['centers'].dtype,np.dtype('float64'))
  for key in ['centers','labels','core_labels','n_iter']:np.testing.assert_array_equal(before[key],after[key])
 def test_reviewed_overflow_regressions(self):
  known_errors = {
   'squared distance range may overflow; rescale input',
   'squared distances overflowed; rescale input',
  }
  for n,d,k,hexvalue in _REVIEWED_OVERFLOW_CASES:
   b=float.fromhex(hexvalue);X=np.full((n,d),b);C=np.zeros((k,d));C[1]=X[0]*.1;C[2]=X[0]*.2;C[-1]=-X[0]
   expected_message = None
   for name,module,backend in [('oracle',checkpoint,'numpy'),('oracle',checkpoint,'scipy'),('candidate',candidate,'scipy_blas')]:
    with self.subTest(shape=(n,d,k), value=hexvalue, backend=f'{name}.{backend}'):
     with np.errstate(over='ignore',invalid='ignore'), self.assertRaises(ValueError) as rejected:
      module.fit_kmeans(X,C,max_iter=1,finalize=False,backend=backend,policy=module.ExecutionPolicy(threads=1,block_rows=32))
     message = str(rejected.exception)
     self.assertIn(message, known_errors)
     if expected_message is None:
      expected_message = message
     else:
      self.assertEqual(message, expected_message)

 def test_reviewed_nonwinning_overflow_kernel_guards(self):
  from importlib import import_module
  from portable_accel._backends import blas_certificate, kmeans_scipy_blas
  oracle_core = import_module(checkpoint.__name__ + '._kmeans_lagged')
  for n,d,k,hexvalue in _REVIEWED_OVERFLOW_CASES:
   b=float.fromhex(hexvalue);X=np.full((n,d),b);C=np.zeros((k,d));C[1]=X[0]*.1;C[2]=X[0]*.2;C[-1]=-X[0]
   with self.subTest(shape=(n,d,k), value=hexvalue, backend='ordered-distance-fixture'):
    distances = np.zeros((n,k))
    with np.errstate(over='ignore',invalid='ignore'):
     for feature in range(d):
      distances += np.square(X[:,feature,None] - C[None,:,feature])
    self.assertTrue(np.isfinite(distances[:,:-1]).all())
    self.assertTrue(np.isposinf(distances[:,-1]).all())
   for backend,kernel in [('oracle.numpy',oracle_core._numpy_lloyd),('oracle.scipy',oracle_core._scipy_lloyd)]:
    with self.subTest(shape=(n,d,k), value=hexvalue, backend=backend):
     policy = checkpoint.ExecutionPolicy(threads=1,block_rows=32)
     with policy.activate(), np.errstate(over='ignore',invalid='ignore'), self.assertRaises(ValueError) as rejected:
      kernel(X,C,1,policy)
     self.assertEqual(str(rejected.exception), 'squared distances overflowed; rescale input')
   policy = candidate.ExecutionPolicy(threads=1,block_rows=32)
   with self.subTest(shape=(n,d,k), value=hexvalue, backend='candidate.scipy_blas.default'):
    certificate_expected = blas_certificate._blas_certificate_supported(X,C)
    with patch.object(kmeans_scipy_blas,'_distance_bounds',wraps=kmeans_scipy_blas._distance_bounds) as bounds:
     with policy.activate(), np.errstate(over='ignore',invalid='ignore'), self.assertRaises(ValueError) as rejected:
      kmeans_scipy_blas.run(X,C,1,policy)
     self.assertEqual(str(rejected.exception), 'squared distances overflowed; rescale input')
     self.assertEqual(bool(bounds.call_count), certificate_expected)
   with self.subTest(shape=(n,d,k), value=hexvalue, backend='candidate.scipy_blas.unknown_build'):
    with patch.object(blas_certificate,'_BLAS_CERTIFICATE_BUILD',False), patch.object(kmeans_scipy_blas,'_distance_bounds',wraps=kmeans_scipy_blas._distance_bounds) as bounds:
     with policy.activate(), np.errstate(over='ignore',invalid='ignore'), self.assertRaises(ValueError) as rejected:
      kmeans_scipy_blas.run(X,C,1,policy)
     self.assertEqual(str(rejected.exception), 'squared distances overflowed; rescale input')
     bounds.assert_not_called()
 def test_eligible_route_is_exercised_and_unknown_build_falls_back(self):
  from portable_accel._backends import blas_certificate
  rng=np.random.default_rng(981);X=np.ascontiguousarray(rng.normal(size=(256,784)));C=X[:16].copy()
  actual=candidate.fit_kmeans(X,C,max_iter=1,finalize=False,backend='scipy_blas',policy=candidate.ExecutionPolicy(threads=1,block_rows=128))
  if blas_certificate._blas_certificate_supported(X[:128],C):
   self.assertTrue(actual['certificate_enabled']);self.assertGreater(actual['blas_attempted_rows'],0);self.assertLess(actual['guard_fallbacks'],len(X))
  else:self.assertFalse(actual['certificate_enabled'])
  with patch.object(blas_certificate,'_BLAS_CERTIFICATE_BUILD',False):
   disabled=candidate.fit_kmeans(X,C,max_iter=1,finalize=False,backend='scipy_blas',policy=candidate.ExecutionPolicy(threads=1,block_rows=128))
  self.assertFalse(disabled['certificate_enabled']);self.assertEqual(disabled['blas_attempted_rows'],0)
  for key in ['centers','labels','n_iter']:np.testing.assert_array_equal(disabled[key],actual[key])
if __name__=='__main__':unittest.main(verbosity=2)
