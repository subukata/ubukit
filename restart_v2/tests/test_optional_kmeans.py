import unittest
import numpy as np
from portable_accel import fit_kmeans,prepare,ExecutionPolicy
from portable_accel._optional import numba_available
@unittest.skipUnless(numba_available(),'optional Numba unavailable')
class OptionalKmeans(unittest.TestCase):
    def test_cores_and_finalization(self):
        rng=np.random.default_rng(9)
        for x in (rng.normal(size=(23,9)),np.zeros((13,3))):
            init=np.ascontiguousarray(x[:5]);base=fit_kmeans(x,init,max_iter=4,backend='numpy')
            for backend in ('numba','numba_blas','numba_blas_vector'):
                with self.subTest(backend=backend,shape=x.shape):
                    result=fit_kmeans(x,init,max_iter=4,backend=backend,policy=ExecutionPolicy(threads=1,block_rows=4))
                    np.testing.assert_array_equal(result['core_labels'],base['core_labels'])
                    np.testing.assert_allclose(result['centers'],base['centers'],atol=1e-12,rtol=1e-12)
                    np.testing.assert_array_equal(result['labels'],base['labels']);self.assertAlmostEqual(result['inertia'],base['inertia'],places=10)
                    self.assertEqual(result['n_iter'],base['n_iter'])
if __name__=='__main__':unittest.main()
