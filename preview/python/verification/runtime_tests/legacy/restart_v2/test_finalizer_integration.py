import unittest
import numpy as np
from portable_accel import fit_kmeans,ExecutionPolicy
from portable_accel._optional import numba_available
@unittest.skipUnless(numba_available(),'Numba optional')
class FinalizerIntegration(unittest.TestCase):
    def test_public_fit_finalizers(self):
        x=np.random.default_rng(26).normal(size=(53,7));init=x[[0,10,20,30,40]]
        for backend in ('numpy','numba_blas_vector'):
            a=fit_kmeans(x,init,max_iter=4,backend=backend,policy=ExecutionPolicy(threads=1),finalizer='sklearn')
            b=fit_kmeans(x,init,max_iter=4,backend=backend,policy=ExecutionPolicy(threads=1),finalizer='numba')
            np.testing.assert_array_equal(a['labels'],b['labels']);np.testing.assert_array_equal(a['centers'],b['centers']);self.assertAlmostEqual(a['inertia'],b['inertia'],places=10)
            self.assertTrue(b['finalized_labels']);self.assertEqual(b['finalizer'],'numba')
        raw=fit_kmeans(x,init,max_iter=4,backend='numpy',finalize=False,finalizer='numba');self.assertIsNone(raw['inertia']);self.assertIsNone(raw['finalizer'])
if __name__=='__main__':unittest.main()
