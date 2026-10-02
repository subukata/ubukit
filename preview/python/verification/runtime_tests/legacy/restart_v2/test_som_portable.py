import unittest
import numpy as np
from ubukit._impl.portable_accel import fit_som_olp,initialize_som_olp,run_som_olp,ExecutionPolicy,prepare
class SomPortableIntegration(unittest.TestCase):
    def test_kernels_and_initialization(self):
        rng=np.random.default_rng(12);grid=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        for offset in (0.,1e9):
            x=rng.normal(size=(21,6))+offset
            for initialization in ('original','svd_lowrank'):
                w,p=initialize_som_olp(x,grid,1.2,initializer=initialization)
                reference=run_som_olp(x,grid,w,p,gamma=.5,lam=1.2,max_iters=4,tol=0.)
                for backend in ('gemm_guarded','cdist_optimized'):
                    actual=run_som_olp(prepare(x),grid,w,p,gamma=.5,lam=1.2,max_iters=4,tol=0.,backend=backend,policy=ExecutionPolicy(block_rows=3))
                    for key in ('W','P','V','history'):np.testing.assert_allclose(actual[key],reference[key],atol=1e-8,rtol=1e-8)
                    self.assertEqual(actual['n_iter'],reference['n_iter']);self.assertLessEqual(actual['primary_scratch_bytes'],32*2**20)
if __name__=='__main__':unittest.main()
