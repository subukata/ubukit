import unittest
import numpy as np
from portable_accel import fit_som_olp,run_som_olp,initialize_som_olp,ExecutionPolicy
class ThreadpoolIntegration(unittest.TestCase):
    def test_public_fit(self):
        rng=np.random.default_rng(5);r=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        for offset in (0.,1e9):
            x=rng.normal(size=(31,5))+offset
            ref=fit_som_olp(x,r,gamma=.4,lam=1.2,max_iters=4,tol=0.,backend='cdist',policy=ExecutionPolicy(threads=2))
            for initializer in ('original','svd_lowrank'):
                got=fit_som_olp(x,r,gamma=.4,lam=1.2,max_iters=4,tol=0.,backend='threadpool',initializer=initializer,policy=ExecutionPolicy(threads=2,block_rows=7,max_scratch_bytes=64*2**20))
                for key in ('W','P','V','history'):np.testing.assert_allclose(got[key],ref[key],atol=1e-8,rtol=1e-8)
                self.assertEqual(got['n_iter'],ref['n_iter']);self.assertLessEqual(got['primary_scratch_budgeted_bytes'],64*2**20)
        with self.assertRaises(ValueError):fit_som_olp(x,r,gamma=.4,lam=1.2,max_iters=1,backend='threadpool',policy=ExecutionPolicy(threads=2,max_scratch_bytes=1))
if __name__=='__main__':unittest.main()
