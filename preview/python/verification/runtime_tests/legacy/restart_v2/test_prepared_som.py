import unittest
from unittest.mock import patch
import numpy as np
from portable_accel import ExecutionPolicy, fit_som_olp, PreparedSOM

class PreparedSOMTests(unittest.TestCase):
    def test_fresh_fit_equivalence_and_one_svd(self):
        rng=np.random.default_rng(924)
        R=rng.normal(size=(9,2))
        policy=ExecutionPolicy(threads=1,block_rows=8,max_scratch_bytes=64<<20)
        for kind in ('normal','duplicates','offset','zero','float32','uint8'):
            X=rng.normal(size=(23,5))
            if kind=='duplicates':X[1:]=X[0]
            if kind=='offset':X+=1e8
            if kind=='zero':X[:]=0
            if kind=='float32':X=X.astype(np.float32)
            if kind=='uint8':X=np.clip(X*10+40,0,255).astype(np.uint8)
            original=X.copy()
            with patch('numpy.linalg.svd',wraps=np.linalg.svd) as svd:
                prepared=PreparedSOM(X,max_rank=2,threads=1)
                outputs=[]
                for gamma,lam,scale in ((1.2,.3,2.),(2.4,.6,1.5),(.2,.001,2.)):
                    outputs.append(prepared.fit(R,gamma=gamma,lam=lam,pca_scale=scale,
                                                max_iters=7,tol=0.,policy=policy))
                self.assertEqual(svd.call_count,1)
            for out,(gamma,lam,scale) in zip(outputs,((1.2,.3,2.),(2.4,.6,1.5),(.2,.001,2.))):
                ref=fit_som_olp(original,R,gamma=gamma,lam=lam,pca_scale=scale,
                               max_iters=7,tol=0.,backend='threadpool',initializer='svd_lowrank',policy=policy)
                self.assertEqual(out['n_iter'],ref['n_iter'])
                for key in ('W','P','V','history'):
                    np.testing.assert_array_equal(out[key],ref[key])
            before=prepared.initialize(R,.6);X[:]=123.;after=prepared.initialize(R,.6)
            for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
            for array in (prepared._X,prepared._mean,prepared._projection,prepared._basis,prepared._singular):
                with self.assertRaises(ValueError):array.setflags(write=True)
            with self.assertRaises(AttributeError):prepared._rank=1
            self.assertEqual(prepared.describe()['preparation_threads'],1)

    def test_identity_and_contract_validation(self):
        X=np.full((7,3),2**60,dtype=np.int64)
        prepared=PreparedSOM(X,max_rank=2)
        prepared.assert_same_input(X)
        changed=X.copy();changed[0,0]+=1
        for bad in (changed,X.astype(float),X.reshape(3,7)):
            with self.assertRaises(ValueError):prepared.assert_same_input(bad)
        with self.assertRaises(ValueError):prepared.initialize(np.ones((5,3)),.5)
        for bad in (np.ones((3,4),dtype=bool),np.ones((3,4),dtype=complex),np.full((3,4),'1')):
            with self.assertRaises(TypeError):PreparedSOM(bad)

if __name__=='__main__':unittest.main()
