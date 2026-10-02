import json,subprocess,sys,unittest
import numpy as np
from ubukit._impl.portable_accel import ExecutionPolicy,prepare,fit_kmeans,joint_quality,fit_som_olp,initialize_som_olp,run_som_olp
class Foundation(unittest.TestCase):
    def test_lazy(self):
        s=subprocess.check_output([sys.executable,'-c',"import sys,json;import ubukit._impl.portable_accel;print(json.dumps([n for n in ('numpy','scipy','sklearn','numba','llvmlite') if n in sys.modules]))"],text=True)
        self.assertEqual(json.loads(s),[])
    def test_snapshot(self):
        x=np.arange(30.).reshape(10,3);p=prepare(x);x[:]=0;self.assertEqual(p.X[1,0],3.)
        with self.assertRaises(ValueError):p.X.setflags(write=True)
        original=p.norms();self.assertFalse(np.array_equal(original,p.norms(preprocessing='mean-centered')))
        again=p.norms();self.assertIsNot(original,again);self.assertTrue(np.shares_memory(original,again));self.assertEqual(p.cache_info()['hits'],1)
        expected=again.copy();original.shape=(2,5);original.dtype=np.int64
        np.testing.assert_array_equal(p.norms(),expected);self.assertEqual(p.norms().shape,(10,))
    def test_kmeans_core(self):
        rng=np.random.default_rng(7)
        for x,init in [(rng.normal(size=(17,4)),rng.normal(size=(5,4))),(np.zeros((9,2)),np.array([[0.,0.],[0.,0.],[20.,20.]]))]:
            centers=init.copy();labels=np.full(len(x),-1)
            for it in range(1,6):
                assigned=((x[:,None]-centers)**2).sum(axis=2).argmin(axis=1);updated=centers.copy()
                for k in range(len(centers)):
                    if (assigned==k).any():updated[k]=x[assigned==k].mean(axis=0)
                same=np.array_equal(assigned,labels);labels=assigned;centers=updated
                if same:break
            r=fit_kmeans(x,init,max_iter=5,backend='numpy',finalize=False)
            np.testing.assert_array_equal(r['labels'],labels);np.testing.assert_allclose(r['centers'],centers,atol=1e-12,rtol=1e-12);self.assertEqual(r['n_iter'],it);self.assertIsNone(r['inertia'])
    def test_kmeans_final_sklearn(self):
        from sklearn.cluster import KMeans
        rng=np.random.default_rng(3);x=np.r_[rng.normal(-8,.2,(12,3)),rng.normal(0,.2,(12,3)),rng.normal(8,.2,(12,3))];init=x[[0,12,24]]
        with ExecutionPolicy().activate():expected=KMeans(n_clusters=3,init=init,n_init=1,max_iter=20,tol=0,algorithm='lloyd').fit(x)
        r=fit_kmeans(x,init,backend='numpy');self.assertTrue(r['finalized_labels']);np.testing.assert_array_equal(r['labels'],expected.labels_);np.testing.assert_allclose(r['centers'],expected.cluster_centers_,atol=1e-12);self.assertAlmostEqual(r['inertia'],expected.inertia_,places=10)
        p=fit_kmeans(prepare(x),init,backend='numpy');np.testing.assert_array_equal(p['labels'],r['labels'])
    def test_metrics_exact(self):
        from sklearn.manifold import trustworthiness
        rng=np.random.default_rng(4)
        for dtype in (np.float32,np.float64):
            for x in (rng.normal(size=(19,5)).astype(dtype),np.repeat(np.arange(7,dtype=dtype),3)[:,None]):
                y=rng.normal(size=(len(x),2)).astype(dtype)
                with ExecutionPolicy().activate():ref=[(trustworthiness(x,y,n_neighbors=k),trustworthiness(y,x,n_neighbors=k)) for k in (2,4)]
                got=joint_quality(x,y,[2,4,2],policy=ExecutionPolicy(block_rows=3));self.assertEqual([(q.trustworthiness,q.continuity) for q in got],ref)
    def test_som_reference(self):
        from ubukit._impl.portable_accel._backends.som_reference import fit
        x=np.random.default_rng(3).normal(size=(17,5));r=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        got=fit_som_olp(x,r,gamma=.5,lam=1.,max_iters=3);ref=fit(x,r,.5,1.,max_iters=3,threads=1)
        for key in ('W','P','V','history'):np.testing.assert_array_equal(got[key],ref[key])
        w,p=initialize_som_olp(x,r,1.);zero=run_som_olp(x,r,w,p,gamma=.5,lam=1.,max_iters=0);self.assertIsNone(zero['V']);self.assertEqual(zero['n_iter'],0)
    def test_validation_restoration(self):
        from threadpoolctl import threadpool_info
        for x in (np.empty((0,2)),np.array([[np.nan]]),np.array([[1j]])):
            with self.assertRaises((ValueError,TypeError)):prepare(x)
        with self.assertRaises(ValueError):ExecutionPolicy(threads=True)
        with self.assertRaises(ValueError):joint_quality(np.eye(5),np.eye(5),2,max_distance_bytes=2)
        before=[(p['filepath'],p['num_threads']) for p in threadpool_info()]
        with self.assertRaises(RuntimeError):
            with ExecutionPolicy().activate():raise RuntimeError()
        self.assertEqual(before,[(p['filepath'],p['num_threads']) for p in threadpool_info()])
if __name__=='__main__':unittest.main()
