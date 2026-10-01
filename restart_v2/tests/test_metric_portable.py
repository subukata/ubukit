import unittest
import numpy as np
from portable_accel import joint_quality,ExecutionPolicy
from portable_accel._optional import numba_available
class MetricPortableIntegration(unittest.TestCase):
    def test_backends_and_rank_blocks(self):
        rng=np.random.default_rng(71)
        backends=['numpy','sqrt_numpy']+(['numba','sqrt_numba'] if numba_available() else [])
        for dtype in (np.float32,np.float64):
            x=np.repeat(rng.normal(size=(7,4)).astype(dtype),2,axis=0);y=rng.normal(size=(14,2)).astype(dtype)
            reference=joint_quality(x,y,[2,4],backend='numpy')
            for backend in backends:
                result,stats=joint_quality(x,y,[2,4],backend=backend,policy=ExecutionPolicy(threads=1,block_rows=3),return_stats=True)
                self.assertEqual(result,reference);self.assertLessEqual(stats['rank_block_rows'],3)
if __name__=='__main__':unittest.main()
