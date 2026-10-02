import unittest
from unittest.mock import patch
import importlib.util
import json
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import ubukit._impl.rough_cmeans as r

BACKENDS = ['naive', 'numpy'] + [b for b in ['scipy', 'numba'] if importlib.util.find_spec(b)]


class ExRCMTests(unittest.TestCase):
    def test_literal_formula(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(23, 4)); C = X[[0, 4, 8]]
        dist = np.sqrt(((X[:, None] - C) ** 2).sum(axis=2))
        for p in [.3, .5, 1, 2, 3, 16, 128]:
            expected = dist ** p <= 1.15 ** p * dist.min(axis=1)[:, None] ** p + .7 ** p
            for b in BACKENDS:
                U, M = r.assign(X, C, alpha=1.15, beta=.7, p=p, backend=b)
                assert_array_equal(M, expected.T)
                assert_array_equal(U, (expected / expected.sum(axis=1)[:, None]).T)
                assert_allclose(U.sum(axis=0), 1., rtol=0, atol=2e-16)

    def test_hand_computed_overlap_and_center(self):
        X = np.array([[0.], [2.], [4.]])
        C = np.array([[0.], [4.]])
        for b in BACKENDS:
            U, M = r.assign(X, C, p=1, alpha=1, beta=0, backend=b)
            assert_array_equal(U, [[1, .5, 0], [0, .5, 1]])
            out = r.fit(X, 2, init=C, alpha=1, beta=0, p=1, max_iter=1, backend=b)
            assert_allclose(out.centers, [[2/3], [10/3]], rtol=1e-15)
            assert_array_equal(out.memberships, U)
            self.assertTrue(out.fixed_point)

    def test_duplicate_zero_ties(self):
        X = np.array([[0., 0.], [0, 0], [2, 2.]])
        C = np.array([[0., 0.], [0, 0], [2, 2.]])
        for p in [.001, .5, 1, 2, 1000]:
            for b in BACKENDS:
                U, M = r.assign(X, C, alpha=1, beta=0, p=p, backend=b)
                assert_array_equal(U, [[.5, .5, 0], [.5, .5, 0], [0, 0, 1]])

    def test_empty_clusters_retain_center(self):
        X = np.array([[0.], [1.], [2.]])
        C = np.array([[1.], [100.]])
        for b in BACKENDS:
            q = r.fit(X, 2, init=C, alpha=1, beta=0, backend=b)
            self.assertEqual(q.centers[1, 0], 100.)
            self.assertGreater(q.empty_cluster_updates, 0)
            self.assertTrue(q.converged)

    def test_seed_and_no_input_mutation(self):
        X = np.random.default_rng(5).normal(size=(31, 4)); before = X.copy()
        a = r.fit(X, 5, seed=4, backend='numpy')
        b = r.fit(X, 5, seed=4, backend='numpy')
        assert_array_equal(a.init_indices, b.init_indices)
        self.assertEqual(len(set(a.init_indices)), 5)
        assert_array_equal(a.centers, b.centers)
        assert_array_equal(X, before)
        C = X[:5].copy(); prev = C.copy()
        r.fit(X, 5, init=C)
        assert_array_equal(C, prev)

    def test_backends_fit_fixed_init(self):
        rng = np.random.default_rng(10)
        X = np.r_[rng.normal(-4, .2, size=(41, 3)), rng.normal(4, .3, size=(41, 3))]
        C = X[[0, 41, 70]]
        for p in [.5, 1, 2, 3, 50, 1000]:
            ref = r.fit(X, 3, init=C, alpha=1.13, beta=.25, p=p, backend='naive')
            for b in BACKENDS:
                q = r.fit(X, 3, init=C, alpha=1.13, beta=.25, p=p, backend=b, block_size=11)
                assert_array_equal(q.upper_memberships, ref.upper_memberships)
                assert_array_equal(q.memberships, ref.memberships)
                assert_allclose(q.centers, ref.centers, atol=8e-14, rtol=5e-14)
                self.assertEqual(q.stop_reason, ref.stop_reason)

    def test_no_memberships_identical(self):
        X = np.random.default_rng(11).normal(size=(40, 3))
        for b in BACKENDS:
            a = r.fit(X, 4, backend=b, return_memberships=True)
            q = r.fit(X, 4, backend=b, return_memberships=False)
            assert_array_equal(q.centers, a.centers)
            self.assertIsNone(q.memberships)
            self.assertIsNone(q.upper_memberships)
            self.assertEqual(q.n_iter, a.n_iter)

    def test_maxiter_final_membership_alignment(self):
        X = np.random.default_rng(42).normal(size=(80, 2))
        for b in BACKENDS:
            q = r.fit(X, 3, init=X[:3], max_iter=1, alpha=1.1, beta=.4, p=3, backend=b)
            U, M = r.assign(X, q.centers, alpha=1.1, beta=.4, p=3, backend=b)
            assert_array_equal(q.memberships, U)
            assert_array_equal(q.upper_memberships, M)
            self.assertEqual(q.stop_reason, 'max_iter')
            self.assertFalse(q.converged)

    def test_large_common_offset(self):
        X = 1e12 + np.array([[0., 0], [.25, .75], [1., 1.], [10., 10.], [11., 10.], [11., 11.]])
        C = X[[0, 5]]
        U0, M0 = r.assign(X - X[0], C - X[0], alpha=1.2, beta=.5, p=3)
        for b in BACKENDS:
            U, M = r.assign(X, C, alpha=1.2, beta=.5, p=3, backend=b)
            assert_array_equal(U, U0); assert_array_equal(M, M0)
            q = r.fit(X, 2, init=C, alpha=1.2, beta=.5, p=3, backend=b)
            self.assertTrue(np.isfinite(q.centers).all())

    def test_power_overflow_avoided(self):
        X = np.array([[0.], [2.], [10.]])
        C = np.array([[0.], [10.]])
        for p in [1e-5, .3, 2, 1000, 1e6]:
            for b in BACKENDS:
                U, M = r.assign(X, C, alpha=1.2, beta=3., p=p, backend=b)
                self.assertTrue(np.isfinite(U).all())
                self.assertTrue((M.sum(axis=0) > 0).all())
        for b in BACKENDS:
            U, M = r.assign(X, C, alpha=1e300, beta=0, p=2, backend=b)
            assert_array_equal(M[:, 1], [True, True])

    def test_ulp_threshold_boundaries(self):
        # The optimized general-p threshold must match powered canonical rule,
        # including values on both sides within a handful of representable steps.
        for p in [.25, .3, .5, 1, 2, 3, 8, 64]:
            a, beta = 1.2, .7
            t = (a ** p + beta ** p) ** (1 / p)
            d = [t]
            lo = hi = t
            for _ in range(30):
                lo = np.nextafter(lo, -np.inf); hi = np.nextafter(hi, np.inf)
                d.extend([lo, hi])
            sq = np.array([[1., v * v] for v in d])
            fast = r._mask_from_sq(sq, a, beta, p)
            ref = r._mask_from_sq(sq, a, beta, p, optimized=False)
            assert_array_equal(fast, ref)

    def test_invalid_parameters(self):
        X = np.ones((3, 2))
        for kw in [dict(alpha=.9), dict(alpha=np.inf), dict(beta=-.1), dict(p=0), dict(p=np.nan), dict(p=np.inf), dict(max_iter=0), dict(block_size=0), dict(cycle_window=-1), dict(backend='missing')]:
            with self.assertRaises(ValueError): r.fit(X, 2, **kw)
        for bad in [np.ones(3), np.empty((0, 2)), [[np.nan]], [[np.inf]]]:
            with self.assertRaises(ValueError): r.fit(bad, 1)
        with self.assertRaises(ValueError): r.fit(X.astype(complex), 2)
        with self.assertRaises(ValueError): r.assign(X, X[:2].astype(complex))
        with self.assertRaises(ValueError): r.fit(X, 4)
        with self.assertRaises(ValueError): r.fit(X, 2, init=np.ones((2, 3)))
        with self.assertRaises(FloatingPointError): r.assign([[1e300], [-1e300]], [[0.]], p=1)

    def test_tiny_distance_handling(self):
        # Near-zero but normal squares work; unreliably represented squares fail
        # explicitly instead of changing nonzero distances to zero memberships.
        for b in BACKENDS:
            U, M = r.assign([[0.], [1e-140], [3e-140]], [[0.], [3e-140]], alpha=1.1, beta=0, p=1000, backend=b)
            assert_array_equal(M, [[True, True, False], [False, False, True]])
            for scale in [1e-160, 1e-200]:
                with self.assertRaises(FloatingPointError):
                    r.assign([[0.], [scale]], [[0.]], backend=b)

    def test_more_centers_than_queries(self):
        for b in BACKENDS:
            U, M = r.assign([[0.]], [[1.], [2.]], alpha=1, beta=0, backend=b)
            assert_array_equal(M, [[True], [False]])

    def test_extreme_p_algebra_and_ratio_underflow(self):
        for b in BACKENDS:
            for p in [1e-17, 1e-300]:
                U, M = r.assign([[0.]], [[1.], [2.]], alpha=1, beta=0, p=p, backend=b)
                assert_array_equal(M, [[True], [False]])
            U, M = r.assign([[0.]], [[1e10], [1e154]], alpha=1, beta=1e-320, p=.001, backend=b)
            ld = np.longdouble
            dist = np.array([1e10, 1e154], dtype=ld)
            literal = dist ** ld(.001) <= dist.min() ** ld(.001) + ld(1e-320) ** ld(.001)
            assert_array_equal(M[:, 0], literal)
            assert_array_equal(M, [[True], [True]])
            U, M = r.assign([[0.]], [[2000.], [3.0223505125121568e47]], alpha=1, beta=1e-320, p=.002, backend=b)
            assert_array_equal(M, [[True], [True]])
            # Forming d/scale can overflow even while its tiny power is small.
            U, M = r.assign([[0.]], [[1e-140], [1e150]], alpha=1, beta=1e-140, p=.0001, backend=b)
            assert_array_equal(M, [[True], [True]])

    def test_cycle_requires_center_state(self):
        A = np.array([[1], [1]], dtype=np.uint8)
        B = np.array([[2], [3]], dtype=np.uint8)
        def step(centers, mask):
            return np.array(centers, float), 0, mask, None, None
        for true_cycle in [False, True]:
            third = [[0.], [5.]] if true_cycle else [[0.], [8.]]
            states = [step([[0.], [5.]], A), step([[3.], [8.]], B),
                      step(third, A), step([[2.], [9.]], B)]
            with patch('ubukit._impl.rough_cmeans._step', side_effect=states):
                q = r.fit([[0.], [2.]], 2, init=[[-1.], [-1.]], max_iter=3, backend='numpy')
            self.assertEqual(q.stop_reason, 'cycle' if true_cycle else 'max_iter')
            self.assertEqual(q.cycle_length, 2 if true_cycle else None)
            self.assertFalse(q.converged)

    def test_rcm_alias(self):
        X = np.arange(15.).reshape(5, 3)
        a = r.fit_rcm(X, 2); b = r.fit_exrcm(X, 2, p=1)
        assert_array_equal(a.centers, b.centers)
        with self.assertRaises(ValueError): r.fit_rcm(X, 2, p=2)


if __name__ == '__main__': unittest.main(verbosity=2)
