"""Public EFCM contracts and independent arithmetic references."""
import json
import math
from pathlib import Path
import unittest

import numpy as np
from ubukit import fit_entropy_fcm


class EntropyFCMTests(unittest.TestCase):
    def test_one_step_uses_linear_membership_weights(self):
        u = np.array([[.8, .2], [.2, .8]])
        r = fit_entropy_fcm([[0.], [2.]], init=u, tau=2., max_iter=1)
        np.testing.assert_allclose(r['centers'], [[.4], [1.6]], atol=1e-15)
        p = 1 / (1 + math.exp(-1.2))
        np.testing.assert_allclose(r['membership'], [[p, 1-p], [1-p, p]], atol=2e-16)

    def test_coincident_center_does_not_force_hard_membership(self):
        r = fit_entropy_fcm([[0.], [2.]], init=np.eye(2), tau=2., max_iter=1)
        p = 1 / (1 + math.exp(-2))
        np.testing.assert_allclose(r['membership'], [[p, 1-p], [1-p, p]], atol=2e-16)
        self.assertAlmostEqual(r['objective'], -4 * math.log1p(math.exp(-2)), places=14)

    def test_objective_matches_returned_pair_and_descends(self):
        x = np.array([[-2., 1.], [-1., 2.], [1., 1.], [3., -1.]])
        for tau in (.02, .7, 10.):
            r = fit_entropy_fcm(x, 3, tau=tau, random_state=123, max_iter=15, tol=0, return_history=True)
            u = r['membership']
            d2 = np.sum((x[:, None, :] - r['centers'][None, :, :])**2, axis=2)
            nz = u > 0
            expected = np.sum(u*d2) + tau*np.sum(u[nz]*np.log(u[nz]))
            self.assertAlmostEqual(r['objective'], expected, delta=2e-13*max(1, abs(expected)))
            self.assertTrue(np.all(np.diff(r['objective_history']) <= 1e-12))
            np.testing.assert_allclose(u.sum(axis=1), 1, rtol=0, atol=3e-16)
            self.assertTrue(np.isfinite(u).all() and np.all(u >= 0))

    def test_identical_data_and_empty_initial_cluster(self):
        r = fit_entropy_fcm([[3., 2.]]*3, init=[[1, 0, 0]]*3, tau=.5, max_iter=2)
        np.testing.assert_array_equal(r['centers'], [[3., 2.]]*3)
        np.testing.assert_array_equal(r['membership'], np.full((3, 3), 1/3))
        self.assertAlmostEqual(r['objective'], -1.5*math.log(3), places=14)

    def test_one_cluster_and_one_sample(self):
        r = fit_entropy_fcm([[4., -5.]], 1, tau=2., max_iter=3, tol=0)
        np.testing.assert_array_equal(r['membership'], [[1.]])
        np.testing.assert_array_equal(r['centers'], [[4., -5.]])
        self.assertEqual(r['objective'], 0)
        self.assertEqual(r['n_iter'], 3)
        self.assertFalse(r['converged'])
        self.assertEqual(fit_entropy_fcm([[4.]], 1)['n_iter'], 1)

    def test_inputs_and_outputs_are_owned(self):
        x = np.array([[0.], [2.]])
        init = np.eye(2)
        r = fit_entropy_fcm(x, init=init, max_iter=2, return_history=True)
        for out in (r['centers'], r['membership'], r['labels'], r['objective_history']):
            self.assertFalse(np.shares_memory(out, x) or np.shares_memory(out, init))
        r['centers'][:] = 90
        r['membership'][:] = 0
        np.testing.assert_array_equal(x, [[0.], [2.]])
        np.testing.assert_array_equal(init, np.eye(2))

    def test_seed_and_initial_row_scaling(self):
        x = [[0.], [1.], [5.]]
        a = fit_entropy_fcm(x, 2, random_state=7, max_iter=3)
        b = fit_entropy_fcm(x, 2, random_state=7, max_iter=3)
        np.testing.assert_array_equal(a['membership'], b['membership'])
        init = np.array([[.1, .9], [.8, .2], [.3, .7]])
        a = fit_entropy_fcm(x, init=init, max_iter=2)
        b = fit_entropy_fcm(x, init=init*np.array([1e200, 1e-200, 5])[:, None], max_iter=2)
        np.testing.assert_allclose(a['membership'], b['membership'], atol=3e-16)

    def test_temperature_validation(self):
        for tau in (0, -1, math.nan, math.inf, -math.inf, True, 1j, '1', [1.]):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                fit_entropy_fcm([[0.]], 1, tau=tau)

    def test_input_and_iteration_validation(self):
        for x in ([], [[math.nan]], [[math.inf]], [[1j]]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                fit_entropy_fcm(x, 1)
        for kw in ({'n_clusters':0}, {'max_iter':0}, {'max_iter':True}, {'tol':-1},
                   {'init':[[0., 0.]]}, {'init':[[-1., 2.]]}, {'init':[[math.inf]]},
                   {'init':[[1., 0.]], 'n_clusters':3}, {'backend':'gpu'}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                fit_entropy_fcm([[0.]], **kw)

    def test_subnormal_temperature_and_distances(self):
        tiny = np.nextafter(0., 1.)
        r = fit_entropy_fcm([[0.], [2.]], init=np.eye(2), tau=tiny, max_iter=1)
        np.testing.assert_array_equal(r['membership'], np.eye(2))
        self.assertEqual(r['objective'], 0)
        r = fit_entropy_fcm([[0.], [1e-160]], init=np.eye(2), tau=1e-320, max_iter=1)
        self.assertTrue(.7 < r['membership'][0, 0] < .75)
        self.assertTrue(np.isfinite(r['membership']).all())

    def test_extreme_finite_coordinates_and_signed_objective_status(self):
        r = fit_entropy_fcm([[-1e308], [1e308]], init=np.eye(2), tau=1e308, max_iter=1)
        np.testing.assert_array_equal(r['centers'], [[-1e308], [1e308]])
        np.testing.assert_array_equal(r['membership'], np.eye(2))
        self.assertEqual(r['objective'], 0)
        r = fit_entropy_fcm([[0.], [0.]], 8, tau=1e308, max_iter=1, random_state=0)
        self.assertEqual(r['objective'], -math.inf)
        self.assertEqual(r['numerical_diagnostics']['objective_status'], 'overflow')
        self.assertEqual(r['numerical_diagnostics']['objective_sign'], -1)
        tiny = np.nextafter(0., 1.)
        r = fit_entropy_fcm([[0.], [tiny]], 1, max_iter=1)
        self.assertEqual(r['objective'], 0)
        self.assertEqual(r['numerical_diagnostics']['objective_status'], 'underflow')

    def test_shared_decimal_reference_fixtures(self):
        path = Path(__file__).resolve().parents[2] / 'javascript/fixtures/entropy-fcm-reference.json'
        for case in json.loads(path.read_text())['cases']:
            with self.subTest(case=case['name']):
                r = fit_entropy_fcm(case['X'], init=case['init'], tau=case['tau'],
                                    max_iter=case['iterations'], tol=0, return_history=True)
                for key in ('centers', 'membership', 'objective_history'):
                    np.testing.assert_allclose(r[key], case[key], rtol=3e-13, atol=3e-14)
                self.assertAlmostEqual(r['objective'], case['objective'], delta=1e-12)

    def test_numpy_matches_reference_at_bounded_scales(self):
        rng = np.random.default_rng(812)
        for scale in (1e-160, 1e-3, 1., 1e3, 1e150):
            for n, d, k in ((2, 1, 2), (7, 3, 4), (17, 2, 3)):
                x = rng.normal(size=(n, d))*scale
                init = rng.random((n, k))
                tau = scale*scale if 1e-100 < scale < 1e100 else 1.
                kw = dict(init=init, tau=tau, max_iter=4, tol=0, return_history=True)
                a = fit_entropy_fcm(x, backend='numpy', **kw)
                b = fit_entropy_fcm(x, backend='reference', **kw)
                with self.subTest(scale=scale, shape=(n, d, k)):
                    np.testing.assert_allclose(a['membership'], b['membership'], rtol=1e-11, atol=5e-14)
                    np.testing.assert_allclose(a['centers'], b['centers'], rtol=1e-11, atol=abs(scale)*5e-14)
                    np.testing.assert_allclose(a['objective_history'], b['objective_history'], rtol=1e-11, atol=max(1., tau)*5e-13)

    def test_tiny_cluster_and_cancelling_center_are_preserved(self):
        x = [[-1e100], [1e100], [2.]]
        init = [[1., 1e-310], [1., 1e-310], [0., 1e-310]]
        for backend in ('numpy', 'reference'):
            r = fit_entropy_fcm(x, init=init, max_iter=1, backend=backend)
            self.assertEqual(r['centers'][0, 0], 0.)
            self.assertTrue(np.isfinite(r['membership']).all())

    def test_large_common_cost_retains_small_temperature_gap(self):
        x = [[-1e8, 0.], [1e8, 0.], [-1e8, 1.], [1e8, 1.]]
        init = [[1., 0.], [1., 0.], [0., 1.], [0., 1.]]
        p = 1/(1+math.exp(-1))
        expected = [[p, 1-p], [p, 1-p], [1-p, p], [1-p, p]]
        for backend in ('numpy', 'reference'):
            r = fit_entropy_fcm(x, init=init, tau=1., max_iter=1, backend=backend)
            np.testing.assert_array_equal(r['centers'], [[0., 0.], [0., 1.]])
            np.testing.assert_allclose(r['membership'], expected, rtol=0, atol=2e-16)


if __name__ == '__main__':
    unittest.main()
