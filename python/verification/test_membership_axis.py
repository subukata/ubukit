"""Public facade contract tests against installed wheel/sdist, never source imports."""
from dataclasses import fields, replace
import importlib.util
import inspect
import os
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import ubukit._impl.rough_cmeans
import ubukit


class MembershipAxisTests(unittest.TestCase):
    def assert_owned(self, value, dtype, shape):
        self.assertEqual(value.shape, shape)
        self.assertEqual(value.dtype, dtype)
        self.assertTrue(value.flags.c_contiguous)
        self.assertTrue(value.flags.owndata)
        self.assertTrue(value.flags.writeable)
        self.assertIsNone(value.base)

    def assert_metadata_equal(self, actual, expected, *, identity=False):
        for field in fields(expected):
            if field.name in ("memberships", "upper_memberships"):
                continue
            left, right = getattr(actual, field.name), getattr(expected, field.name)
            if isinstance(right, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                self.assertEqual(left, right, field.name)
            if identity:
                self.assertIs(left, right, field.name)
            self.assertIn(field.name, dir(actual))

    def test_export_identity_and_signatures(self):
        self.assertEqual(len(ubukit.__all__), 38)
        self.assertEqual(set(ubukit._ADAPTED_EXPORTS), {"fit_rcm", "fit_exrcm", "assign_rcm"})
        for name, (module_name, attribute) in ubukit._EXPORTS.items():
            legacy = getattr(__import__('importlib').import_module(module_name, 'ubukit'), attribute)
            actual = getattr(ubukit, name)
            with self.subTest(name=name):
                if name in ubukit._ADAPTED_EXPORTS:
                    self.assertIsNot(actual, legacy)
                    self.assertEqual(inspect.signature(actual), inspect.signature(legacy))
                else:
                    self.assertIs(actual, legacy)

    def test_actual_fit_and_assignment_all_backends(self):
        backends = ["naive", "numpy", "scipy", "auto"]
        if os.environ.get("UBUKIT_TEST_NUMBA") == "1":
            self.assertIsNotNone(importlib.util.find_spec("numba"))
            backends.append("numba")
        # A square, asymmetric membership fixture catches shape-only heuristics.
        fixtures = [
            (np.array([[0., 0.], [0., 1.], [1., 0.], [8., 8.], [8., 9.], [9., 8.]]),
             np.array([[0., 0.], [8., 8.]])),
            (np.array([[0.], [1.]]), np.array([[0.], [10.]])),
            (np.array([[2.]]), np.array([[2.]])),
            (np.array([[0.], [1.], [2.]]), np.array([[0.]])),
        ]
        for X, init in fixtures:
            for name in ("fit_rcm", "fit_exrcm"):
                for backend in backends:
                    for memberships in (True, False):
                        with self.subTest(shape=(len(X), len(init)), name=name,
                                          backend=backend, memberships=memberships):
                            original_X, original_init = X.copy(), init.copy()
                            kwargs = dict(init=init, backend=backend, max_iter=2,
                                          return_memberships=memberships, block_size=2)
                            if name == "fit_exrcm":
                                kwargs["p"] = 2.5
                            legacy = getattr(ubukit._impl.rough_cmeans, name)(X, len(init), **kwargs)
                            actual = getattr(ubukit, name)(X, len(init), **kwargs)
                            self.assertIs(type(legacy), ubukit._impl.rough_cmeans.RoughCMeansResult)
                            self.assert_metadata_equal(actual, legacy)
                            np.testing.assert_array_equal(X, original_X)
                            np.testing.assert_array_equal(init, original_init)
                            if not memberships:
                                self.assertIsNone(actual.memberships)
                                self.assertIsNone(actual.upper_memberships)
                                continue
                            shape = (len(X), len(init))
                            self.assertEqual(legacy.memberships.shape, shape[::-1])
                            self.assert_owned(actual.memberships, np.float64, shape)
                            self.assert_owned(actual.upper_memberships, np.bool_, shape)
                            np.testing.assert_array_equal(actual.memberships, legacy.memberships.T)
                            np.testing.assert_array_equal(actual.upper_memberships, legacy.upper_memberships.T)
                            np.testing.assert_allclose(actual.memberships.sum(axis=1), 1.)
                            if shape == (2, 2):
                                self.assertFalse(np.array_equal(legacy.memberships, legacy.memberships.T))
                            assigned = ubukit.assign_rcm(X, actual.centers, alpha=actual.alpha,
                                beta=actual.beta, p=actual.p, backend=backend, block_size=2)
                            np.testing.assert_array_equal(assigned[0], actual.memberships)
                            np.testing.assert_array_equal(assigned[1], actual.upper_memberships)
                            legacy_assigned = ubukit._impl.rough_cmeans.assign(X, actual.centers,
                                alpha=actual.alpha, beta=actual.beta, p=actual.p,
                                backend=backend, block_size=2)
                            for current, prior in zip(assigned, legacy_assigned):
                                self.assert_owned(current, prior.dtype, shape)
                                np.testing.assert_array_equal(current, prior.T)

    def test_copy_ownership_and_no_legacy_result_mutation(self):
        X = np.array([[0.], [1.], [10.]])
        template = ubukit._impl.rough_cmeans.fit_rcm(X, 3, backend="numpy", max_iter=1)
        # Covers N == K and both contiguous orders, transposed views and negative strides.
        raw = np.arange(9, dtype=np.float32).reshape(3, 3) / 10
        for values in (raw, np.asfortranarray(raw), raw.T, raw[::-1, ::-1]):
            upper = values > .3
            for name in ("fit_rcm", "fit_exrcm"):
                with self.subTest(name=name, strides=values.strides):
                    legacy = replace(template, memberships=values, upper_memberships=upper)
                    saved_values, saved_upper = values.copy(), upper.copy()
                    with patch.object(ubukit._impl.rough_cmeans, name, return_value=legacy) as call:
                        actual = getattr(ubukit, name)(X, 3)
                        call.assert_called_once()
                    self.assert_metadata_equal(actual, legacy, identity=True)
                    self.assert_owned(actual.memberships, values.dtype, (3, 3))
                    self.assert_owned(actual.upper_memberships, upper.dtype, (3, 3))
                    for a in (actual.memberships, actual.upper_memberships):
                        for b in (values, upper, X):
                            self.assertFalse(np.shares_memory(a, b))
                    self.assertFalse(np.shares_memory(actual.memberships, actual.upper_memberships))
                    np.testing.assert_array_equal(actual.memberships, saved_values.T)
                    actual.memberships.fill(-10)
                    actual.upper_memberships.fill(False)
                    np.testing.assert_array_equal(legacy.memberships, saved_values)
                    np.testing.assert_array_equal(legacy.upper_memberships, saved_upper)
                    self.assertIs(legacy.memberships, values)
                    self.assertIs(legacy.upper_memberships, upper)

    def test_assignment_copy_ownership(self):
        values = np.array([[1., .5, 0.], [0., .5, 1.]], dtype=np.float64)
        upper = values > 0
        saved_values, saved_upper = values.copy(), upper.copy()
        with patch.object(ubukit._impl.rough_cmeans, "assign", return_value=(values, upper)):
            actual = ubukit.assign_rcm([[1.], [2.], [3.]], [[1.], [3.]])
        self.assertIs(type(actual), tuple)
        for current, prior in zip(actual, (values, upper)):
            self.assert_owned(current, prior.dtype, (3, 2))
            self.assertFalse(np.shares_memory(current, prior))
            np.testing.assert_array_equal(current, prior.T)
        actual[0].fill(-1.)
        actual[1].fill(False)
        np.testing.assert_array_equal(values, saved_values)
        np.testing.assert_array_equal(upper, saved_upper)

    def test_noncontiguous_read_only_inputs_are_unchanged(self):
        X = np.arange(32, dtype=np.float32).reshape(8, 4)[::-1, ::2]
        init = X[[0, 4]].copy(order="F")
        X.flags.writeable = init.flags.writeable = False
        saved_X, saved_init = X.copy(), init.copy()
        for function in (ubukit.fit_rcm, ubukit.fit_exrcm):
            result = function(X, 2, init=init, backend="numpy", max_iter=2)
            ubukit.assign_rcm(X, init, backend="numpy")
            result.memberships.fill(0)
            np.testing.assert_array_equal(X, saved_X)
            np.testing.assert_array_equal(init, saved_init)
            self.assertFalse(X.flags.writeable)
            self.assertFalse(init.flags.writeable)

    def test_seed_and_nondefault_metadata_survive(self):
        X = np.array([[0.], [1.], [4.], [8.]])
        kwargs = dict(seed=31, backend="numpy", max_iter=1, alpha=1.7, beta=.3,
                      p=1.3, cycle_window=0, block_size=1)
        prior = ubukit._impl.rough_cmeans.fit_exrcm(X, 3, **kwargs)
        result = ubukit.fit_exrcm(X, 3, **kwargs)
        self.assert_metadata_equal(result, prior)
        self.assertIsNotNone(result.init_indices)
        np.testing.assert_array_equal(result.memberships, prior.memberships.T)

    def test_error_contract_preserved(self):
        for name, kwargs in (("fit_rcm", {"p": 2}), ("fit_exrcm", {"alpha": .5}),
                             ("fit_rcm", {"backend": "unknown"}),
                             ("fit_exrcm", {"n_clusters": 4})):
            with self.subTest(name=name, kwargs=kwargs):
                args = dict(n_clusters=2)
                args.update(kwargs)
                failures = []
                for module in (ubukit._impl.rough_cmeans, ubukit):
                    with self.assertRaises(ValueError) as caught:
                        getattr(module, name)([[0.], [1.]], **args)
                    failures.append((type(caught.exception), str(caught.exception)))
                self.assertEqual(failures[0], failures[1])

    def test_unknown_attribute_and_pickle(self):
        result = ubukit.fit_rcm([[0.], [1.]], 2, backend="numpy")
        with self.assertRaises(AttributeError):
            result.not_an_attribute
        restored = pickle.loads(pickle.dumps(result))
        np.testing.assert_array_equal(restored.memberships, result.memberships)
        self.assertEqual(restored.stop_reason, result.stop_reason)
        self.assertIn("membership_shape=(2, 2)", repr(restored))


if __name__ == "__main__":
    unittest.main(verbosity=2)
