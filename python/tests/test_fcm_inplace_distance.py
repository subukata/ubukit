"""Focused regressions for the owned NumPy FCM distance temporary."""
import unittest
from unittest.mock import patch

import numpy as np

from ubukit._impl.fcm import core


class FCMInplaceDistanceTests(unittest.TestCase):
    def assert_bits_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_distance_layouts_preserve_inputs_and_exact_bits(self):
        values = np.arange(60, dtype=np.float64).reshape(10, 6) / 8
        centers = np.arange(12, dtype=np.float64).reshape(4, 3) / 16
        readonly = values[::2, ::2].copy()
        readonly.flags.writeable = False
        layouts = {
            "c": np.array(values[:5, :3], order="C"),
            "f": np.array(values[:5, :3], order="F"),
            "strided": values[::2, ::2],
            "negative_stride": values[::2, ::-2],
            "readonly": readonly,
        }
        for name, x in layouts.items():
            with self.subTest(layout=name):
                x_before, centers_before = x.copy(), centers.copy()
                difference = x[:, None, :] - centers[None, :, :]
                expected = np.sum(difference * difference, axis=2)
                actual = core._distance(x, centers, "numpy")
                self.assert_bits_equal(actual, expected)
                self.assert_bits_equal(x, x_before)
                self.assert_bits_equal(centers, centers_before)
                self.assertFalse(np.shares_memory(actual, x))
                self.assertFalse(np.shares_memory(actual, centers))

    def test_repeated_calls_and_single_feature_output(self):
        x = np.array([[0.0, 1.0], [2.0, -1.0], [0.0, 1.0]])
        centers = np.array([[0.0, 1.0], [3.0, 4.0]])
        first = core._distance(x, centers, "numpy")
        saved = first.copy()
        second = core._distance(x, centers, "numpy")
        self.assert_bits_equal(first, saved)
        self.assert_bits_equal(second, saved)
        self.assertFalse(np.shares_memory(first, second))

        one_x, one_centers = x[:, :1], centers[:, :1]
        output = np.empty((len(one_x), len(one_centers)))
        actual = core._distance(one_x, one_centers, "numpy", out=output)
        self.assertIs(actual, output)
        self.assert_bits_equal(actual, (one_x - one_centers.T) ** 2)

    def test_full_fit_matches_previous_distance_expression(self):
        x = np.arange(24, dtype=np.float64).reshape(8, 3) / 8
        init = (np.arange(24, dtype=np.float64).reshape(8, 3) % 7 + 1) / 8
        x_before, init_before = x.copy(), init.copy()
        x.flags.writeable = init.flags.writeable = False
        original_distance = core._distance

        def previous_distance(data, centers, backend, xnorm=None, out=None):
            if backend == "numpy" and data.shape[1] != 1:
                difference = data[:, None, :] - centers[None, :, :]
                return np.sum(difference * difference, axis=2)
            return original_distance(data, centers, backend, xnorm, out)

        for m in (2.0, 3.0):
            with self.subTest(m=m):
                kwargs = dict(init=init, m=m, max_iter=3, tol=0,
                              backend="numpy", threads=1, return_history=True)
                actual = core.fit_fcm(x, **kwargs)
                with patch.object(core, "_distance", previous_distance):
                    expected = core.fit_fcm(x, **kwargs)
                self.assertEqual(actual.keys(), expected.keys())
                for key in actual:
                    if isinstance(actual[key], np.ndarray):
                        self.assert_bits_equal(actual[key], expected[key])
                        self.assertFalse(np.shares_memory(actual[key], x))
                        self.assertFalse(np.shares_memory(actual[key], init))
                    else:
                        self.assertEqual(actual[key], expected[key], key)
                self.assert_bits_equal(x, x_before)
                self.assert_bits_equal(init, init_before)


if __name__ == "__main__":
    unittest.main()
