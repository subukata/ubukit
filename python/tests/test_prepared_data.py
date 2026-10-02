"""Public PreparedData metadata isolation and shared immutable payload checks."""
import unittest

import numpy as np
import ubukit as uk


GETTERS = {
    'X': lambda p: p.X,
    'lower': lambda p: p.feature_bounds()[0],
    'upper': lambda p: p.feature_bounds()[1],
    'centered': lambda p: p.centered(),
    'norms': lambda p: p.norms(),
    'centered_norms': lambda p: p.norms(preprocessing='mean-centered'),
}


def capture(p):
    return {name: get(p).copy() for name, get in GETTERS.items()}


def change_metadata(array, change):
    if change == 'shape':
        array.shape = (array.size, 1)
    elif change == 'dtype':
        array.dtype = np.int32 if array.itemsize == 4 else np.int64
    else:
        array.dtype = np.uint8


class PreparedDataHeaders(unittest.TestCase):
    def assert_outputs(self, p, expected):
        for name, get in GETTERS.items():
            actual = get(p)
            self.assertEqual(actual.shape, expected[name].shape, name)
            self.assertEqual(actual.dtype, expected[name].dtype, name)
            np.testing.assert_array_equal(actual, expected[name])

    def test_owned_snapshot_and_ordinary_values(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                source = np.arange(12., dtype=dtype).reshape(4, 3)
                expected = source.copy()
                p = uk.prepare(source)
                source[:] = -99
                centered = expected - expected.mean(axis=0)
                np.testing.assert_array_equal(p.X, expected)
                np.testing.assert_array_equal(p.centered(), centered)
                np.testing.assert_array_equal(p.feature_bounds()[0], expected.min(axis=0))
                np.testing.assert_array_equal(p.feature_bounds()[1], expected.max(axis=0))
                np.testing.assert_array_equal(p.norms(), np.einsum('ij,ij->i', expected, expected))
                np.testing.assert_array_equal(p.norms(preprocessing='mean-centered'),
                                              np.einsum('ij,ij->i', centered, centered))

    def test_independent_headers_share_read_only_payload(self):
        p = uk.prepare(np.arange(12.).reshape(4, 3))
        self.assertIsNot(p.feature_bounds(), p.feature_bounds())
        for name, get in GETTERS.items():
            with self.subTest(output=name):
                first, second = get(p), get(p)
                self.assertIsNot(first, second)
                self.assertTrue(np.shares_memory(first, second))
                self.assertFalse(first.flags.writeable)
                with self.assertRaises(ValueError):
                    first.flat[0] = 99
                with self.assertRaises(ValueError):
                    first.setflags(write=True)
                writable = first.copy()
                writable.flat[0] = 99
                np.testing.assert_array_equal(get(p), second)

    def test_each_public_header_change_leaves_all_outputs_unchanged(self):
        for dtype in (np.float32, np.float64):
            for name, get in GETTERS.items():
                for change in ('shape', 'dtype', 'narrow_dtype'):
                    with self.subTest(dtype=dtype, output=name, change=change):
                        p = uk.prepare(np.arange(12., dtype=dtype).reshape(4, 3))
                        expected = capture(p)
                        change_metadata(get(p), change)
                        self.assert_outputs(p, expected)

    def test_X_header_changes_before_cache_creation(self):
        source = np.arange(12.).reshape(4, 3)
        expected = capture(uk.prepare(source))
        for change in ('shape', 'dtype', 'narrow_dtype'):
            with self.subTest(change=change):
                p = uk.prepare(source)
                change_metadata(p.X, change)
                self.assert_outputs(p, expected)

    def test_exposed_base_header_changes_leave_outputs_unchanged(self):
        for name, get in GETTERS.items():
            for change in ('shape', 'dtype', 'narrow_dtype'):
                with self.subTest(output=name, change=change):
                    p = uk.prepare(np.arange(12.).reshape(4, 3))
                    expected = capture(p)
                    exposed = get(p)
                    expected_exposed = exposed.copy()
                    base = exposed.base
                    while isinstance(base, np.ndarray):
                        with self.assertRaises(ValueError):
                            base.setflags(write=True)
                        change_metadata(base, change)
                        base = base.base
                    self.assertIsInstance(base, bytes)
                    self.assertEqual(exposed.shape, expected_exposed.shape)
                    self.assertEqual(exposed.dtype, expected_exposed.dtype)
                    np.testing.assert_array_equal(exposed, expected_exposed)
                    self.assert_outputs(p, expected)

    def test_PreparedData_identity_and_cached_conversion_are_preserved(self):
        for name, get in GETTERS.items():
            with self.subTest(output=name):
                p = uk.prepare(np.arange(12.).reshape(4, 3))
                self.assertIs(uk.prepare(p), p)
                self.assertIs(uk.prepare(p, dtype=np.float64), p)
                self.assertIs(p.as_dtype(np.float64), p)
                q = p.as_dtype(np.float32)
                expected_p, expected_q = capture(p), capture(q)
                change_metadata(get(q), 'shape')
                change_metadata(get(q), 'narrow_dtype')
                self.assertIs(p.as_dtype(np.float32), q)
                self.assertIs(uk.prepare(p, dtype=np.float32), q)
                self.assertIs(q.as_dtype(np.float32), q)
                self.assert_outputs(p, expected_p)
                self.assert_outputs(q, expected_q)

    def test_cache_reuse_and_clear_keep_existing_semantics(self):
        p = uk.prepare(np.arange(12.).reshape(4, 3))
        first = p.norms()
        initial = p.cache_info()
        self.assertEqual((initial['hits'], initial['misses']), (0, 1))
        second = p.norms()
        self.assertEqual(p.cache_info()['keys'], initial['keys'])
        self.assertEqual((p.cache_info()['hits'], p.cache_info()['misses']), (1, 1))
        self.assertTrue(np.shares_memory(first, second))
        p.clear_cache()
        self.assertEqual(p.cache_info(), {'hits': 0, 'misses': 0, 'keys': ()})
        third = p.norms()
        np.testing.assert_array_equal(third, first)
        self.assertFalse(np.shares_memory(third, first))

    def test_public_numpy_kmeans_survives_exported_header_changes(self):
        source = np.array([[0., 1.], [2., 3.], [8., 9.], [10., 11.]])
        centers = source[[0, 2]].copy()
        kwargs = dict(max_iter=2, backend='numpy', finalize=False,
                      policy=uk.ExecutionPolicy(threads=1))
        expected = uk.fit_kmeans(source, centers, **kwargs)
        for name, get in GETTERS.items():
            for change in ('shape', 'dtype'):
                with self.subTest(output=name, change=change):
                    p = uk.prepare(source)
                    capture(p)
                    change_metadata(get(p), change)
                    actual = uk.fit_kmeans(p, centers, **kwargs)
                    for key in ('centers', 'labels', 'core_labels'):
                        np.testing.assert_array_equal(actual[key], expected[key])
                    self.assertEqual(actual['n_iter'], expected['n_iter'])
                    self.assertIsNone(actual['inertia'])
                    self.assertTrue(actual['prepared_input'])


if __name__ == '__main__':
    unittest.main()
