"""Tiny v2 correctness tests, not a performance campaign."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from numba import get_num_threads
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from threadpoolctl import threadpool_limits
from .finalizer import finalize


def direct_reference(X, centers):
    distances = np.zeros((len(X), len(centers)), dtype=np.float64)
    for j in range(X.shape[1]):
        delta = X[:, None, j] - centers[None, :, j]
        distances += delta * delta
    labels = distances.argmin(axis=1)
    squared = distances[np.arange(len(X)), labels]
    return labels, squared, float(np.sum(squared, dtype=np.float64))


def main():
    started = time.perf_counter()
    rng = np.random.default_rng(202609302334)
    cases = {
        'exact_ties': (np.array([[0., 0.], [2., 0.], [-2., 0.], [0., 2.], [0., -2.]]),
                       np.array([[1., 1.], [-1., -1.], [99., 99.]])),
        'duplicates_empty': (np.array([[0., 0.], [1., 1.], [-1., -1.]]),
                             np.array([[0., 0.], [0., 0.], [99., 99.], [1., 1.], [1., 1.]])),
        'identical': (np.ones((9, 3)), np.ones((5, 3))),
        'one_row': (np.array([[2., -1.]]), np.array([[8., 5.]])),
        'k_above_n': (np.array([[0.], [1.]]), np.array([[0.], [1.], [2.], [3.], [4.]])),
        'large_offset_ties': (np.array([[1e9], [1e9 + 1], [1e9 + 2]]),
                              np.array([[1e9], [1e9 + 2]])),
        'near_tie': (np.array([[0.], [np.finfo(float).eps], [-np.finfo(float).eps]]),
                     np.array([[-1.], [1.], [1.]])),
    }
    for n, d, k in [(31, 7, 5), (37, 13, 9), (51, 32, 17), (53, 129, 33), (1, 7, 17)]:
        X = rng.normal(size=(n, d)); centers = rng.normal(size=(k, d))
        cases[f'random_{n}_{d}_{k}'] = X, centers
    records = []
    for name, (X, centers) in cases.items():
        expected = direct_reference(X, centers)
        original_X, original_centers = X.copy(), centers.copy()
        first = None
        for threads in [1, 2]:
            previous_threads = get_num_threads()
            out = finalize(X, centers, threads=threads)
            assert get_num_threads() == previous_threads
            np.testing.assert_array_equal(out['labels'], expected[0])
            np.testing.assert_array_equal(out['squared_distances'], expected[1])
            assert out['inertia'] == expected[2]
            np.testing.assert_array_equal(X, original_X)
            np.testing.assert_array_equal(centers, original_centers)
            if first is not None:
                np.testing.assert_array_equal(out['labels'], first['labels'])
                np.testing.assert_array_equal(out['squared_distances'], first['squared_distances'])
                assert out['inertia'] == first['inertia']
            first = out
            records.append({'case': name, 'threads': threads, 'status': 'bitwise_direct_equal'})
        with threadpool_limits(limits=1):
            labels, rooted = pairwise_distances_argmin_min(X, centers, metric='euclidean')
        mismatch = int(np.count_nonzero(labels != expected[0]))
        sensitive = name in ['large_offset_ties', 'near_tie', 'exact_ties', 'duplicates_empty', 'identical']
        if not sensitive:
            np.testing.assert_array_equal(labels, expected[0])
            np.testing.assert_allclose(rooted * rooted, expected[1], rtol=1e-12, atol=1e-12)
        records.append({'case': name, 'sklearn_argmin_label_mismatches': mismatch,
                        'sklearn_squared_max_abs_error': float(np.max(np.abs(rooted * rooted - expected[1]))),
                        'numerically_sensitive': sensitive,
                        'difference_policy': 'reported, not hidden' if sensitive else 'assert_equal_labels_close_distances'})
    X, C = cases['random_37_13_9']
    immutable_X = np.frombuffer(X.tobytes(), dtype=np.float64).reshape(X.shape)
    immutable_C = np.frombuffer(C.tobytes(), dtype=np.float64).reshape(C.shape)
    for XX, CC in [(immutable_X, immutable_C), (X[:, ::2], C[:, ::2])]:
        out = finalize(XX, CC, threads=2)
        expected = direct_reference(XX, CC)
        np.testing.assert_array_equal(out['labels'], expected[0])
        np.testing.assert_array_equal(out['squared_distances'], expected[1])
    fits = []
    for name, n, d, k, offset in [('separated', 75, 7, 5, 0.),
                                 ('random', 63, 13, 7, 0.),
                                 ('offset_sensitive', 75, 7, 5, 1e9)]:
        if name == 'random':
            X = rng.normal(size=(n, d)); initial = X[:k].copy()
        else:
            means = rng.normal(size=(k, d)) * 5
            X = np.repeat(means, n // k, axis=0) + rng.normal(size=(n, d)) * 0.3 + offset
            initial = X[np.arange(k) * (n // k)].copy()
        with threadpool_limits(limits=1):
            model = KMeans(n_clusters=k, init=initial, n_init=1, max_iter=12,
                           tol=0., algorithm='lloyd').fit(X)
            predicted = model.predict(X)
        counts = np.bincount(model.labels_, minlength=k)
        assert np.all(counts > 0)
        out = finalize(X, model.cluster_centers_, threads=2)
        expected = direct_reference(X, model.cluster_centers_)
        np.testing.assert_array_equal(out['labels'], expected[0])
        np.testing.assert_array_equal(out['squared_distances'], expected[1])
        label_mismatch = int(np.count_nonzero(out['labels'] != model.labels_))
        predict_mismatch = int(np.count_nonzero(out['labels'] != predicted))
        if offset == 0:
            assert label_mismatch == 0
            assert predict_mismatch == 0
            np.testing.assert_allclose(out['inertia'], model.inertia_, rtol=1e-12, atol=1e-12)
        fits.append({'case': name, 'all_clusters_nonempty': True,
                     'sklearn_fit_label_mismatches': label_mismatch,
                     'sklearn_predict_label_mismatches': predict_mismatch,
                     'direct_inertia': out['inertia'], 'sklearn_fit_inertia': float(model.inertia_),
                     'inertia_abs_difference': abs(out['inertia'] - float(model.inertia_)),
                     'recentering_sensitive': offset != 0})
    invalid = [(np.array([[np.nan]]), np.array([[0.]])),
               (np.array([[1.]], np.float32), np.array([[0.]])),
               (np.empty((0, 1)), np.array([[0.]])),
               (np.array([[1e308]]), np.array([[-1e308]])),
               (np.array([[1e154], [1e154]]), np.array([[0.]]))]
    for X, C in invalid:
        old = get_num_threads()
        try:
            finalize(X, C, threads=2)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('invalid domain accepted')
        assert get_num_threads() == old
    root = Path(__file__).parent
    result = {'status': 'passed', 'scope': 'NEW restart-v2 tiny correctness; not speed timing',
              'direct_checks': 24, 'immutable_and_strided_checks': 2,
              'invalid_rejections': len(invalid), 'records': records,
              'actual_sklearn_nonempty_fits': fits,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(root.glob('*.py'))},
              'wall_seconds_including_compilation': time.perf_counter() - started}
    (root / 'test_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': 'passed', 'direct_checks': 24,
                      'sensitive_comparisons': [r for r in records if r.get('numerically_sensitive')],
                      'sklearn_fits': fits}, indent=2))


if __name__ == '__main__':
    main()
