"""Regressions for the within-call NumPy EMI pair-order optimization."""
import numpy as np
import pytest

import ubukit._impl.external_metrics as em


@pytest.mark.parametrize("left,right", [
    ([1, 2, 2, 5], [3, 3, 4]),
    ([1, 1, 1, 7], [2, 2, 6]),
    ([2, 2, 2, 2], [1, 3, 4]),
    ([10], [2, 3, 5]),
    ([1] * 8, [1, 1, 2, 4]),
])
def test_margin_pairs_already_have_stable_low_order(left, right):
    """The producer must preserve the ordering promised to the EMI consumer."""
    a, b = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    assert a.sum() == b.sum()
    pairs = em._margin_pairs(a, b)
    expected = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    np.testing.assert_array_equal(pairs, expected)
    np.testing.assert_array_equal(pairs, pairs[np.argsort(pairs[:, 0], kind="stable")])
    np.testing.assert_array_equal(pairs, em._margin_pairs(b, a))
    assert int(pairs[:, 2].sum()) == len(a) * len(b)


def test_default_emi_call_still_stable_sorts_unsorted_pairs(monkeypatch):
    """Direct internal callers need not know about the producer's invariant."""
    pairs = np.array([[5, 8, 1], [2, 4, 1], [2, 3, 2], [3, 4, 1]], dtype=np.int64)
    original = pairs.copy()
    argsort = np.argsort
    sorted_pairs = pairs[argsort(pairs[:, 0], kind="stable")]
    expected = em._emi_numpy(20, sorted_pairs, presorted=True)
    calls = []

    def record_argsort(values, *args, **kwargs):
        calls.append((np.array(values, copy=True), kwargs.get("kind")))
        return argsort(values, *args, **kwargs)

    monkeypatch.setattr(em.np, "argsort", record_argsort)
    assert em._emi_numpy(20, pairs) == expected
    assert len(calls) == 1 and calls[0][1] == "stable"
    np.testing.assert_array_equal(calls[0][0], original[:, 0])
    np.testing.assert_array_equal(pairs, original)


def test_presorted_emi_call_skips_sort_without_changing_value(monkeypatch):
    pairs = em._margin_pairs(np.array([1, 2, 2, 5]), np.array([3, 3, 4]))
    original = pairs.copy()
    expected = em._emi_numpy(10, pairs)

    def reject_redundant_sort(*args, **kwargs):
        raise AssertionError("presorted EMI must not sort or copy via argsort")

    monkeypatch.setattr(em.np, "argsort", reject_redundant_sort)
    assert em._emi_numpy(10, pairs, presorted=True) == expected
    np.testing.assert_array_equal(pairs, original)


def test_numpy_dispatch_passes_the_presorted_contract(monkeypatch):
    a, b = np.array([1, 2, 2, 5]), np.array([3, 3, 4])
    pairs = em._margin_pairs(a, b)
    implementation = em._emi_numpy
    expected = implementation(10, pairs)
    flags = []

    def capture(n, grouped_pairs, *, presorted=False):
        flags.append(presorted)
        np.testing.assert_array_equal(grouped_pairs, pairs)
        return implementation(n, grouped_pairs, presorted=presorted)

    monkeypatch.setattr(em, "_emi_numpy", capture)
    assert em._emi(10, a, b, "numpy") == expected
    assert flags == [True]
