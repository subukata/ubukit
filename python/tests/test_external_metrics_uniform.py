"""Exact pair construction regressions for uniform cluster-size margins."""
import numpy as np
import pytest

import ubukit._impl.external_metrics as em


def _assert_pairs(left, right, expected, dtype=np.int64):
    before = left.copy(), right.copy()
    actual = em._margin_pairs(left, right)
    assert actual.dtype == np.dtype(dtype)
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=dtype))
    np.testing.assert_array_equal(left, before[0])
    np.testing.assert_array_equal(right, before[1])
    return actual


def _reject_general_pair_expansion(*args, **kwargs):
    raise AssertionError("uniform integer margins must not expand Cartesian pairs")


@pytest.mark.parametrize("reverse", [False, True])
def test_uniform_margin_crossover_is_exact_and_skips_expansion(monkeypatch, reverse):
    """Exercise each shortcut and values below, equal to, and above its size."""
    left = np.array([5, 5, 5], dtype=np.int64)
    right = np.array([9, 2, 5, 2, 7, 9], dtype=np.int64)
    if reverse:
        left, right = right, left
    monkeypatch.setattr(em.np, "repeat", _reject_general_pair_expansion)
    _assert_pairs(left, right, [[2, 5, 6], [5, 5, 3], [5, 7, 3], [5, 9, 6]])


@pytest.mark.parametrize("other", [4, 9])
@pytest.mark.parametrize("reverse", [False, True])
def test_both_uniform_margins_have_one_exact_pair(monkeypatch, other, reverse):
    left = np.array([4, 4, 4], dtype=np.int64)
    right = np.array([other, other], dtype=np.int64)
    if reverse:
        left, right = right, left
    monkeypatch.setattr(em.np, "repeat", _reject_general_pair_expansion)
    _assert_pairs(left, right, [[4, other, 6]])


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_integer_dtypes_preserve_general_promotion(monkeypatch, reverse):
    """The private mixed-domain path can merge sizes after float64 promotion."""
    left = np.array([1], dtype=np.uint64)
    right = np.array([2**53, 2**53 + 1], dtype=np.int64)
    if reverse:
        left, right = right, left
    original = em.np.lexsort
    calls = []

    def track(keys, *args, **kwargs):
        calls.append(True)
        return original(keys, *args, **kwargs)

    monkeypatch.setattr(em.np, "lexsort", track)
    _assert_pairs(left, right, [[1, 2**53, 2]], dtype=np.float64)
    assert calls == [True]


@pytest.mark.parametrize("left,right", [([], []), ([2], []), ([], [2])])
def test_private_empty_margin_behavior_is_preserved(left, right):
    """Public empty labels exit earlier; the private helper retains its error."""
    with pytest.raises(IndexError):
        em._margin_pairs(np.array(left, dtype=np.int64), np.array(right, dtype=np.int64))


@pytest.mark.parametrize("reverse", [False, True])
def test_irregular_margins_preserve_aggregation_and_order(reverse):
    left = np.array([2, 3, 3, 5], dtype=np.int64)
    right = np.array([5, 3, 2, 3], dtype=np.int64)
    if reverse:
        left, right = right, left
    actual = _assert_pairs(left, right, [
        [2, 2, 1], [2, 3, 4], [2, 5, 2],
        [3, 3, 4], [3, 5, 4], [5, 5, 1],
    ])
    np.testing.assert_array_equal(actual, actual[np.lexsort((actual[:, 1], actual[:, 0]))])


@pytest.mark.parametrize("dtype,center", [(np.int64, 2**53), (np.uint64, 2**64 - 2)])
def test_same_integer_dtype_keeps_large_adjacent_sizes_distinct(dtype, center):
    left = np.array([center, center], dtype=dtype)
    right = np.array([center - 1, center, center + 1], dtype=dtype)
    # uint64 columns and int64 multiplicities promote only at final stacking,
    # matching the general path; rows must remain distinct before that point.
    output_dtype = np.result_type(dtype, np.int64)
    _assert_pairs(left, right, [
        [center - 1, center, 2],
        [center, center, 2],
        [center, center + 1, 2],
    ], dtype=output_dtype)


@pytest.mark.skipif(
    np.iinfo(np.intp).max < 2_500_000_000,
    reason="The unchanged baseline unique-count arithmetic can overflow on 32-bit platforms",
)
def test_uniform_multiplicity_is_not_narrowed_to_int32():
    left = np.full(50_000, 3, dtype=np.int64)
    right = np.full(50_000, 4, dtype=np.int64)
    _assert_pairs(left, right, [[3, 4, 2_500_000_000]])
