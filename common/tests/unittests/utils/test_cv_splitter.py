from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from autogluon.common.utils.cv_splitter import CVSplitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n: int = 10, index=None):
    """Return (X, y) with n samples and the given DataFrame index."""
    if index is None:
        index = pd.RangeIndex(n)
    X = pd.DataFrame({"f": range(n)}, index=index)
    y = pd.Series([i % 2 for i in range(n)], index=index, name="label")
    return X, y


def _two_fold_splits(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build a simple 2-fold positional split over n samples."""
    mid = n // 2
    return [
        (np.arange(mid, n), np.arange(0, mid)),  # fold 0: train=second half, test=first half
        (np.arange(0, mid), np.arange(mid, n)),  # fold 1: train=first half, test=second half
    ]


# ---------------------------------------------------------------------------
# Basic correctness – split() returns the exact custom_splits provided
# ---------------------------------------------------------------------------


class TestCustomSplitsReturnValue:
    def test_returns_same_object(self):
        splits = _two_fold_splits(10)
        cv = CVSplitter(n_splits=2, n_repeats=1, custom_splits=splits)
        X, y = _make_data(10)
        result = cv.split(X, y)
        assert result is splits

    def test_two_folds_one_repeat(self):
        n = 10
        splits = _two_fold_splits(n)
        cv = CVSplitter(n_splits=2, n_repeats=1, custom_splits=splits)
        X, y = _make_data(n)
        result = cv.split(X, y)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0][0], splits[0][0])
        np.testing.assert_array_equal(result[0][1], splits[0][1])

    def test_three_folds_two_repeats(self):
        n = 12
        # Build 3-fold splits for 2 repeats (6 total)
        idx = np.arange(n)
        splits = [
            (np.delete(idx, np.arange(0, 4)), np.arange(0, 4)),
            (np.delete(idx, np.arange(4, 8)), np.arange(4, 8)),
            (np.delete(idx, np.arange(8, 12)), np.arange(8, 12)),
            (np.delete(idx, np.arange(0, 4)), np.arange(0, 4)),  # repeat 2
            (np.delete(idx, np.arange(4, 8)), np.arange(4, 8)),
            (np.delete(idx, np.arange(8, 12)), np.arange(8, 12)),
        ]
        cv = CVSplitter(n_splits=3, n_repeats=2, custom_splits=splits)
        X, y = _make_data(n)
        result = cv.split(X, y)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# Assertion errors – invalid custom_splits configurations
# ---------------------------------------------------------------------------


class TestCustomSplitsAssertions:
    """Every assertion in CVSplitter.split() must raise AssertionError when violated."""

    def test_wrong_number_of_splits(self):
        splits = _two_fold_splits(10)  # 2 splits, but we declare n_splits=3
        cv = CVSplitter(n_splits=3, n_repeats=1, custom_splits=splits)
        X, y = _make_data(10)
        with pytest.raises(AssertionError, match="n_splits"):
            cv.split(X, y)

    def test_wrong_number_of_splits_message_mentions_repeats(self):
        splits = _two_fold_splits(10)  # 2 splits, but 2*2=4 expected
        cv = CVSplitter(n_splits=2, n_repeats=2, custom_splits=splits)
        X, y = _make_data(10)
        with pytest.raises(AssertionError, match="n_repeats"):
            cv.split(X, y)

    def test_split_not_two_elements(self):
        bad = [(np.arange(5), np.arange(5, 10), np.array([]))]  # 3-element tuple
        cv = CVSplitter(n_splits=1, n_repeats=1, custom_splits=bad)
        X, y = _make_data(10)
        with pytest.raises(AssertionError, match="2-element"):
            cv.split(X, y)

    def test_empty_train_set(self):
        bad = [(np.array([], dtype=int), np.arange(10))]
        cv = CVSplitter(n_splits=1, n_repeats=1, custom_splits=bad)
        X, y = _make_data(10)
        with pytest.raises(AssertionError, match="empty training"):
            cv.split(X, y)

    def test_empty_test_set(self):
        bad = [(np.arange(10), np.array([], dtype=int))]
        cv = CVSplitter(n_splits=1, n_repeats=1, custom_splits=bad)
        X, y = _make_data(10)
        with pytest.raises(AssertionError, match="empty test"):
            cv.split(X, y)

    def test_overlapping_train_test(self):
        # Index 0 appears in both train and test
        bad = [(np.arange(0, 8), np.arange(5, 10))]
        cv = CVSplitter(n_splits=1, n_repeats=1, custom_splits=bad)
        X, y = _make_data(10)
        with pytest.raises(AssertionError, match="overlap"):
            cv.split(X, y)

    def test_out_of_range_index_high(self):
        # Index 10 is out of range for n=10 (valid: 0..9)
        bad = [(np.arange(1, 10), np.array([10]))]
        cv = CVSplitter(n_splits=1, n_repeats=1, custom_splits=bad)
        X, y = _make_data(10)
        with pytest.raises(AssertionError, match="positional"):
            cv.split(X, y)

    def test_out_of_range_index_negative(self):
        bad = [(np.arange(0, 9), np.array([-1]))]
        cv = CVSplitter(n_splits=1, n_repeats=1, custom_splits=bad)
        X, y = _make_data(10)
        with pytest.raises(AssertionError, match="positional"):
            cv.split(X, y)


# ---------------------------------------------------------------------------
# Correct behaviour across different DataFrame index types
# ---------------------------------------------------------------------------


class TestCustomSplitsDataFrameIndex:
    """custom_splits always use *positional* indices (0..n-1), regardless of the
    DataFrame's own index labels. These tests verify that split() returns the
    custom splits unchanged for various index types, and that the fold configs
    generated from them refer to positional indices that work correctly with
    DataFrame.iloc.
    """

    def _run(self, X, y, splits):
        cv = CVSplitter(n_splits=2, n_repeats=1, custom_splits=splits)
        result = cv.split(X, y)
        assert len(result) == 2
        # Verify iloc access works for both folds
        for train_pos, test_pos in result:
            X_train = X.iloc[train_pos]
            X_test = X.iloc[test_pos]
            assert len(X_train) + len(X_test) == len(X)
        return result

    def test_default_rangeindex(self):
        n = 10
        X, y = _make_data(n)
        splits = _two_fold_splits(n)
        result = self._run(X, y, splits)
        # Positional index == label index for default RangeIndex
        train_pos, test_pos = result[0]
        np.testing.assert_array_equal(X.iloc[test_pos].index.to_numpy(), test_pos)

    def test_offset_integer_index(self):
        """DataFrame with index starting at 100 – positional splits still work."""
        n = 10
        index = pd.RangeIndex(start=100, stop=100 + n)
        X, y = _make_data(n, index=index)
        splits = _two_fold_splits(n)
        result = self._run(X, y, splits)
        # iloc[0] should give the row with label 100
        assert X.iloc[0].name == 100
        _, test_pos = result[0]
        assert set(X.iloc[test_pos].index.tolist()) == set(range(100, 100 + n // 2))

    def test_noncontiguous_integer_index(self):
        """DataFrame with a shuffled / non-contiguous integer index."""
        n = 10
        index = pd.Index([5, 99, 3, 42, 7, 0, 11, 88, 1, 55])
        X, y = _make_data(n, index=index)
        splits = _two_fold_splits(n)
        result = self._run(X, y, splits)
        # Positional 0 → label 5
        assert X.iloc[0].name == 5
        _, test_pos = result[0]
        # test_pos = [0,1,2,3,4] → labels [5,99,3,42,7]
        assert X.iloc[test_pos].index.tolist() == [5, 99, 3, 42, 7]

    def test_string_index(self):
        """DataFrame with string index labels."""
        n = 6
        index = pd.Index([f"row_{i}" for i in range(n)])
        X, y = _make_data(n, index=index)
        np.arange(n)
        splits = [
            (np.array([3, 4, 5]), np.array([0, 1, 2])),
            (np.array([0, 1, 2]), np.array([3, 4, 5])),
        ]
        cv = CVSplitter(n_splits=2, n_repeats=1, custom_splits=splits)
        result = cv.split(X, y)
        _, test_pos = result[0]
        assert X.iloc[test_pos].index.tolist() == ["row_0", "row_1", "row_2"]

    def test_generate_fold_configs_with_offset_index(self):
        """Integration: BaggedEnsembleModel._generate_fold_configs uses positional indices
        from custom_splits regardless of the DataFrame's own index labels.
        """
        from autogluon.core.models import BaggedEnsembleModel

        n = 8
        index = pd.RangeIndex(start=1000, stop=1000 + n)
        X, y = _make_data(n, index=index)

        splits = _two_fold_splits(n)
        cv = CVSplitter(n_splits=2, n_repeats=1, custom_splits=splits)

        fold_fit_args_list, _, _ = BaggedEnsembleModel._generate_fold_configs(
            X=X,
            y=y,
            cv_splitter=cv,
            k_fold_start=0,
            k_fold_end=2,
            n_repeat_start=0,
            n_repeat_end=1,
            vary_seed_across_folds=False,
            random_seed_offset=0,
        )

        assert len(fold_fit_args_list) == 2
        _, test_idx_0 = fold_fit_args_list[0]["fold"]
        _, test_idx_1 = fold_fit_args_list[1]["fold"]
        # Positional indices: fold 0 tests rows 0..3, fold 1 tests rows 4..7
        np.testing.assert_array_equal(test_idx_0, np.arange(0, n // 2))
        np.testing.assert_array_equal(test_idx_1, np.arange(n // 2, n))
        # iloc access with those positional indices gives the correct labels
        assert X.iloc[test_idx_0].index[0] == 1000
        assert X.iloc[test_idx_1].index[0] == 1000 + n // 2
