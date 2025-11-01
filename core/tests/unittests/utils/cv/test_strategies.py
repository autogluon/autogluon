"""Unit tests for CV strategy generators."""

import numpy as np
import pandas as pd
import pytest

from autogluon.core.utils.cv import forward_chaining_cv, sliding_window_cv, time_series_cv


class TestForwardChainingCV:
    """Test suite for forward_chaining_cv function."""

    def test_basic_functionality(self):
        """Test basic forward chaining CV generation."""
        cv_matrix = forward_chaining_cv(n_samples=100, n_folds=5)

        assert cv_matrix.shape == (100, 5)
        assert list(cv_matrix.columns) == ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

        # Verify all values are 0, 1, or 2
        unique_values = set()
        for col in cv_matrix.columns:
            unique_values.update(cv_matrix[col].unique())
        assert unique_values.issubset({0, 1, 2})

    def test_expanding_training_set(self):
        """Test that training set expands across folds."""
        cv_matrix = forward_chaining_cv(n_samples=20, n_folds=4)

        # Count training samples in each fold
        train_counts = []
        for col in cv_matrix.columns:
            train_counts.append(np.sum(cv_matrix[col] == 0))

        # Training set should be expanding
        assert train_counts == sorted(train_counts)
        assert train_counts[0] < train_counts[-1]

    def test_each_fold_has_train_and_test(self):
        """Test that each fold has both train and test samples."""
        cv_matrix = forward_chaining_cv(n_samples=100, n_folds=5)

        for col in cv_matrix.columns:
            n_train = np.sum(cv_matrix[col] == 0)
            n_test = np.sum(cv_matrix[col] == 1)

            assert n_train > 0, f"{col} has no training samples"
            assert n_test > 0, f"{col} has no test samples"

    def test_with_min_train_size(self):
        """Test forward chaining with custom min_train_size."""
        min_train_size = 30
        cv_matrix = forward_chaining_cv(n_samples=100, n_folds=5, min_train_size=min_train_size)

        # First fold should have at least min_train_size training samples
        first_fold_train = np.sum(cv_matrix.iloc[:, 0] == 0)
        assert first_fold_train >= min_train_size

    def test_with_gap(self):
        """Test forward chaining with gap between train and test."""
        gap = 5
        cv_matrix = forward_chaining_cv(n_samples=100, n_folds=5, gap=gap)

        # For each fold, verify there's a gap between train and test
        for col in cv_matrix.columns:
            fold_values = cv_matrix[col].values

            # Find last train index and first test index
            train_indices = np.where(fold_values == 0)[0]
            test_indices = np.where(fold_values == 1)[0]

            if len(train_indices) > 0 and len(test_indices) > 0:
                last_train = train_indices[-1]
                first_test = test_indices[0]

                # Gap should be at least the specified size
                assert (first_test - last_train) >= gap

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # n_samples too small
        with pytest.raises(ValueError, match="must be greater than n_folds"):
            forward_chaining_cv(n_samples=3, n_folds=5)

        # n_folds < 1
        with pytest.raises(ValueError, match="must be at least 1"):
            forward_chaining_cv(n_samples=100, n_folds=0)

        # Negative gap
        with pytest.raises(ValueError, match="must be non-negative"):
            forward_chaining_cv(n_samples=100, n_folds=5, gap=-1)

        # min_train_size too large
        with pytest.raises(ValueError, match="Not enough samples"):
            forward_chaining_cv(n_samples=100, n_folds=5, min_train_size=96)


class TestSlidingWindowCV:
    """Test suite for sliding_window_cv function."""

    def test_basic_functionality(self):
        """Test basic sliding window CV generation."""
        cv_matrix = sliding_window_cv(n_samples=100, n_folds=5)

        assert cv_matrix.shape == (100, 5)
        assert list(cv_matrix.columns) == ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

    def test_fixed_window_size(self):
        """Test that training window size remains relatively constant."""
        window_size = 30
        cv_matrix = sliding_window_cv(n_samples=100, n_folds=5, window_size=window_size)

        # Check that most folds have approximately window_size training samples
        for col in cv_matrix.columns:
            n_train = np.sum(cv_matrix[col] == 0)
            # Allow some variation for last fold
            assert n_train <= window_size + 5  # Small tolerance

    def test_with_custom_step_size(self):
        """Test sliding window with custom step size."""
        window_size = 30
        step_size = 15
        cv_matrix = sliding_window_cv(
            n_samples=100,
            n_folds=5,
            window_size=window_size,
            step_size=step_size
        )

        # Verify window slides by step_size
        fold1_train = np.where(cv_matrix.iloc[:, 0] == 0)[0]
        fold2_train = np.where(cv_matrix.iloc[:, 1] == 0)[0]

        if len(fold1_train) > 0 and len(fold2_train) > 0:
            # Start of fold2 should be approximately step_size after fold1
            diff = fold2_train[0] - fold1_train[0]
            assert abs(diff - step_size) <= 2  # Small tolerance

    def test_with_gap(self):
        """Test sliding window with gap between train and test."""
        gap = 5
        cv_matrix = sliding_window_cv(n_samples=100, n_folds=5, gap=gap)

        # For each fold, verify there's a gap between train and test
        for col in cv_matrix.columns:
            fold_values = cv_matrix[col].values

            train_indices = np.where(fold_values == 0)[0]
            test_indices = np.where(fold_values == 1)[0]

            if len(train_indices) > 0 and len(test_indices) > 0:
                last_train = train_indices[-1]
                first_test = test_indices[0]
                assert (first_test - last_train) >= gap

    def test_each_fold_has_train_and_test(self):
        """Test that each fold has both train and test samples."""
        cv_matrix = sliding_window_cv(n_samples=100, n_folds=5)

        for col in cv_matrix.columns:
            n_train = np.sum(cv_matrix[col] == 0)
            n_test = np.sum(cv_matrix[col] == 1)

            assert n_train > 0, f"{col} has no training samples"
            assert n_test > 0, f"{col} has no test samples"

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # n_samples too small
        with pytest.raises(ValueError):
            sliding_window_cv(n_samples=3, n_folds=5)

        # n_folds < 1
        with pytest.raises(ValueError, match="must be at least 1"):
            sliding_window_cv(n_samples=100, n_folds=0)

        # window_size too large
        with pytest.raises(ValueError, match="must be less than n_samples"):
            sliding_window_cv(n_samples=100, n_folds=5, window_size=100)

        # Negative gap
        with pytest.raises(ValueError, match="must be non-negative"):
            sliding_window_cv(n_samples=100, n_folds=5, gap=-1)


class TestTimeSeriesCV:
    """Test suite for time_series_cv function."""

    def test_with_datetime_series(self):
        """Test time_series_cv with datetime series."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=100, freq='D'))
        cv_matrix = time_series_cv(dates, n_folds=5, strategy='forward_chaining')

        assert cv_matrix.shape == (100, 5)
        # Index should match dates.index
        assert all(cv_matrix.index == dates.index)

    def test_temporal_ordering_preserved(self):
        """Test that temporal ordering is preserved even if dates are unsorted."""
        # Create unsorted dates
        dates = pd.Series(pd.date_range('2020-01-01', periods=100, freq='D'))
        shuffled_dates = dates.sample(frac=1.0, random_state=42)

        cv_matrix = time_series_cv(shuffled_dates, n_folds=5, strategy='forward_chaining')

        # CV matrix should be aligned with original (shuffled) index
        assert all(cv_matrix.index == shuffled_dates.index)
        # Reorder to check for correctness
        cv_matrix_reordered = cv_matrix.loc[dates.index]
        for col in cv_matrix_reordered.columns:
            train_indices = np.where(cv_matrix_reordered[col] == 0)[0]
            test_indices = np.where(cv_matrix_reordered[col] == 1)[0]
            assert train_indices.max() < test_indices.min()

    def test_forward_chaining_strategy(self):
        """Test time_series_cv with forward_chaining strategy."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=100, freq='D'))
        cv_matrix = time_series_cv(dates, n_folds=5, strategy='forward_chaining')

        # Should create expanding training sets
        train_counts = []
        for col in cv_matrix.columns:
            train_counts.append(np.sum(cv_matrix[col] == 0))

        assert train_counts == sorted(train_counts)

    def test_sliding_window_strategy(self):
        """Test time_series_cv with sliding_window strategy."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=100, freq='D'))
        cv_matrix = time_series_cv(dates, n_folds=5, strategy='sliding_window', window_size=30)

        assert cv_matrix.shape == (100, 5)

    def test_with_timedelta_gap(self):
        """Test time_series_cv with timedelta gap."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=100, freq='D'))
        gap = pd.Timedelta(days=7)

        cv_matrix = time_series_cv(dates, n_folds=5, gap=gap)

        # Gap should be approximately 7 samples
        # (This is an approximation test due to conversion from timedelta)
        assert cv_matrix.shape == (100, 5)

    def test_with_integer_gap(self):
        """Test time_series_cv with integer gap."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=100, freq='D'))
        gap = 5

        cv_matrix = time_series_cv(dates, n_folds=5, gap=gap)

        assert cv_matrix.shape == (100, 5)

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=100, freq='D'))

        with pytest.raises(ValueError, match="Invalid strategy"):
            time_series_cv(dates, n_folds=5, strategy='invalid_strategy')

    def test_invalid_dates_type(self):
        """Test that non-Series dates raises error."""
        with pytest.raises(TypeError, match="must be pd.Series"):
            time_series_cv([1, 2, 3, 4, 5], n_folds=3)

    def test_invalid_gap_type(self):
        """Test that invalid gap type raises error."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=100, freq='D'))

        with pytest.raises(TypeError, match="must be int or pd.Timedelta"):
            time_series_cv(dates, n_folds=5, gap="invalid")


class TestStrategyIntegration:
    """Integration tests for CV strategies."""

    def test_forward_chaining_no_data_leakage(self):
        """Test that forward chaining prevents data leakage."""
        cv_matrix = forward_chaining_cv(n_samples=100, n_folds=5)

        # For each fold, verify that all test indices are after all train indices
        for col in cv_matrix.columns:
            fold_values = cv_matrix[col].values

            train_indices = np.where(fold_values == 0)[0]
            test_indices = np.where(fold_values == 1)[0]

            if len(train_indices) > 0 and len(test_indices) > 0:
                max_train_idx = train_indices.max()
                min_test_idx = test_indices.min()

                # Test samples should come after train samples
                assert min_test_idx > max_train_idx, \
                    f"Data leakage in {col}: test samples before train samples"

    def test_strategies_create_valid_matrices(self):
        """Test that all strategies create valid CV matrices."""
        n_samples = 100
        n_folds = 5

        # Test forward chaining
        fc_matrix = forward_chaining_cv(n_samples=n_samples, n_folds=n_folds)
        assert fc_matrix.shape == (n_samples, n_folds)

        # Test sliding window
        sw_matrix = sliding_window_cv(n_samples=n_samples, n_folds=n_folds)
        assert sw_matrix.shape == (n_samples, n_folds)

        # Test time series (forward chaining)
        dates = pd.Series(pd.date_range('2020-01-01', periods=n_samples, freq='D'))
        ts_matrix = time_series_cv(dates, n_folds=n_folds, strategy='forward_chaining')
        assert ts_matrix.shape == (n_samples, n_folds)

    def test_reproducibility(self):
        """Test that strategies produce consistent results."""
        # Forward chaining should be deterministic
        cv1 = forward_chaining_cv(n_samples=100, n_folds=5)
        cv2 = forward_chaining_cv(n_samples=100, n_folds=5)

        assert cv1.equals(cv2)

        # Sliding window should be deterministic
        cv1 = sliding_window_cv(n_samples=100, n_folds=5, window_size=30)
        cv2 = sliding_window_cv(n_samples=100, n_folds=5, window_size=30)

        assert cv1.equals(cv2)
