"""Unit tests for CustomCVSplitter."""

import numpy as np
import pandas as pd
import pytest

from autogluon.core.utils.cv import CustomCVSplitter


class TestCustomCVSplitter:
    """Test suite for CustomCVSplitter class."""

    def test_basic_functionality(self):
        """Test basic CustomCVSplitter functionality."""
        # Create simple CV matrix
        cv_matrix = pd.DataFrame({
            'fold_1': [0, 1, 2],
            'fold_2': [0, 0, 1],
        })

        splitter = CustomCVSplitter(cv_matrix)
        assert splitter.n_splits == 2

        # Test split generation
        splits = list(splitter.split(X=np.zeros((3, 2))))
        assert len(splits) == 2

        # Check first fold
        train_idx, test_idx = splits[0]
        assert list(train_idx) == [0]
        assert list(test_idx) == [1]

        # Check second fold
        train_idx, test_idx = splits[1]
        assert list(train_idx) == [0, 1]
        assert list(test_idx) == [2]

    def test_numpy_array_input(self):
        """Test that numpy arrays are accepted and converted."""
        cv_matrix = np.array([[0, 0], [1, 0], [2, 1]])

        splitter = CustomCVSplitter(cv_matrix)
        assert splitter.n_splits == 2
        assert isinstance(splitter.cv_matrix, pd.DataFrame)

    def test_get_n_splits(self):
        """Test get_n_splits method."""
        cv_matrix = pd.DataFrame({
            'fold_1': [0, 1],
            'fold_2': [0, 1],
            'fold_3': [0, 1],
        })

        splitter = CustomCVSplitter(cv_matrix)
        assert splitter.get_n_splits() == 3
        assert splitter.get_n_splits(X=None, y=None) == 3

    def test_invalid_values(self):
        """Test validation of invalid matrix values."""
        # Invalid value (3)
        with pytest.raises(ValueError, match="invalid values"):
            cv_matrix = pd.DataFrame({'fold_1': [0, 3]})
            CustomCVSplitter(cv_matrix)

        # Invalid value (-1)
        with pytest.raises(ValueError, match="invalid values"):
            cv_matrix = pd.DataFrame({'fold_1': [0, -1]})
            CustomCVSplitter(cv_matrix)

    def test_no_train_samples(self):
        """Test validation when fold has no training samples."""
        with pytest.raises(ValueError, match="no training samples"):
            cv_matrix = pd.DataFrame({'fold_1': [1, 2]})  # No 0s
            CustomCVSplitter(cv_matrix)

    def test_no_test_samples(self):
        """Test validation when fold has no test samples."""
        with pytest.raises(ValueError, match="no test samples"):
            cv_matrix = pd.DataFrame({'fold_1': [0, 0]})  # No 1s
            CustomCVSplitter(cv_matrix)

    def test_empty_matrix(self):
        """Test validation of empty matrix."""
        with pytest.raises(ValueError, match="cannot be empty"):
            cv_matrix = pd.DataFrame()
            CustomCVSplitter(cv_matrix)

    def test_sample_count_mismatch(self):
        """Test error when X has different number of samples than cv_matrix."""
        cv_matrix = pd.DataFrame({'fold_1': [0, 1, 2]})
        splitter = CustomCVSplitter(cv_matrix)

        # X has 5 samples but cv_matrix has 3
        with pytest.raises(ValueError, match="does not match"):
            list(splitter.split(X=np.zeros((5, 2))))

    def test_get_fold_stats(self):
        """Test get_fold_stats method."""
        cv_matrix = pd.DataFrame({
            'fold_1': [0, 0, 1, 2],
            'fold_2': [0, 0, 0, 1],
        })

        splitter = CustomCVSplitter(cv_matrix)
        stats = splitter.get_fold_stats()

        assert len(stats) == 2
        assert list(stats.columns) == ['fold', 'n_train', 'n_test', 'n_exclude', 'train_pct', 'test_pct']

        # Check fold 1
        fold1_stats = stats[stats['fold'] == 'fold_1'].iloc[0]
        assert fold1_stats['n_train'] == 2
        assert fold1_stats['n_test'] == 1
        assert fold1_stats['n_exclude'] == 1
        assert fold1_stats['train_pct'] == 50.0
        assert fold1_stats['test_pct'] == 25.0

        # Check fold 2
        fold2_stats = stats[stats['fold'] == 'fold_2'].iloc[0]
        assert fold2_stats['n_train'] == 3
        assert fold2_stats['n_test'] == 1
        assert fold2_stats['n_exclude'] == 0

    def test_get_sample_coverage(self):
        """Test get_sample_coverage method."""
        cv_matrix = pd.DataFrame({
            'fold_1': [0, 1, 2],
            'fold_2': [0, 0, 1],
            'fold_3': [1, 0, 0],
        })

        splitter = CustomCVSplitter(cv_matrix)
        coverage = splitter.get_sample_coverage()

        # All samples should be tested at least once
        assert coverage['tested_samples'] == {0, 1, 2}
        assert coverage['never_tested'] == set()
        assert coverage['coverage_pct'] == 100.0

        # Check test counts
        assert coverage['test_counts'][0] == 1  # Sample 0 tested in fold_3
        assert coverage['test_counts'][1] == 1  # Sample 1 tested in fold_1
        assert coverage['test_counts'][2] == 1  # Sample 2 tested in fold_2

    def test_samples_tested_multiple_times(self):
        """Test coverage when samples are tested in multiple folds."""
        cv_matrix = pd.DataFrame({
            'fold_1': [1, 0, 2],
            'fold_2': [1, 0, 1],  # Samples 0 and 2 both tested
        })

        splitter = CustomCVSplitter(cv_matrix)
        coverage = splitter.get_sample_coverage()

        # Sample 0 is tested twice, sample 2 is tested once
        assert coverage['test_counts'][0] == 2
        assert coverage['test_counts'][1] == 0  # Never tested
        assert coverage['test_counts'][2] == 1

        assert coverage['never_tested'] == {1}
        assert coverage['coverage_pct'] == pytest.approx(66.666, rel=0.01)

    def test_forward_chaining_pattern(self):
        """Test that forward chaining pattern works correctly."""
        # Simulate forward chaining for 4 time periods
        cv_matrix = pd.DataFrame({
            'fold_1': [0, 1, 2, 2],  # train t1, test t2
            'fold_2': [0, 0, 1, 2],  # train t1-t2, test t3
            'fold_3': [0, 0, 0, 1],  # train t1-t3, test t4
        })

        splitter = CustomCVSplitter(cv_matrix)

        splits = list(splitter.split(X=np.zeros((4, 2))))

        # Verify expanding training set
        assert len(splits[0][0]) == 1  # First fold: 1 train sample
        assert len(splits[1][0]) == 2  # Second fold: 2 train samples
        assert len(splits[2][0]) == 3  # Third fold: 3 train samples

        # Verify single test sample per fold
        assert len(splits[0][1]) == 1
        assert len(splits[1][1]) == 1
        assert len(splits[2][1]) == 1

    def test_repr(self):
        """Test string representation."""
        cv_matrix = pd.DataFrame({
            'fold_1': [0, 1, 2],
            'fold_2': [0, 0, 1],
        })

        splitter = CustomCVSplitter(cv_matrix)
        repr_str = repr(splitter)

        assert 'CustomCVSplitter' in repr_str
        assert 'n_splits=2' in repr_str
        assert 'n_samples=3' in repr_str

    def test_with_pandas_dataframe_features(self):
        """Test split with pandas DataFrame as X."""
        cv_matrix = pd.DataFrame({
            'fold_1': [0, 1, 2],
            'fold_2': [0, 0, 1],
        })

        splitter = CustomCVSplitter(cv_matrix)

        # Create DataFrame X
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

        splits = list(splitter.split(X=X))
        assert len(splits) == 2

        # Verify indices work correctly
        train_idx, test_idx = splits[0]
        assert list(X.iloc[train_idx].index) == [0]
        assert list(X.iloc[test_idx].index) == [1]

    def test_invalid_type(self):
        """Test that invalid types raise appropriate errors."""
        with pytest.raises(TypeError, match="must be pd.DataFrame or np.ndarray"):
            CustomCVSplitter([0, 1, 2])  # List is not valid
