"""
Integration tests for custom cross-validation feature.

These tests verify the complete end-to-end functionality of custom CV
from the TabularPredictor API through to model training.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from autogluon.core.utils.cv import CustomCVSplitter, forward_chaining_cv, sliding_window_cv
from autogluon.tabular import TabularPredictor


@pytest.fixture
def temp_path():
    """Create temporary directory for predictor outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    return df


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    return df


class TestCustomCVWithTabularPredictor:
    """Test custom CV integration with TabularPredictor."""

    def test_forward_chaining_cv_basic(self, temp_path, sample_classification_data):
        """Test basic forward chaining CV with TabularPredictor."""
        df = sample_classification_data
        
        # Generate forward chaining CV matrix
        cv_matrix = forward_chaining_cv(n_samples=len(df), n_folds=3, gap=5)
        
        # Train with custom CV
        predictor = TabularPredictor(
            label='target',
            path=os.path.join(temp_path, 'test_fc_cv')
        )
        
        predictor.fit(
            df,
            custom_cv_matrix=cv_matrix,
            hyperparameters={'GBM': {}},  # Only GBM for speed
            time_limit=30,
            verbosity=0
        )
        
        # Verify predictor works
        assert len(predictor.model_names()) > 0
        assert predictor.model_best is not None
        
        # Make predictions
        predictions = predictor.predict(df.head(10))
        assert len(predictions) == 10
        assert all(isinstance(p, (int, np.integer)) for p in predictions)

    def test_sliding_window_cv(self, temp_path, sample_regression_data):
        """Test sliding window CV with TabularPredictor."""
        df = sample_regression_data
        
        # Generate sliding window CV matrix
        cv_matrix = sliding_window_cv(
            n_samples=len(df),
            n_folds=3,
            window_size=100,
            step_size=50,
            gap=5
        )
        
        # Train with custom CV
        predictor = TabularPredictor(
            label='target',
            path=os.path.join(temp_path, 'test_sw_cv')
        )
        
        predictor.fit(
            df,
            custom_cv_matrix=cv_matrix,
            hyperparameters={'GBM': {}},
            time_limit=30,
            verbosity=0
        )
        
        # Verify predictor works
        assert len(predictor.model_names()) > 0
        
        # Make predictions
        predictions = predictor.predict(df.head(10))
        assert len(predictions) == 10

    def test_custom_cv_conflict_detection(self, temp_path, sample_classification_data):
        """Test that conflicting parameters are detected."""
        df = sample_classification_data
        cv_matrix = forward_chaining_cv(n_samples=len(df), n_folds=3)
        
        predictor = TabularPredictor(
            label='target',
            path=os.path.join(temp_path, 'test_conflict')
        )
        
        # Should raise ValueError when both custom_cv_matrix and num_bag_folds are specified
        with pytest.raises(ValueError, match="Cannot specify both"):
            predictor.fit(
                df,
                custom_cv_matrix=cv_matrix,
                num_bag_folds=5,  # Conflict!
                time_limit=10,
                verbosity=0
            )

    def test_custom_cv_shape_validation(self, temp_path, sample_classification_data):
        """Test that shape mismatches are detected."""
        df = sample_classification_data
        
        # Create CV matrix with wrong size
        wrong_cv_matrix = forward_chaining_cv(n_samples=100, n_folds=3)  # Wrong size!
        
        predictor = TabularPredictor(
            label='target',
            path=os.path.join(temp_path, 'test_shape')
        )
        
        # Should raise ValueError for shape mismatch
        with pytest.raises(ValueError, match="has.*samples.*but train_data has"):
            predictor.fit(
                df,
                custom_cv_matrix=wrong_cv_matrix,
                time_limit=10,
                verbosity=0
            )

    def test_custom_cv_auto_configures_num_bag_folds(
        self, temp_path, sample_classification_data
    ):
        """Test that num_bag_folds is automatically set from CV matrix."""
        df = sample_classification_data
        
        # Create CV matrix with 5 folds
        cv_matrix = forward_chaining_cv(n_samples=len(df), n_folds=5)
        
        predictor = TabularPredictor(
            label='target',
            path=os.path.join(temp_path, 'test_auto_config')
        )
        
        # Should work without specifying num_bag_folds
        predictor.fit(
            df,
            custom_cv_matrix=cv_matrix,
            hyperparameters={'GBM': {}},
            time_limit=30,
            verbosity=0
        )
        
        # Verify it worked
        assert len(predictor.model_names()) > 0

    def test_custom_cv_no_data_leakage(self, temp_path, sample_regression_data):
        """Verify that custom CV prevents data leakage in temporal data."""
        df = sample_regression_data
        
        # Create forward chaining CV
        cv_matrix = forward_chaining_cv(n_samples=len(df), n_folds=5, gap=5)
        
        # Verify temporal ordering
        splitter = CustomCVSplitter(cv_matrix)
        splits = list(splitter.split(df.drop(columns='target'), df['target']))
        
        for train_idx, test_idx in splits:
            # In forward chaining, max train index should be less than min test index
            assert max(train_idx) < min(test_idx), \
                "Data leakage detected: test samples appear before training samples"

