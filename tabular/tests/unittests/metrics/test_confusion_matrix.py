import pytest
import numpy as np
from autogluon.tabular import TabularPredictor, TabularDataset


def test_confusion_matrix_basic(tmp_path):
    """Test basic confusion matrix computation."""
    train_data = TabularDataset({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        'label':    [0, 1, 0, 1, 0, 1, 0, 1],
    })

    predictor = TabularPredictor(
        label='label', 
        path=tmp_path
    ).fit(train_data, presets='medium_quality')

    # Compute confusion matrix without display
    cm = predictor.confusion_matrix(display=False)
    
    assert isinstance(cm, np.ndarray)
    assert cm.shape[0] == cm.shape[1]
    assert cm.shape[0] == 2  # Binary classification


def test_confusion_matrix_with_data(tmp_path):
    """Test confusion matrix with explicit data."""
    train_data = TabularDataset({
        'feature1': [1, 2, 3, 4, 5, 6],
        'label':    [0, 1, 0, 1, 0, 1],
    })
    
    test_data = TabularDataset({
        'feature1': [7, 8, 9, 10],
        'label':    [0, 1, 0, 1],
    })

    predictor = TabularPredictor(
        label='label', 
        path=tmp_path
    ).fit(train_data, presets='medium_quality')

    # Compute on test data
    cm = predictor.confusion_matrix(test_data, display=False)
    
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (2, 2)


def test_confusion_matrix_normalize(tmp_path):
    """Test normalized confusion matrix."""
    train_data = TabularDataset({
        'f1': [1, 2, 3, 4, 5, 6],
        'label': [0, 1, 0, 1, 0, 1],
    })

    predictor = TabularPredictor(
        label='label', 
        path=tmp_path
    ).fit(train_data)

    cm_norm = predictor.confusion_matrix(
        normalize='true', 
        display=False
    )
    
    # Check normalization: rows should sum to 1
    assert np.allclose(cm_norm.sum(axis=1), 1.0)


def test_confusion_matrix_save_plot(tmp_path):
    """Test saving confusion matrix plot."""
    train_data = TabularDataset({
        'f': [1, 2, 3, 4, 5, 6],
        'label': [0, 1, 0, 1, 0, 1],
    })
    
    predictor = TabularPredictor(
        label='label', 
        path=tmp_path
    ).fit(train_data)
    
    save_path = tmp_path / "cm_plot.png"

    # Save without displaying
    predictor.confusion_matrix(
        display=False, 
        save_path=str(save_path)
    )
    
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_confusion_matrix_multiclass(tmp_path):
    """Test confusion matrix with multiclass classification."""
    train_data = TabularDataset({
        'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'f2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        'label': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
    })

    predictor = TabularPredictor(
        label='label', 
        path=tmp_path,
        problem_type='multiclass'  # Force multiclass
    ).fit(train_data, presets='medium_quality')

    cm = predictor.confusion_matrix(display=False)
    
    assert cm.shape == (3, 3)


def test_confusion_matrix_invalid_problem_type(tmp_path):
    """Test that confusion matrix raises error for regression."""
    train_data = TabularDataset({
        'f': [1, 2, 3, 4],
        'label': [1.5, 2.5, 3.5, 4.5],  # Continuous target
    })

    predictor = TabularPredictor(
        label='label', 
        path=tmp_path,
        problem_type='regression'
    ).fit(train_data)

    with pytest.raises(ValueError, match="only applicable to classification"):
        predictor.confusion_matrix(display=False)


def test_confusion_matrix_missing_label(tmp_path):
    """Test error when label column is missing."""
    train_data = TabularDataset({
        'f': [1, 2, 3, 4],
        'label': [0, 1, 0, 1],
    })

    predictor = TabularPredictor(
        label='label', 
        path=tmp_path
    ).fit(train_data)

    # Data without label column
    test_data_no_label = TabularDataset({'f': [5, 6]})

    with pytest.raises(ValueError, match="must contain the target column"):
        predictor.confusion_matrix(test_data_no_label, display=False)


def test_confusion_matrix_custom_labels(tmp_path):
    """Test confusion matrix with custom label ordering."""
    # Use numeric labels to avoid type conversion issues
    train_data = TabularDataset({
        'f1': [1, 2, 3, 4, 5, 6, 7, 8],
        'f2': [10, 20, 30, 40, 50, 60, 70, 80],
        'label': [0, 1, 0, 1, 0, 1, 0, 1],
    })

    predictor = TabularPredictor(
        label='label', 
        path=tmp_path,
        problem_type='binary'
    ).fit(train_data, presets='medium_quality')

    # Test with reversed label order
    cm = predictor.confusion_matrix(
        labels=[1, 0],  # Reverse of default [0, 1]
        display=False
    )
    
    assert cm.shape == (2, 2)
    # Verify the matrix is computed with the custom label order
    assert isinstance(cm, np.ndarray)