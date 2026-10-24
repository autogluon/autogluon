import numpy as np
import pytest

from autogluon.core.calibrate import calibrate_decision_threshold
from autogluon.core.metrics import balanced_accuracy, f1, roc_auc


def _get_sample_data():
    y = np.array(
        [
            1,
            0,
            1,
            1,
            1,
            0,
        ]
    )
    y_pred_proba = np.array(
        [
            0.0,
            0.24,
            0.25,
            0.25,
            0.5,
            1.0,
        ]
    )
    return y, y_pred_proba


def test_calibrate_decision_threshold():
    y, y_pred_proba = _get_sample_data()
    decision_threshold = calibrate_decision_threshold(
        y=y,
        y_pred_proba=y_pred_proba,
        metric=f1,
        decision_thresholds=50,
        secondary_decision_thresholds=None,
    )
    assert decision_threshold == 0.24

    decision_threshold = calibrate_decision_threshold(
        y=y,
        y_pred_proba=y_pred_proba,
        metric=f1,
    )
    assert decision_threshold == 0.249

    decision_threshold = calibrate_decision_threshold(
        y=y,
        y_pred_proba=y_pred_proba,
        metric=balanced_accuracy,
    )
    assert decision_threshold == 0.249

    decision_threshold = calibrate_decision_threshold(
        y=y,
        y_pred_proba=y_pred_proba,
        metric=balanced_accuracy,
        decision_thresholds=10,
    )
    assert decision_threshold == 1.0

    decision_threshold = calibrate_decision_threshold(
        y=y,
        y_pred_proba=y_pred_proba,
        metric=balanced_accuracy,
        decision_thresholds=[0.88],
    )
    assert decision_threshold == 0.88


def test_calibrate_decision_threshold_select_closer_to_0_5():
    """Test that calibration will choose the threshold closer to 0.5 in the case of a tie"""
    y, y_pred_proba = _get_sample_data()
    decision_threshold = calibrate_decision_threshold(
        y=y,
        y_pred_proba=y_pred_proba,
        metric=balanced_accuracy,
        decision_thresholds=[0.5, 0.244, 0.247],
        secondary_decision_thresholds=None,
    )
    assert decision_threshold == 0.247

    decision_threshold = calibrate_decision_threshold(
        y=y,
        y_pred_proba=y_pred_proba,
        metric=balanced_accuracy,
        decision_thresholds=[0.5, 0.247, 0.244],
        secondary_decision_thresholds=None,
    )
    assert decision_threshold == 0.247


def test_calibrate_decision_threshold_proba_metric_0_5():
    """Test that non-pred metrics will always return 0.5"""
    y, y_pred_proba = _get_sample_data()
    decision_threshold = calibrate_decision_threshold(
        y=y,
        y_pred_proba=y_pred_proba,
        metric=roc_auc,
    )
    assert decision_threshold == 0.5
    decision_threshold = calibrate_decision_threshold(
        y=y,
        y_pred_proba=y_pred_proba,
        metric=roc_auc,
        decision_thresholds=[0.1],
    )
    assert decision_threshold == 0.5


def test_calibrate_decision_threshold_out_of_bounds():
    y, y_pred_proba = _get_sample_data()
    with pytest.raises(ValueError):
        calibrate_decision_threshold(
            y=y,
            y_pred_proba=y_pred_proba,
            metric=balanced_accuracy,
            decision_thresholds=[1.0, 0.5, 2.0],
        )
    with pytest.raises(ValueError):
        calibrate_decision_threshold(
            y=y,
            y_pred_proba=y_pred_proba,
            metric=balanced_accuracy,
            decision_thresholds=[-0.01, 0.5],
        )


def test_calibrate_decision_threshold_invalid_args():
    y, y_pred_proba = _get_sample_data()
    with pytest.raises(AssertionError):
        calibrate_decision_threshold(
            y=y,
            y_pred_proba=y_pred_proba,
            metric=balanced_accuracy,
            decision_thresholds="invalid",
        )
    with pytest.raises(AssertionError):
        calibrate_decision_threshold(
            y=y,
            y_pred_proba=y_pred_proba,
            metric=balanced_accuracy,
            decision_thresholds=0.01,
        )
    with pytest.raises(AssertionError):
        calibrate_decision_threshold(
            y=y,
            y_pred_proba=y_pred_proba,
            metric=balanced_accuracy,
            decision_thresholds=None,
        )
    with pytest.raises(AssertionError):
        calibrate_decision_threshold(
            y=y,
            y_pred_proba=y_pred_proba,
            metric=balanced_accuracy,
            secondary_decision_thresholds=[0.2, 0.4],
        )


def test_calibrate_decision_threshold_sample_weight_with_subsample():
    """`sample_weight` is subsampled with `y`, so the threshold still optimizes the weighted metric."""
    from autogluon.core.metrics import accuracy

    # 50 heavy positives at 0.3 and 950 light negatives at 0.35: weighted accuracy calls for a threshold below 0.3,
    # unweighted accuracy for predicting every row negative
    y = np.array([1] * 50 + [0] * 950)
    y_pred_proba = np.array([0.3] * 50 + [0.35] * 950)
    sample_weight = np.array([100.0] * 50 + [1.0] * 950)

    threshold = calibrate_decision_threshold(
        y=y, y_pred_proba=y_pred_proba, metric=accuracy, sample_weight=sample_weight, subsample_size=500
    )
    assert threshold < 0.3
    threshold_unweighted = calibrate_decision_threshold(
        y=y, y_pred_proba=y_pred_proba, metric=accuracy, subsample_size=500
    )
    assert threshold_unweighted == 0.5
