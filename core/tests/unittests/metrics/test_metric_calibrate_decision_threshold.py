from typing import Tuple

import numpy as np
import pytest

from autogluon.core.constants import BINARY
from autogluon.core.metrics import METRICS, Scorer
from autogluon.core.utils import get_pred_from_proba

BINARY_METRICS = list(METRICS[BINARY].keys())
EPSILON = 1e-6


@pytest.mark.parametrize("metric", BINARY_METRICS, ids=BINARY_METRICS)  # noqa
def test_metrics_calibrate_decision_threshold_custom_simple(metric: str):
    metric = METRICS[BINARY][metric]
    for reverse in [True, False]:
        y_true, y_pred_proba = fetch_example_y_pred_proba_binary_simple(reverse=reverse)
        _assert_calibration_decision_threshold(metric=metric, y_true=y_true, y_pred_proba=y_pred_proba)


@pytest.mark.parametrize("metric", BINARY_METRICS, ids=BINARY_METRICS)  # noqa
def test_metrics_calibrate_decision_threshold_custom_only_pos(metric: str):
    metric = METRICS[BINARY][metric]
    for reverse in [True, False]:
        y_true, y_pred_proba = fetch_example_y_pred_proba_binary_only_pos(reverse=reverse)
        _assert_calibration_decision_threshold(metric=metric, y_true=y_true, y_pred_proba=y_pred_proba)


@pytest.mark.parametrize("metric", BINARY_METRICS, ids=BINARY_METRICS)  # noqa
def test_metrics_calibrate_decision_threshold_custom_only_neg(metric: str):
    metric = METRICS[BINARY][metric]
    for reverse in [True, False]:
        y_true, y_pred_proba = fetch_example_y_pred_proba_binary_only_neg(reverse=reverse)
        _assert_calibration_decision_threshold(metric=metric, y_true=y_true, y_pred_proba=y_pred_proba)


@pytest.mark.parametrize("metric", BINARY_METRICS, ids=BINARY_METRICS)  # noqa
def test_metrics_calibrate_decision_threshold_custom_max(metric: str):
    metric = METRICS[BINARY][metric]
    for reverse in [True, False]:
        y_true, y_pred_proba = fetch_example_y_pred_proba_binary_max(reverse=reverse)
        _assert_calibration_decision_threshold(metric=metric, y_true=y_true, y_pred_proba=y_pred_proba)


@pytest.mark.parametrize("metric", BINARY_METRICS, ids=BINARY_METRICS)  # noqa
def test_metrics_calibrate_decision_threshold_custom_with_duplicates(metric: str):
    metric = METRICS[BINARY][metric]
    for reverse in [True, False]:
        y_true, y_pred_proba = fetch_example_y_pred_proba_binary_with_duplicates_v1(reverse=reverse)
        _assert_calibration_decision_threshold(metric=metric, y_true=y_true, y_pred_proba=y_pred_proba)

        y_true, y_pred_proba = fetch_example_y_pred_proba_binary_with_duplicates_v2(reverse=reverse)
        _assert_calibration_decision_threshold(metric=metric, y_true=y_true, y_pred_proba=y_pred_proba)

        y_true, y_pred_proba = fetch_example_y_pred_proba_binary_with_duplicates_v3(reverse=reverse)
        _assert_calibration_decision_threshold(metric=metric, y_true=y_true, y_pred_proba=y_pred_proba)

        y_true, y_pred_proba = fetch_example_y_pred_proba_binary_with_duplicates_v4(reverse=reverse)
        _assert_calibration_decision_threshold(metric=metric, y_true=y_true, y_pred_proba=y_pred_proba)

        y_true, y_pred_proba = fetch_example_y_pred_proba_binary_with_duplicates_v5(reverse=reverse)
        _assert_calibration_decision_threshold(metric=metric, y_true=y_true, y_pred_proba=y_pred_proba)


@pytest.mark.parametrize("metric", BINARY_METRICS, ids=BINARY_METRICS)  # noqa
def test_metrics_calibrate_decision_threshold_custom_real(metric: str):
    """
    Test calibration on 100 real predictions and ground truths from the AdultIncome dataset
    """
    metric = METRICS[BINARY][metric]
    for reverse in [True, False]:
        y_true, y_pred_proba = fetch_example_y_pred_proba_binary_real(reverse=reverse)
        _assert_calibration_decision_threshold(metric=metric, y_true=y_true, y_pred_proba=y_pred_proba)


def _assert_calibration_decision_threshold(metric: Scorer, y_true: np.ndarray, y_pred_proba: np.ndarray):
    y_true_og = y_true.copy()
    y_pred_proba_og = y_pred_proba.copy()

    if not metric.needs_class:
        # Check that an AssertionError is raised if called on an incompatible metric
        with pytest.raises(AssertionError):
            metric.calibrate_decision_threshold(y_true, y_pred_proba)
        return

    threshold_custom = metric.calibrate_decision_threshold(y_true, y_pred_proba)

    # Verify that inputs have not been mutated
    assert np.array_equal(y_true, y_true_og)
    assert np.array_equal(y_pred_proba, y_pred_proba_og)

    # Verify valid threshold
    assert threshold_custom >= 0
    assert threshold_custom <= 1

    y_pred_threshold_custom = get_pred_from_proba(
        y_pred_proba=y_pred_proba,
        problem_type=BINARY,
        decision_threshold=threshold_custom,
    )

    score_threshold_custom = metric.score(y_true=y_true, y_pred=y_pred_threshold_custom)

    if metric._calibration_func is not None:
        threshold_naive = metric._calibrate_decision_threshold_default(y_true, y_pred_proba)

        # Verify that inputs have not been mutated
        assert np.array_equal(y_true, y_true_og)
        assert np.array_equal(y_pred_proba, y_pred_proba_og)

        # Verify valid threshold
        assert threshold_naive >= 0
        assert threshold_naive <= 1

        y_pred_threshold_naive = get_pred_from_proba(
            y_pred_proba=y_pred_proba,
            problem_type=BINARY,
            decision_threshold=threshold_naive,
        )
        score_threshold_naive = metric.score(y_true=y_true, y_pred=y_pred_threshold_naive)
        if score_threshold_custom < (score_threshold_naive - EPSILON):
            raise AssertionError(
                f"Custom threshold logic underperformed naive threshold logic for metric '{metric.name}'! This should never happen.\n"
                f"(score_threshold_custom={score_threshold_custom}, score_threshold_naive={score_threshold_naive}, "
                f"threshold_custom={threshold_custom}, threshold_naive={threshold_naive})"
            )

    threshold_default = 0.5
    y_pred_threshold_default = get_pred_from_proba(
        y_pred_proba=y_pred_proba,
        problem_type=BINARY,
        decision_threshold=threshold_default,
    )
    score_threshold_default = metric.score(y_true=y_true, y_pred=y_pred_threshold_default)
    if score_threshold_custom < (score_threshold_default - EPSILON):
        raise AssertionError(
            f"Custom threshold logic underperformed default threshold for metric '{metric.name}'! This should never happen.\n"
            f"(score_threshold_custom={score_threshold_custom}, score_threshold_default={score_threshold_default}, "
            f"threshold_custom={threshold_custom}, threshold_default={threshold_default})"
        )


def fetch_example_y_pred_proba_binary_real(reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.array(
        [
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
        ]
    )
    y_pred_proba = np.array(
        [
            0.09988541,
            0.0009847643,
            0.9885485,
            0.002805695,
            0.00066008844,
            0.8613145,
            0.9982843,
            0.4091767,
            0.0015445104,
            0.0538882,
            0.028213877,
            0.002600598,
            0.01051364,
            0.47404587,
            0.04643202,
            0.2615043,
            0.4132154,
            0.8364282,
            0.8728589,
            0.000566786,
            0.29402304,
            0.0005646895,
            0.07163984,
            0.002832665,
            0.05579408,
            0.03324698,
            0.020024087,
            0.017691828,
            0.3144096,
            0.0026646908,
            0.0074746087,
            0.0026268153,
            0.15745641,
            0.043211184,
            0.2892855,
            0.57701963,
            0.042080633,
            0.0003402038,
            0.11204968,
            0.18905386,
            0.0027627628,
            0.023044664,
            0.10830529,
            0.4354616,
            0.029509429,
            0.0006289021,
            0.15105118,
            0.0026147524,
            0.054255098,
            0.010514567,
            0.17059574,
            0.0005426764,
            0.12966208,
            0.54238737,
            0.0025665036,
            0.7208377,
            0.99421173,
            0.0067156744,
            0.0029254658,
            0.00041716083,
            0.098347634,
            0.36101213,
            0.13895252,
            0.9962068,
            0.18807508,
            0.16711129,
            0.9966102,
            0.0136143975,
            0.01868336,
            0.24376073,
            0.022078237,
            0.06165174,
            0.0746088,
            0.04368908,
            0.0013534917,
            0.0069359792,
            0.014623916,
            0.03306944,
            0.9356479,
            0.0004658401,
            0.009022627,
            0.9867372,
            0.008184292,
            0.9969599,
            0.07384454,
            0.8396382,
            0.043205388,
            0.7896248,
            0.11838616,
            0.005467606,
            0.0017219092,
            0.0048511866,
            0.23898569,
            0.027551392,
            0.016499963,
            0.26120213,
            0.9600574,
            0.0053676404,
            0.11447967,
            0.15039557,
        ]
    )
    if reverse:
        y_true = np.flip(y_true)
        y_pred_proba = np.flip(y_pred_proba)
    return y_true, y_pred_proba


def fetch_example_y_pred_proba_binary_simple(reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0, 1, 1, 0, 0, 1])
    y_pred_proba = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    if reverse:
        y_true = np.flip(y_true)
        y_pred_proba = np.flip(y_pred_proba)
    return y_true, y_pred_proba


def fetch_example_y_pred_proba_binary_max(reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.array([1, 0])
    y_pred_proba = np.array([1.0, 0.0])
    if reverse:
        y_true = np.flip(y_true)
        y_pred_proba = np.flip(y_pred_proba)
    return y_true, y_pred_proba


def fetch_example_y_pred_proba_binary_only_pos(reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.array([1, 1])
    y_pred_proba = np.array([1.0, 0.0])
    if reverse:
        y_true = np.flip(y_true)
        y_pred_proba = np.flip(y_pred_proba)
    return y_true, y_pred_proba


def fetch_example_y_pred_proba_binary_only_neg(reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0, 0])
    y_pred_proba = np.array([1.0, 0.0])
    if reverse:
        y_true = np.flip(y_true)
        y_pred_proba = np.flip(y_pred_proba)
    return y_true, y_pred_proba


def fetch_example_y_pred_proba_binary_with_duplicates_v1(reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Duplicate y_pred_proba edge case where majority of duplicates are class 0 and duplicate y_pred_proba starts with class 1
    """
    y_true = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred_proba = np.array([1.0, 0.8, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4, 0.0, 0.0])
    if reverse:
        y_true = np.flip(y_true)
        y_pred_proba = np.flip(y_pred_proba)
    return y_true, y_pred_proba


def fetch_example_y_pred_proba_binary_with_duplicates_v2(reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as v1 but duplicate y_pred_proba contain majority class 1
    """
    y_true = np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    y_pred_proba = np.array([1.0, 0.8, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4, 0.0, 0.0])
    if reverse:
        y_true = np.flip(y_true)
        y_pred_proba = np.flip(y_pred_proba)
    return y_true, y_pred_proba


def fetch_example_y_pred_proba_binary_with_duplicates_v3(reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as v1 but duplicate y_pred_proba starts with class 0
    """
    y_true = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 0])
    y_pred_proba = np.array([1.0, 0.8, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4, 0.0, 0.0])
    if reverse:
        y_true = np.flip(y_true)
        y_pred_proba = np.flip(y_pred_proba)
    return y_true, y_pred_proba


def fetch_example_y_pred_proba_binary_with_duplicates_v4(reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as v2 but duplicate y_pred_proba starts with class 0
    """
    y_true = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0])
    y_pred_proba = np.array([1.0, 0.8, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4, 0.0, 0.0])
    if reverse:
        y_true = np.flip(y_true)
        y_pred_proba = np.flip(y_pred_proba)
    return y_true, y_pred_proba


def fetch_example_y_pred_proba_binary_with_duplicates_v5(reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as v4 but the duplicates are the lowest y_pred_proba values
    """
    y_true = np.array([0, 1, 1, 1, 0, 1, 1, 1])
    y_pred_proba = np.array([1.0, 0.8, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4])
    if reverse:
        y_true = np.flip(y_true)
        y_pred_proba = np.flip(y_pred_proba)
    return y_true, y_pred_proba
