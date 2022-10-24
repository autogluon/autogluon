from math import isclose

import numpy as np
import pytest

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import METRICS, Scorer


BINARY_METRICS = list(METRICS[BINARY].keys())
MULTICLASS_METRICS = list(METRICS[MULTICLASS].keys())
REGRESSION_METRICS = list(METRICS[REGRESSION].keys())


def test_metric_exists():
    """
    Ensure all expected metrics are present and no unexpected metrics are present
    """
    expected_metrics_and_aliases = {
        "accuracy",
        "acc",
        "balanced_accuracy",
        "mcc",
        "roc_auc_ovo_macro",
        "log_loss",
        "nll",
        "pac_score",
        "quadratic_kappa",
        "roc_auc",
        "average_precision",
        "precision",
        "precision_macro",
        "precision_micro",
        "precision_weighted",
        "recall",
        "recall_macro",
        "recall_micro",
        "recall_weighted",
        "f1",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "accuracy",
        "acc",
        "balanced_accuracy",
        "mcc",
        "roc_auc_ovo_macro",
        "log_loss",
        "nll",
        "pac_score",
        "quadratic_kappa",
        "precision_macro",
        "precision_micro",
        "precision_weighted",
        "recall_macro",
        "recall_micro",
        "recall_weighted",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "r2",
        "mean_squared_error",
        "mse",
        "root_mean_squared_error",
        "rmse",
        "mean_absolute_error",
        "mae",
        "median_absolute_error",
        "mean_absolute_percentage_error",
        "mape",
        "spearmanr",
        "pearsonr",
        "pinball_loss",
        "pinball",
    }

    seen_metrics = set()

    for problem_type in METRICS.keys():
        for metric_name, metric_obj in METRICS[problem_type].items():
            assert (metric_name == metric_obj.name) or (metric_name in metric_obj.alias)
            assert metric_obj.greater_is_better is True
            if metric_name not in seen_metrics:
                seen_metrics.add(metric_name)
    diff_metrics = seen_metrics.symmetric_difference(expected_metrics_and_aliases)
    if len(diff_metrics) > 0:
        missing_metrics = expected_metrics_and_aliases.difference(seen_metrics)
        if len(missing_metrics) > 0:
            raise AssertionError(f'Missing metrics: {list(missing_metrics)}')
        else:
            unknown_metrics = seen_metrics.difference(expected_metrics_and_aliases)
            raise AssertionError(f'Invalid metrics (If you have added a new metric, '
                                 f'please include it in the variable `expected_metrics_and_aliases`):'
                                 f'\n\t{list(unknown_metrics)}')


@pytest.mark.parametrize("metric", BINARY_METRICS, ids=BINARY_METRICS)  # noqa
def test_metrics_perfect_binary(metric: str):
    _assert_perfect_score(scorer=METRICS[BINARY][metric])


@pytest.mark.parametrize("metric", MULTICLASS_METRICS, ids=MULTICLASS_METRICS)  # noqa
def test_metrics_perfect_multiclass(metric: str):
    _assert_perfect_score(scorer=METRICS[MULTICLASS][metric])


@pytest.mark.parametrize("metric", REGRESSION_METRICS, ids=REGRESSION_METRICS)  # noqa
def test_metrics_perfect_regression(metric: str):
    _assert_perfect_score(scorer=METRICS[REGRESSION][metric])


@pytest.mark.parametrize("metric", BINARY_METRICS, ids=BINARY_METRICS)  # noqa
def test_metrics_imperfect_binary(metric: str):
    _assert_imperfect_score(scorer=METRICS[BINARY][metric])


@pytest.mark.parametrize("metric", MULTICLASS_METRICS, ids=MULTICLASS_METRICS)  # noqa
def test_metrics_imperfect_multiclass(metric: str):
    _assert_imperfect_score(scorer=METRICS[MULTICLASS][metric])


@pytest.mark.parametrize("metric", REGRESSION_METRICS, ids=REGRESSION_METRICS)  # noqa
def test_metrics_imperfect_regression(metric: str):
    _assert_imperfect_score(scorer=METRICS[REGRESSION][metric])


def _assert_perfect_score(scorer: Scorer, abs_tol=1e-5):
    """
    Ensure a perfect prediction has an error of 0
    and a score equal to scorer's optimum for a given scorer.
    """
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0.0, 0.0, 1.0])
    score = scorer(y_true, y_pred)
    assert score == scorer.score(y_true, y_pred)
    error = scorer.error(y_true, y_pred)
    assert error == scorer.convert_score_to_error(score)
    assert isclose(error, 0, abs_tol=abs_tol)
    assert isclose(score, scorer.optimum, abs_tol=abs_tol)


def _assert_imperfect_score(scorer: Scorer, abs_tol=1e-5):
    """
    Ensure an imperfect prediction has an error greater than 0
    and a score less than the scorer's optimum for a given scorer.
    """
    y_true = np.array([0, 0, 1])
    y_pred = np.array([1.0, 1.0, 0.0])
    score = scorer(y_true, y_pred)
    assert score == scorer.score(y_true, y_pred)
    error = scorer.error(y_true, y_pred)
    assert error == scorer.convert_score_to_error(score)
    assert error > 0
    assert score < scorer.optimum
    assert not isclose(error, 0, abs_tol=abs_tol)
    assert not isclose(score, scorer.optimum, abs_tol=abs_tol)
