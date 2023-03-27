from math import isclose

import numpy as np
import pytest
import sklearn

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE
from autogluon.core.metrics import METRICS, Scorer, rmse_func


BINARY_METRICS = list(METRICS[BINARY].keys())
MULTICLASS_METRICS = list(METRICS[MULTICLASS].keys())
REGRESSION_METRICS = list(METRICS[REGRESSION].keys())

EXPECTED_BINARY_METRICS = {
    "acc",
    "accuracy",
    "average_precision",
    "balanced_accuracy",
    "f1",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "log_loss",
    "mcc",
    "nll",
    "pac",
    "pac_score",
    "precision",
    "precision_macro",
    "precision_micro",
    "precision_weighted",
    "quadratic_kappa",
    "recall",
    "recall_macro",
    "recall_micro",
    "recall_weighted",
    "roc_auc",
    "roc_auc_ovo_macro",
}

EXPECTED_MULTICLASS_METRICS = {
    "acc",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "log_loss",
    "mcc",
    "nll",
    "pac",
    "pac_score",
    "precision_macro",
    "precision_micro",
    "precision_weighted",
    "quadratic_kappa",
    "recall_macro",
    "recall_micro",
    "recall_weighted",
    "roc_auc_ovo_macro",
}

EXPECTED_REGRESSION_METRICS = {
    "mae",
    "mape",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "median_absolute_error",
    "mse",
    "pearsonr",
    "r2",
    "rmse",
    "root_mean_squared_error",
    "spearmanr",
}

EXPECTED_QUANTILE_METRICS = {
    "pinball",
    "pinball_loss",
}


@pytest.mark.parametrize("metrics,expected_metrics_and_aliases", [
    [METRICS[BINARY], EXPECTED_BINARY_METRICS],
    [METRICS[MULTICLASS], EXPECTED_MULTICLASS_METRICS],
    [METRICS[REGRESSION], EXPECTED_REGRESSION_METRICS],
    [METRICS[QUANTILE], EXPECTED_QUANTILE_METRICS],
], ids=[
    BINARY,
    MULTICLASS,
    REGRESSION,
    QUANTILE,
])  # noqa
def test_metric_exists(metrics: dict, expected_metrics_and_aliases: set):
    """
    Ensure all expected metrics are present and no unexpected metrics are present
    """
    seen_metrics = set()

    for metric_name, metric_obj in metrics.items():
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


@pytest.mark.parametrize("sample_weight",
                        [None,
                        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3])])
def test_rmse_with_sklearn(sample_weight):
    """
    Ensure 
    (1) Without sample_weight, AutoGluon's custom rmse produces the same result as sklearn's rmse
    (2) With sample_weight, computed wrmse is as expected
    """
    y_true = np.array([0, 0, 1, 1, 1, 1, 0])
    y_pred = np.array([0.13, 0.09, 0.78, 0.43, 0.8, 0.91, 0.32])
    expected_rmse = sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False, sample_weight=sample_weight)

    kwargs = {"y_true": y_true, "y_pred": y_pred}
    if sample_weight is not None: kwargs["sample_weight"] = sample_weight
    computed_rmse = rmse_func(**kwargs)

    assert np.isclose(computed_rmse, expected_rmse)