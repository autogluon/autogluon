from math import isclose

import numpy as np
import pytest
import sklearn

from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION
from autogluon.core.metrics import METRICS, Scorer, make_scorer, rmse_func

METRICS_NEEDS_CLASS = {}
METRICS_NEEDS_PROBA = {}
METRICS_NEEDS_THRESHOLD = {}
NOT_METRICS_NEEDS_THRESHOLD = {}
for problem_type in METRICS:
    METRICS_NEEDS_CLASS[problem_type] = [k for k, v in METRICS[problem_type].items() if v.needs_class]
    METRICS_NEEDS_PROBA[problem_type] = [k for k, v in METRICS[problem_type].items() if v.needs_proba or v.needs_threshold]
    METRICS_NEEDS_THRESHOLD[problem_type] = [k for k, v in METRICS[problem_type].items() if v.needs_threshold]
    NOT_METRICS_NEEDS_THRESHOLD[problem_type] = [k for k, v in METRICS[problem_type].items() if not v.needs_threshold]

BINARY_METRICS = list(METRICS[BINARY].keys())
MULTICLASS_METRICS = list(METRICS[MULTICLASS].keys())
REGRESSION_METRICS = list(METRICS[REGRESSION].keys())

BINARY_METRICS_NEEDS_POS_LABEL = [k for k, v in METRICS[BINARY].items() if v.needs_pos_label]

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
    "roc_auc_ovo",
    "roc_auc_ovo_macro",
    "roc_auc_ovo_weighted",
    "roc_auc_ovr",
    "roc_auc_ovr_macro",
    "roc_auc_ovr_micro",
    "roc_auc_ovr_weighted",
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
    "smape",
    "spearmanr",
    "symmetric_mean_absolute_percentage_error",
}

EXPECTED_QUANTILE_METRICS = {
    "pinball",
    "pinball_loss",
}


@pytest.mark.parametrize(
    "metrics,expected_metrics_and_aliases",
    [
        [METRICS[BINARY], EXPECTED_BINARY_METRICS],
        [METRICS[MULTICLASS], EXPECTED_MULTICLASS_METRICS],
        [METRICS[REGRESSION], EXPECTED_REGRESSION_METRICS],
        [METRICS[QUANTILE], EXPECTED_QUANTILE_METRICS],
    ],
    ids=[
        BINARY,
        MULTICLASS,
        REGRESSION,
        QUANTILE,
    ],
)  # noqa
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
            raise AssertionError(f"Missing metrics: {list(missing_metrics)}")
        else:
            unknown_metrics = seen_metrics.difference(expected_metrics_and_aliases)
            raise AssertionError(
                f"Invalid metrics (If you have added a new metric, "
                f"please include it in the variable `expected_metrics_and_aliases`):"
                f"\n\t{list(unknown_metrics)}"
            )


@pytest.mark.parametrize("metric", BINARY_METRICS, ids=BINARY_METRICS)  # noqa
def test_metrics_perfect_binary(metric: str):
    _assert_valid_scorer_classifier(scorer=METRICS[BINARY][metric])
    _assert_perfect_score(scorer=METRICS[BINARY][metric])


@pytest.mark.parametrize("metric", MULTICLASS_METRICS, ids=MULTICLASS_METRICS)  # noqa
def test_metrics_perfect_multiclass(metric: str):
    _assert_valid_scorer_classifier(scorer=METRICS[MULTICLASS][metric])
    _assert_perfect_score(scorer=METRICS[MULTICLASS][metric])


@pytest.mark.skip(reason="average_precision doesn't raise an exception here when it should, so this test currently fails")
@pytest.mark.parametrize("metric", METRICS_NEEDS_THRESHOLD[BINARY], ids=METRICS_NEEDS_THRESHOLD[BINARY])  # noqa
def test_metrics_perfect_raises_binary_single_sample(metric: str):
    with pytest.raises(Exception):
        # threshold metrics should not be able to predict on a single sample
        _assert_perfect_score_single_sample(scorer=METRICS[BINARY][metric])


@pytest.mark.skip(reason="mcc, quadradic_kappa, and pac produce unexpected values here, so this test currently fails")
@pytest.mark.parametrize("metric", NOT_METRICS_NEEDS_THRESHOLD[BINARY], ids=NOT_METRICS_NEEDS_THRESHOLD[BINARY])  # noqa
def test_metrics_perfect_binary_single_sample(metric: str):
    _assert_perfect_score_single_sample(scorer=METRICS[BINARY][metric])


@pytest.mark.skip(reason="mcc and quadradic_kappa produce unexpected values here, so this test currently fails")
@pytest.mark.parametrize("metric", METRICS_NEEDS_CLASS[MULTICLASS], ids=METRICS_NEEDS_CLASS[MULTICLASS])  # noqa
def test_metrics_perfect_multiclass_single_sample(metric: str):
    _assert_perfect_score_single_sample(scorer=METRICS[MULTICLASS][metric])


@pytest.mark.parametrize("metric", METRICS_NEEDS_CLASS[MULTICLASS], ids=METRICS_NEEDS_CLASS[MULTICLASS])  # noqa
def test_metrics_perfect_str_multiclass(metric: str):
    scorer = METRICS[MULTICLASS][metric]
    _assert_perfect_score_str_binary(scorer=scorer)
    _assert_perfect_score_str_multiclass(scorer=scorer)


@pytest.mark.parametrize("metric", METRICS_NEEDS_CLASS[BINARY], ids=METRICS_NEEDS_CLASS[BINARY])  # noqa
def test_metrics_perfect_str_binary(metric: str):
    scorer = METRICS[BINARY][metric]
    if metric not in BINARY_METRICS_NEEDS_POS_LABEL:
        _assert_perfect_score_str_binary(scorer=scorer)
    else:
        with pytest.raises(ValueError):
            # pos_label should raise exception when passed string values
            _assert_perfect_score_str_binary(scorer=scorer)


@pytest.mark.parametrize("metric", METRICS_NEEDS_PROBA[BINARY], ids=METRICS_NEEDS_PROBA[BINARY])  # noqa
def test_metrics_perfect_proba_raises_str_binary(metric: str):
    scorer = METRICS[BINARY][metric]
    with pytest.raises(Exception):
        # proba metrics should fail with string inputs
        scorer(np.array(["a", "a", "b"]), np.array(["a", "a", "b"]))


@pytest.mark.parametrize("metric", METRICS_NEEDS_PROBA[MULTICLASS], ids=METRICS_NEEDS_PROBA[MULTICLASS])  # noqa
def test_metrics_perfect_proba_raises_str_multiclass(metric: str):
    scorer = METRICS[MULTICLASS][metric]
    with pytest.raises(Exception):
        # proba metrics should fail with string inputs
        scorer(np.array(["a", "a", "b"]), np.array(["a", "a", "b"]))


@pytest.mark.parametrize("metric", REGRESSION_METRICS, ids=REGRESSION_METRICS)  # noqa
def test_metrics_perfect_regression(metric: str):
    _assert_valid_scorer_regressor(scorer=METRICS[REGRESSION][metric])
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


@pytest.mark.parametrize("metric", METRICS_NEEDS_CLASS[BINARY], ids=METRICS_NEEDS_CLASS[BINARY])  # noqa
def test_metrics_imperfect_str_binary(metric: str):
    if metric not in BINARY_METRICS_NEEDS_POS_LABEL:
        _assert_imperfect_score_str_binary(scorer=METRICS[BINARY][metric])
    else:
        with pytest.raises(ValueError):
            # pos_label should raise exception when passed string values
            _assert_imperfect_score_str_binary(scorer=METRICS[BINARY][metric])


@pytest.mark.parametrize("metric", METRICS_NEEDS_CLASS[MULTICLASS], ids=METRICS_NEEDS_CLASS[MULTICLASS])  # noqa
def test_metrics_imperfect_str_multiclass(metric: str):
    _assert_imperfect_score_str_multiclass(scorer=METRICS[MULTICLASS][metric])
    _assert_imperfect_score_str_binary(scorer=METRICS[MULTICLASS][metric])


def _assert_valid_scorer_classifier(scorer: Scorer):
    _assert_valid_scorer(scorer=scorer)
    num_true = sum([1 if needs else 0 for needs in [scorer.needs_proba, scorer.needs_threshold, scorer.needs_class]])
    if num_true != 1:
        raise AssertionError(
            f"Classification scorer '{scorer.name}' (class={scorer.__class__.__name__}) has invalid needs (exactly 1 must be True): "
            f"(needs_proba={scorer.needs_proba}, needs_threshold={scorer.needs_threshold}, needs_class={scorer.needs_class})"
        )


def _assert_valid_scorer_regressor(scorer: Scorer):
    _assert_valid_scorer(scorer=scorer)
    num_true = sum([1 if needs else 0 for needs in [scorer.needs_pred, scorer.needs_quantile]])
    if num_true != 1:
        raise AssertionError(
            f"Regression scorer '{scorer.name}' (class={scorer.__class__.__name__}) has invalid needs (exactly 1 must be True): "
            f"(needs_pred={scorer.needs_pred}, needs_quantile={scorer.needs_quantile})"
        )


def _assert_valid_scorer(scorer: Scorer):
    if scorer.needs_class and not scorer.needs_pred:
        raise AssertionError(
            f"Invalid Scorer definition! If `needs_class=True`, then `needs_pred` must also be True. "
            f"(name={scorer.name}, needs_class={scorer.needs_class}, needs_pred={scorer.needs_pred})"
        )


def _assert_perfect_score(scorer: Scorer, abs_tol=1e-5):
    """
    Ensure a perfect prediction has an error of 0
    and a score equal to scorer's optimum for a given scorer.
    """
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0.0, 0.0, 1.0])
    _assert_perfect_score_generic(scorer=scorer, abs_tol=abs_tol, y_true=y_true, y_pred=y_pred)


def _assert_perfect_score_single_sample(scorer: Scorer, abs_tol=1e-5):
    """
    Ensure a perfect prediction has an error of 0 when given only a single sample
    and a score equal to scorer's optimum for a given scorer.
    """
    y_true = np.array([1])
    y_pred = np.array([1.0])
    _assert_perfect_score_generic(scorer=scorer, abs_tol=abs_tol, y_true=y_true, y_pred=y_pred)


def _assert_perfect_score_generic(scorer: Scorer, y_true, y_pred, abs_tol=1e-5):
    """
    Ensure a perfect prediction has an error of 0
    and a score equal to scorer's optimum for a given scorer.
    """
    score = scorer(y_true, y_pred)
    assert score == scorer.score(y_true, y_pred)
    error = scorer.error(y_true, y_pred)
    assert error == scorer.convert_score_to_error(score)
    assert isclose(score, scorer.convert_error_to_score(error), abs_tol=abs_tol)
    assert isclose(error, 0, abs_tol=abs_tol)
    assert isclose(score, scorer.optimum, abs_tol=abs_tol)


def _assert_perfect_score_str_binary(scorer: Scorer, abs_tol=1e-5):
    """
    Ensure a perfect prediction has an error of 0
    and a score equal to scorer's optimum for a given scorer.
    """
    y_true = np.array(["a", "a", "b"])
    y_pred = np.array(["a", "a", "b"])
    _assert_perfect_score_generic(scorer=scorer, abs_tol=abs_tol, y_true=y_true, y_pred=y_pred)


def _assert_perfect_score_str_multiclass(scorer: Scorer, abs_tol=1e-5):
    """
    Ensure a perfect prediction has an error of 0
    and a score equal to scorer's optimum for a given scorer.
    """
    y_true = np.array(["b", "a", "b", "c"])
    y_pred = np.array(["b", "a", "b", "c"])
    _assert_perfect_score_generic(scorer=scorer, abs_tol=abs_tol, y_true=y_true, y_pred=y_pred)


def _assert_imperfect_score(scorer: Scorer, abs_tol: float = 1e-5) -> float:
    """
    Ensure an imperfect prediction has an error greater than 0
    and a score less than the scorer's optimum for a given scorer.
    """
    y_true = np.array([0, 0, 1])
    y_pred = np.array([1.0, 1.0, 0.0])
    return _assert_imperfect_score_generic(scorer=scorer, y_true=y_true, y_pred=y_pred, abs_tol=abs_tol)


def _assert_imperfect_score_str_binary(scorer: Scorer, abs_tol: float = 1e-5):
    """
    Ensure an imperfect prediction has an error greater than 0
    and a score less than the scorer's optimum for a given scorer.

    Also ensure that both numeric and string representations of the input get the same score without raising an exception.
    """
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0.0, 1.0, 0.0])
    score_numeric = _assert_imperfect_score_generic(scorer=scorer, y_true=y_true, y_pred=y_pred, abs_tol=abs_tol)

    y_true = np.array(["b", "b", "a"])
    y_pred = np.array(["b", "a", "b"])
    score_str = _assert_imperfect_score_generic(scorer=scorer, y_true=y_true, y_pred=y_pred, abs_tol=abs_tol)

    assert isclose(score_numeric, score_str, abs_tol=abs_tol)


def _assert_imperfect_score_str_multiclass(scorer: Scorer, abs_tol: float = 1e-5):
    """
    Ensure an imperfect prediction has an error greater than 0
    and a score less than the scorer's optimum for a given scorer.

    Also ensure that both numeric and string representations of the input get the same score without raising an exception.
    """
    y_true = np.array([1, 0, 1, 2, 2, 4])
    y_pred = np.array([1.0, 1.0, 0.0, 0.0, 3.0, 1.0])
    score_numeric = _assert_imperfect_score_generic(scorer=scorer, y_true=y_true, y_pred=y_pred, abs_tol=abs_tol)

    y_true = np.array(["b", "a", "b", "c", "c", "e"])
    y_pred = np.array(["b", "b", "a", "a", "d", "b"])
    score_str = _assert_imperfect_score_generic(scorer=scorer, y_true=y_true, y_pred=y_pred, abs_tol=abs_tol)

    assert isclose(score_numeric, score_str, abs_tol=abs_tol)


def _assert_imperfect_score_generic(scorer: Scorer, y_true, y_pred, abs_tol: float = 1e-5) -> float:
    """
    Ensure an imperfect prediction has an error greater than 0
    and a score less than the scorer's optimum for a given scorer.
    """
    score = scorer(y_true, y_pred)
    assert score == scorer.score(y_true, y_pred)
    error = scorer.error(y_true, y_pred)
    assert error == scorer.convert_score_to_error(score)
    assert isclose(score, scorer.convert_error_to_score(error), abs_tol=abs_tol)
    assert error > 0
    assert score < scorer.optimum
    assert not isclose(error, 0, abs_tol=abs_tol)
    assert not isclose(score, scorer.optimum, abs_tol=abs_tol)
    return score


@pytest.mark.parametrize("sample_weight", [None, np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3])])
def test_rmse_with_sklearn(sample_weight):
    """
    Ensure
    (1) Without sample_weight, AutoGluon's custom rmse produces the same result as sklearn's rmse
    (2) With sample_weight, computed wrmse is as expected
    """
    y_true = np.array([0, 0, 1, 1, 1, 1, 0])
    y_pred = np.array([0.13, 0.09, 0.78, 0.43, 0.8, 0.91, 0.32])
    expected_rmse = sklearn.metrics.root_mean_squared_error(y_true, y_pred, sample_weight=sample_weight)

    kwargs = {"y_true": y_true, "y_pred": y_pred}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    computed_rmse = rmse_func(**kwargs)

    assert np.isclose(computed_rmse, expected_rmse)


def test_invalid_scorer():
    """
    Ensure various edge-cases are appropriately handled when Scorers are created incorrectly
    """
    with pytest.raises(ValueError):
        # Invalid: Specifying multiple needs_*
        make_scorer("dummy", score_func=sklearn.metrics.accuracy_score, needs_proba=True, needs_class=True)

    with pytest.raises(ValueError):
        # Invalid: Specifying False for all needs_*
        make_scorer("dummy", score_func=sklearn.metrics.accuracy_score, needs_pred=False)

    with pytest.raises(ValueError):
        # Invalid: Specifying needs_pred=False when needs_class=True
        make_scorer("dummy", score_func=sklearn.metrics.accuracy_score, needs_pred=False, needs_class=True)

    # Valid
    make_scorer("dummy", score_func=sklearn.metrics.accuracy_score, needs_pred=True, needs_class=True)

    with pytest.raises(ValueError):
        # Invalid: Specifying needs_pred=True when needs_proba=True
        make_scorer("dummy", score_func=sklearn.metrics.accuracy_score, needs_pred=True, needs_proba=True)
