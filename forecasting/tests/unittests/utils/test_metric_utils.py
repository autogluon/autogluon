import pytest

from autogluon.forecasting.utils.metric_utils import (
    METRIC_COEFFICIENTS,
    AVAILABLE_METRICS,
    DEFAULT_METRIC,
    check_get_evaluation_metric,
)


def test_available_metrics_have_coefficients():
    for m in AVAILABLE_METRICS:
        assert METRIC_COEFFICIENTS[m]


@pytest.mark.parametrize(
    "check_input, expected_output",
    [
        (None, DEFAULT_METRIC),
    ]
    + [(k, k) for k in AVAILABLE_METRICS],
)
@pytest.mark.parametrize("raise_errors", [True, False])
def test_given_correct_input_check_get_eval_metric_output_correct(
    check_input, expected_output, raise_errors
):
    assert expected_output == check_get_evaluation_metric(
        check_input, raise_if_not_available=raise_errors
    )


@pytest.mark.parametrize("raise_errors", [True, False])
def test_given_no_input_check_get_eval_metric_output_default(raise_errors):
    assert DEFAULT_METRIC == check_get_evaluation_metric(
        raise_if_not_available=raise_errors
    )


def test_given_unavailable_input_and_raise_check_get_eval_metric_raises():
    with pytest.raises(ValueError):
        check_get_evaluation_metric(
            "some_nonsense_eval_metric", raise_if_not_available=True
        )


def test_given_unavailable_input_and_no_raise_check_get_eval_metric_output_default():
    assert DEFAULT_METRIC == check_get_evaluation_metric(
        "some_nonsense_eval_metric", raise_if_not_available=False
    )
