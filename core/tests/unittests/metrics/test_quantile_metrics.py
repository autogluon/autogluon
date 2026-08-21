import numpy as np
import pytest

from autogluon.core.metrics.quantile_metrics import pinball_loss


def test_invalid_quantile_values_shape_raises():
    # Given
    input_target_values = [1.0, 2.0, 3.0]
    input_quantile_values = [1.0, 2.0, 3.0]
    input_quantile_levels = [0.25, 0.5, 0.75, 0.9, 0.95]

    # When-Then
    with pytest.raises(ValueError):
        observed_output = pinball_loss(input_target_values, input_quantile_values, input_quantile_levels)


def test_mismatched_target_prediction_length_raises():
    # Given
    input_target_values = [1.0, 2.0]
    input_quantile_values = [[1.0], [2.0], [3.0]]
    input_quantile_levels = [0.5, 0.75]

    # When/Then
    with pytest.raises(ValueError):
        observed_output = pinball_loss(input_target_values, input_quantile_values, input_quantile_levels)


def test_mismatched_quantiles_raises():
    # Given
    input_target_values = [1.0, 2.0]
    input_quantile_values = [[1.0], [2.0]]
    input_quantile_levels = [0.5, 0.75]

    # When/Then
    with pytest.raises(ValueError):
        observed_output = pinball_loss(input_target_values, input_quantile_values, input_quantile_levels)


def test_single_prediction():
    # Given
    input_target_values = [100]
    input_quantile_values = [[90.0]]
    input_quantile_levels = [0.9]
    expected_output = 9.0

    # When
    observed_output = pinball_loss(input_target_values, input_quantile_values, input_quantile_levels)

    # Then
    assert np.isclose(expected_output, observed_output)


def test_multiple_predictions():
    # Given
    input_target_values = [1.0, 2.0, 3.0]
    input_quantile_values = [[1.0, 1.1], [2.0, 1.9], [2.0, 4.0]]
    input_quantile_levels = [0.5, 0.75]
    expected_output = 0.425 / 3

    # When
    observed_output = pinball_loss(input_target_values, input_quantile_values, input_quantile_levels)

    # Then
    assert np.isclose(expected_output, observed_output)


def test_multiple_predictions_with_weights():
    # Given
    input_target_values = [1.0, 2.0, 3.0]
    input_quantile_values = [[1.0, 1.1], [2.0, 1.9], [2.0, 4.0]]
    input_quantile_levels = [0.5, 0.75]
    input_sample_weights = [0.25, 0.5, 0.25]
    input_quantile_weights = [0.4, 0.6]
    expected_output = 0.11375

    # When
    observed_output = pinball_loss(
        input_target_values, input_quantile_values, input_quantile_levels, input_sample_weights, input_quantile_weights
    )

    # Then
    assert np.isclose(expected_output, observed_output)
