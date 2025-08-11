"""Metrics for quantile regression"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def pinball_loss(target_value, quantile_values, quantile_levels, sample_weight=None, quantile_weight=None):
    # "target_value" must be 2D pandas or numpy arrays
    target_value = np.array(target_value).reshape(-1, 1)

    # here we assume, 'quantile_values' is 2D pandas array
    quantile_values = np.array(quantile_values)
    if len(quantile_values.shape) != 2:
        raise ValueError("quantile prediction values must be 2D numpy arrays [num_samples x num_quantiles]")

    if target_value.shape[0] != quantile_values.shape[0]:
        raise ValueError("target and quantile prediction values must have the same number of examples ")

    # quantile levels as list
    quantile_levels = np.array(quantile_levels).reshape(1, -1)
    if quantile_values.shape[1] != quantile_levels.shape[1]:
        raise ValueError(
            "quantile prediction values must have the same number of predictions as the number of quantile levels"
        )

    # pinball loss
    error_values = target_value - quantile_values
    loss_values = np.maximum(quantile_levels * error_values, (quantile_levels - 1) * error_values)

    # return mean over all samples (sample weighted) and quantile levels
    return np.average(np.average(loss_values, weights=sample_weight, axis=0), weights=quantile_weight, axis=0)
