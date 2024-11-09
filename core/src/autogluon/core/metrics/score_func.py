from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # avoid circular import for type hints
    from . import Scorer

logger = logging.getLogger(__name__)


def compute_metric(
    y: np.ndarray,
    metric: "Scorer",
    *,
    y_pred: np.ndarray = None,
    y_pred_proba: np.ndarray = None,
    weights: np.ndarray = None,
    weight_evaluation: bool = None,
    as_error: bool = False,
    silent: bool = False,
    **kwargs,
) -> float:
    """
    Returns the metric score for the given Scorer object based on the input y and y_pred or y_pred_proba.

    Parameters
    ----------
    y : np.ndarray
        The ground truth target labels used for scoring.
        Must be the same length as y_pred / y_pred_proba.
    metric : Scorer
        The Scorer object used to calculate the score given y and y_pred or y_pred_proba.
        Based on the Scorer, either y_pred or y_pred_proba must be specified.
        Either pass both y_pred and y_pred_proba as input, or use the Scorer's properties to determine the correct input:
            If metric.needs_pred or metric.needs_quantile, then use `y_pred`.
            If metric.needs_proba or metric.needs_threshold, then use `y_pred_proba`.
    y_pred : np.ndarray, optional
        The target predictions. Typically, the output of calling `model.predict(X)`.
        Quantile regression predictions should use this argument.
        Will raise an exception if the metric requires `y_pred` and `y_pred` is unspecified.
    y_pred_proba : np.ndarray, optional
        The prediction probabilities. Typically, the output of calling `model.predict_proba(X)`.
        Will raise an exception if the metric requires `y_pred_proba` and `y_pred_proba` is unspecified.
    weights : np.ndarray, optional
        The sample weights for the metric calculation.
        If unspecified, the metric will be calculated assuming uniform weights.
    weight_evaluation : bool, optional
        If True, will use `weights` in the metric and will raise an exception if `weights` is unspecified.
        If False, will not use `weights`.
        If unspecified, will use `weights` only if `weights` is specified.
    as_error : bool, default = False
        If True, returns error (lower is better, optimum is 0) (calls metric.error(...))
        If False, returns score (higher is better) (calls metric(...))
    silent : bool, default = False
        If True, will not log any warnings if `weights` is specified but unsupported by the metric.
        If False, will log a warning if `weights` is specified but unsupported by the metric.
    **kwargs
        Additional keyword arguments that are passed to the metric call.

    Returns
    -------
    score or error: float
        If as_error=True, return error (lower is better, optimum is 0)
        If as_error=False, return score (higher is better)
    """
    if not metric.needs_quantile:
        kwargs.pop("quantile_levels", None)
    if weight_evaluation is None:
        weight_evaluation = not (weights is None)
    if weight_evaluation and weights is None:
        raise ValueError("Sample weights cannot be None when weight_evaluation=True.")
    if as_error:
        func = metric.error
    else:
        func = metric
    if metric.needs_pred or metric.needs_quantile:
        if y_pred is None:
            raise ValueError(f"y_pred must be specified for metric {metric.name}... (needs_pred={metric.needs_pred}, need_quantile={metric.needs_quantile}")
        predictions = y_pred
    elif metric.needs_proba or metric.needs_threshold:
        if y_pred_proba is None:
            raise ValueError(
                f"y_pred_proba must be specified for metric {metric.name}... "
                f"(needs_proba={metric.needs_proba}, needs_threshold={metric.needs_threshold}"
            )
        predictions = y_pred_proba
    else:
        raise AssertionError(
            f"Metric {metric.name} does not support predictions or prediction probabilities as input. "
            f"Ensure the metric is constructed properly."
        )
    if not weight_evaluation:
        return func(y, predictions, **kwargs)
    try:
        weighted_metric = func(y, y_pred, sample_weight=weights, **kwargs)
    except (ValueError, TypeError, KeyError):
        if hasattr(metric, "name"):
            metric_name = metric.name
        else:
            metric_name = metric
        if not silent:
            logger.log(30, f"WARNING: eval_metric='{metric_name}' does not support sample weights so they will be ignored in reported metric.")
        weighted_metric = func(y, y_pred, **kwargs)
    return weighted_metric
