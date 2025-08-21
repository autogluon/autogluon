from __future__ import annotations

from pprint import pformat
from typing import Any, Optional, Sequence, Type, Union

import numpy as np

from .abstract import TimeSeriesScorer
from .point import MAE, MAPE, MASE, MSE, RMSE, RMSLE, RMSSE, SMAPE, WAPE, WCD
from .quantile import SQL, WQL

__all__ = [
    "TimeSeriesScorer",
    "check_get_evaluation_metric",
    "MAE",
    "MAPE",
    "MASE",
    "SMAPE",
    "MSE",
    "RMSE",
    "RMSLE",
    "RMSSE",
    "SQL",
    "WAPE",
    "WCD",
    "WQL",
]

DEFAULT_METRIC_NAME = "WQL"

AVAILABLE_METRICS: dict[str, Type[TimeSeriesScorer]] = {
    "MASE": MASE,
    "MAPE": MAPE,
    "SMAPE": SMAPE,
    "RMSE": RMSE,
    "RMSLE": RMSLE,
    "RMSSE": RMSSE,
    "WAPE": WAPE,
    "SQL": SQL,
    "WQL": WQL,
    "MSE": MSE,
    "MAE": MAE,
}

# For backward compatibility
DEPRECATED_METRICS = {
    "mean_wQuantileLoss": "WQL",
}

# Experimental metrics that are not yet user facing
EXPERIMENTAL_METRICS: dict[str, Type[TimeSeriesScorer]] = {
    "WCD": WCD,
}


def check_get_evaluation_metric(
    eval_metric: Union[str, TimeSeriesScorer, Type[TimeSeriesScorer], None],
    prediction_length: int,
    seasonal_period: Optional[int] = None,
    horizon_weight: Optional[Sequence[float] | np.ndarray] = None,
) -> TimeSeriesScorer:
    """Factory method for TimeSeriesScorer objects.

    Returns
    -------
    scorer
        A `TimeSeriesScorer` object based on the provided `eval_metric`.

        `scorer.prediction_length` is always set to the `prediction_length` provided to this method.

        If `seasonal_period` is not `None`, then `scorer.seasonal_period` is set to this value. Otherwise the original
        value of `seasonal_period` is kept.

        If `horizon_weight` is not `None`, then `scorer.horizon_weight` is set to this value. Otherwise the original
        value of `horizon_weight` is kept.
    """
    scorer: TimeSeriesScorer
    metric_kwargs: dict[str, Any] = dict(
        prediction_length=prediction_length, seasonal_period=seasonal_period, horizon_weight=horizon_weight
    )
    if isinstance(eval_metric, TimeSeriesScorer):
        scorer = eval_metric
        scorer.prediction_length = prediction_length
        if seasonal_period is not None:
            scorer.seasonal_period = seasonal_period
        if horizon_weight is not None:
            scorer.horizon_weight = scorer.check_get_horizon_weight(
                horizon_weight, prediction_length=prediction_length
            )
    elif isinstance(eval_metric, type) and issubclass(eval_metric, TimeSeriesScorer):
        # e.g., user passed `eval_metric=CustomMetric` instead of `eval_metric=CustomMetric()`
        scorer = eval_metric(**metric_kwargs)
    elif isinstance(eval_metric, str):
        metric_name = DEPRECATED_METRICS.get(eval_metric, eval_metric).upper()
        if metric_name in AVAILABLE_METRICS:
            scorer = AVAILABLE_METRICS[metric_name](**metric_kwargs)
        elif metric_name in EXPERIMENTAL_METRICS:
            scorer = EXPERIMENTAL_METRICS[metric_name](**metric_kwargs)
        else:
            raise ValueError(
                f"Time series metric {eval_metric} not supported. Available metrics are:\n"
                f"{pformat(sorted(AVAILABLE_METRICS.keys()))}"
            )
    elif eval_metric is None:
        scorer = AVAILABLE_METRICS[DEFAULT_METRIC_NAME](**metric_kwargs)
    else:
        raise ValueError(
            f"eval_metric must be of type str, TimeSeriesScorer or None "
            f"(received eval_metric = {eval_metric} of type {type(eval_metric)})"
        )
    return scorer
