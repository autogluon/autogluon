from pprint import pformat
from typing import Type, Union

from .abstract import TimeSeriesScorer
from .point import MAE, MAPE, MASE, MSE, RMSE, RMSLE, RMSSE, SMAPE, WAPE
from .quantile import SQL, WQL

__all__ = [
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
    "WQL",
]

DEFAULT_METRIC_NAME = "WQL"

AVAILABLE_METRICS = {
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


def check_get_evaluation_metric(
    eval_metric: Union[str, TimeSeriesScorer, Type[TimeSeriesScorer], None] = None
) -> TimeSeriesScorer:
    if isinstance(eval_metric, TimeSeriesScorer):
        eval_metric = eval_metric
    elif isinstance(eval_metric, type) and issubclass(eval_metric, TimeSeriesScorer):
        # e.g., user passed `eval_metric=CustomMetric` instead of `eval_metric=CustomMetric()`
        eval_metric = eval_metric()
    elif isinstance(eval_metric, str):
        eval_metric = DEPRECATED_METRICS.get(eval_metric, eval_metric)
        if eval_metric.upper() not in AVAILABLE_METRICS:
            raise ValueError(
                f"Time series metric {eval_metric} not supported. Available metrics are:\n"
                f"{pformat(sorted(AVAILABLE_METRICS.keys()))}"
            )
        eval_metric = AVAILABLE_METRICS[eval_metric.upper()]()
    elif eval_metric is None:
        eval_metric = AVAILABLE_METRICS[DEFAULT_METRIC_NAME]()
    else:
        raise ValueError(
            f"eval_metric must be of type str, TimeSeriesScorer or None "
            f"(received eval_metric = {eval_metric} of type {type(eval_metric)})"
        )
    return eval_metric
