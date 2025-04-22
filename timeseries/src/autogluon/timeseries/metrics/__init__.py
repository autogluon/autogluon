from pprint import pformat
from typing import Type, Union

from .abstract import TimeSeriesScorer
from .point import MAE, MAPE, MASE, MSE, RMSE, RMSLE, RMSSE, SMAPE, WAPE, WCD
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
    "WCD",
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

# Experimental metrics that are not yet user facing
EXPERIMENTAL_METRICS = {
    "WCD": WCD,
}


def check_get_evaluation_metric(
    eval_metric: Union[str, TimeSeriesScorer, Type[TimeSeriesScorer], None] = None
) -> TimeSeriesScorer:
    scorer: TimeSeriesScorer
    if isinstance(eval_metric, TimeSeriesScorer):
        scorer = eval_metric
    elif isinstance(eval_metric, type) and issubclass(eval_metric, TimeSeriesScorer):
        # e.g., user passed `eval_metric=CustomMetric` instead of `eval_metric=CustomMetric()`
        scorer = eval_metric()
    elif isinstance(eval_metric, str):
        metric_name = DEPRECATED_METRICS.get(eval_metric, eval_metric).upper()
        if metric_name in AVAILABLE_METRICS:
            scorer = AVAILABLE_METRICS[metric_name]()
        elif metric_name in EXPERIMENTAL_METRICS:
            scorer = EXPERIMENTAL_METRICS[metric_name]()
        else:
            raise ValueError(
                f"Time series metric {eval_metric} not supported. Available metrics are:\n"
                f"{pformat(sorted(AVAILABLE_METRICS.keys()))}"
            )
    elif eval_metric is None:
        scorer = AVAILABLE_METRICS[DEFAULT_METRIC_NAME]()
    else:
        raise ValueError(
            f"eval_metric must be of type str, TimeSeriesScorer or None "
            f"(received eval_metric = {eval_metric} of type {type(eval_metric)})"
        )
    return scorer
