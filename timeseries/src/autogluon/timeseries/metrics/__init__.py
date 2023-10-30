import json
from typing import Type, Union

from .abstract import TimeSeriesScorer
from .point import MAE, MAPE, MASE, MSE, RMSE, RMSSE, WAPE, sMAPE
from .quantile import SQL, WQL

__all__ = [
    "MAE",
    "MAPE",
    "MASE",
    "sMAPE",
    "MSE",
    "RMSE",
    "RMSSE",
    "SQL",
    "WAPE",
    "WQL",
]

DEFAULT_METRIC_NAME = "WQL"

AVAILABLE_METRICS = {
    "MASE": MASE,
    "MAPE": MAPE,
    "SMAPE": sMAPE,
    "RMSE": RMSE,
    "RMSSE": RMSSE,
    "WAPE": WAPE,
    "SQL": SQL,
    "WQL": WQL,
    # Exist for compatibility
    "MSE": MSE,
    "MAE": MAE,
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
        if eval_metric.upper() not in AVAILABLE_METRICS:
            raise ValueError(
                f"Time series metric {eval_metric} not supported. Available metrics are:\n"
                f"{json.dumps(list(AVAILABLE_METRICS.keys()), indent=2)}"
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
