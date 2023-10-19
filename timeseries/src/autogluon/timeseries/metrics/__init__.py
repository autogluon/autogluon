import json
from typing import Union, Type
from .abstract import TimeSeriesScorer
from .point import MAE, MAPE, MASE, sMAPE, MSE, RMSE, RMSSE, WAPE
from .quantile import WQL

__all__ = [
    "MAE",
    "MAPE",
    "MASE",
    "sMAPE",
    "MSE",
    "RMSE",
    "RMSSE",
    "WAPE",
    "WQL",
]

DEFAULT_METRIC = "WQL"

AVAILABLE_METRICS = {
    "MASE": MASE,
    "MAPE": MAPE,
    "SMAPE": sMAPE,
    "RMSE": RMSE,
    "RMSSE": RMSSE,
    "WAPE": WAPE,
    "WQL": WQL,
    # Exist for compatibility
    "MSE": MSE,
    "MAE": MAE,
}


def check_get_evaluation_metric(
    eval_metric: Union[str, TimeSeriesScorer, Type[TimeSeriesScorer], None] = None,
) -> TimeSeriesScorer:
    if isinstance(metric, Type[TimeSeriesScorer]):
        metric = metric()
    elif isinstance(metric, TimeSeriesScorer):
        metric = metric
    elif isinstance(metric, str):
        if metric.upper() not in AVAILABLE_METRICS:
            raise ValueError(
                f"Metric {metric} not supported. Available metrics are:\n"
                f"{json.dumps(list(AVAILABLE_METRICS.keys()), indent=2)}"
            )
        metric = AVAILABLE_METRICS[metric]()
    elif metric is None:
        metric = AVAILABLE_METRICS[DEFAULT_METRIC]()
    else:
        raise ValueError(
            f"eval_metric must be of type str, TimeSeriesScorer or None "
            f"(received eval_metric = {eval_metric} of type {type(eval_metric)})"
        )
