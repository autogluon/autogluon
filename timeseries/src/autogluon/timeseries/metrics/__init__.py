from pprint import pformat
from typing import Optional, Type, Union, overload

import numpy as np
import numpy.typing as npt

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

@overload
def check_get_horizon_weight(horizon_weight: None, prediction_length: int) -> None: ...
@overload
def check_get_horizon_weight(horizon_weight: list[float], prediction_length: int) -> np.ndarray: ...

def check_get_horizon_weight(horizon_weight: list[float] | None, prediction_length: int) -> Optional[np.ndarray]:
    """Convert horizon_weight to a non-negative numpy array that sums up to prediction_length.

    Raises an exception if horizon_weight has an invalid shape or contains invalid values.
    """
    if horizon_weight is None:
        return None
    horizon_weight_np = np.array(list(horizon_weight), dtype=np.float64)
    if horizon_weight_np.shape != (prediction_length,):
        raise ValueError(
            f"horizon_weight must have length equal to {prediction_length=} (got {len(horizon_weight)=})"
        )
    if not (horizon_weight_np >= 0).all():
        raise ValueError(f"All values in horizon_weight must be >= 0 (got {horizon_weight})")
    if not horizon_weight_np.sum() > 0:
        raise ValueError(f"At least some values in horizon_weight must be > 0 (got {horizon_weight})")
    if not np.isfinite(horizon_weight_np).all():
        raise ValueError(f"All horizon_weight values must be finite (got {horizon_weight})")
    horizon_weight_np = horizon_weight_np / (horizon_weight_np.sum() / prediction_length)
    return horizon_weight_np
