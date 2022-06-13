from typing import Optional

METRIC_COEFFICIENTS = {"MASE": -1, "MAPE": -1, "sMAPE": -1, "mean_wQuantileLoss": -1}

AVAILABLE_METRICS = ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]
DEFAULT_METRIC = "mean_wQuantileLoss"
assert DEFAULT_METRIC in AVAILABLE_METRICS


def check_get_evaluation_metric(
    metric_name: Optional[str] = DEFAULT_METRIC,
    raise_if_not_available: bool = True,
):
    """A utility function that checks if a given evaluation metric
    name is available in autogluon.timeseries, and optionally raises
    a ValueError otherwise.

    Parameters
    ----------
    metric_name: str
        The requested metric name, currently one of the evaluation metrics available
        in GluonTS.
    raise_if_not_available: bool
        if True, a ValueError will be raised if the requested metric is not yet
        available in autogluon.timeseries. Otherwise, the default metric name will be
        returned instead of the requested metric.

    Returns
    -------
    checked_metric_name: str
        The requested metric name if it is available in autogluon.timeseries.
    """
    metric = metric_name or DEFAULT_METRIC
    if metric not in AVAILABLE_METRICS:
        if raise_if_not_available:
            raise ValueError(f"metric {metric} is not available yet.")
        return DEFAULT_METRIC
    return metric
