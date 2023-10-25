from typing import Optional

from autogluon.common.utils.deprecated_utils import Deprecated
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics import AVAILABLE_METRICS, check_get_evaluation_metric


@Deprecated(
    min_version_to_warn="1.0",
    min_version_to_error="1.1",
    custom_warning_msg="Please use the metrics defined in autogluon.timeseries.metrics instead.",
)
class TimeSeriesEvaluator:
    """This class has been deprecated in AutoGluon v1.0 and is only provided for backward compatibility!"""

    METRIC_COEFFICIENTS = {metric_name: metric_cls().sign for metric_name, metric_cls in AVAILABLE_METRICS.items()}
    AVAILABLE_METRICS = list(AVAILABLE_METRICS.keys())
    DEFAULT_METRIC = check_get_evaluation_metric(None).name

    def __init__(
        self,
        eval_metric: str,
        prediction_length: int,
        target_column: str = "target",
        eval_metric_seasonal_period: Optional[int] = None,
    ):
        self.eval_metric = check_get_evaluation_metric(eval_metric)
        self.prediction_length = prediction_length
        self.target_column = target_column
        self.seasonal_period = eval_metric_seasonal_period

    @property
    def coefficient(self) -> int:
        return self.eval_metric.sign

    @property
    def higher_is_better(self) -> bool:
        return self.eval_metric.greater_is_better_internal

    @staticmethod
    def check_get_evaluation_metric(
        metric_name: Optional[str] = None,
        raise_if_not_available: bool = True,
    ):
        return check_get_evaluation_metric(metric_name)

    def __call__(self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame) -> float:
        quantile_levels = [float(col) for col in predictions.columns if col != "mean"]
        score = self.eval_metric(
            data=data,
            predictions=predictions,
            prediction_length=self.prediction_length,
            target=self.target_column,
            seasonal_period=self.seasonal_period,
            quantile_levels=quantile_levels,
        )
        # Return raw metric in lower-is-better format to match the old Evaluator API
        return score * self.eval_metric.sign
