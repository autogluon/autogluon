from typing import Optional

import numpy as np
import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame

from .abstract import TimeSeriesScorer
from .utils import _in_sample_abs_seasonal_error


class WQL(TimeSeriesScorer):
    """Weighted quantile loss.

    Also known as weighted pinball loss.
    """

    needs_quantile = True

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, q_pred, quantile_levels = self._get_quantile_forecast_score_inputs(data_future, predictions, target)
        values_true = y_true.values[:, None]  # shape [N, 1]
        values_pred = q_pred.values  # shape [N, len(quantile_levels)]

        return 2 * np.mean(
            np.abs((values_true - values_pred) * ((values_true <= values_pred) - quantile_levels)).sum(axis=0)
            / np.abs(values_true).sum()
        )


class SQL(TimeSeriesScorer):
    """Scaled quantile loss.

    Also known as scaled pinball loss.
    """

    needs_quantile = True

    def __init__(self):
        self._past_abs_seasonal_error: Optional[pd.Series] = None

    def save_past_metrics(
        self, data_past: TimeSeriesDataFrame, target: str = "target", seasonal_period: int = 1, **kwargs
    ) -> None:
        self._past_abs_seasonal_error = _in_sample_abs_seasonal_error(
            y_past=data_past[target], seasonal_period=seasonal_period
        )

    def clear_past_metrics(self) -> None:
        self._past_abs_seasonal_error = None

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        if self._past_abs_seasonal_error is None:
            raise AssertionError("Call `save_past_metrics` before `compute_metric`")

        y_true, q_pred, quantile_levels = self._get_quantile_forecast_score_inputs(data_future, predictions, target)
        values_true = y_true.values[:, None]  # shape [N, 1]

        ql = ((q_pred - values_true) * ((values_true <= q_pred) - quantile_levels)).mean(axis=1).abs()
        # TODO: Speed up computation by using np.arrays & replace groupby with reshapes [-1, prediction_length]?
        ql_per_item = ql.groupby(level=ITEMID, sort=False).mean()
        return 2 * self._safemean(ql_per_item / self._past_abs_seasonal_error)
