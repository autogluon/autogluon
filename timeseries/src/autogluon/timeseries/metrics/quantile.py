from typing import List
import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame

from .abstract import TimeSeriesMetric


class QuantileForecastMetric(TimeSeriesMetric):
    def score_with_saved_past_metrics(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true = data_future[target]
        quantile_pred_columns = [col for col in predictions.columns if col != "mean"]
        y_pred = predictions[quantile_pred_columns]
        quantile_levels = np.array([float(q) for q in quantile_pred_columns], dtype=float)
        return self._compute_quantile_metric(y_true=y_true, y_pred=y_pred, quantile_levels=quantile_levels)


class WQL(TimeSeriesMetric):
    def _compute_quantile_metric(self, y_true: pd.Series, q_pred: pd.DataFrame, quantile_levels: List[float]) -> float:
        values_true = y_true.values[:, None]  # shape [N, 1]
        values_pred = q_pred.values  # shape [N, len(quantile_levels)]

        return 2 * np.mean(
            np.abs((values_true - values_pred) * ((values_true <= values_pred) - quantile_levels)).sum(axis=0)
            / np.abs(values_true).sum()
        )
