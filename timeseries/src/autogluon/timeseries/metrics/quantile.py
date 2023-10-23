import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame

from .abstract import TimeSeriesScorer


class QuantileForecastScorer(TimeSeriesScorer):
    """Base class for all quantile forecast metrics."""

    is_quantile_metric = True

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        quantile_columns = [col for col in predictions.columns if col != "mean"]
        quantile_levels = np.array(quantile_columns, dtype=float)
        y_true = data_future[target]
        q_pred = predictions[quantile_columns]
        return self._compute_quantile_metric(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)

    def _compute_quantile_metric(self, y_true: pd.Series, q_pred: pd.DataFrame, quantile_levels: np.ndarray) -> float:
        raise NotImplementedError


class WQL(QuantileForecastScorer):
    """Weighted quantile loss.

    Also known as weighted pinball loss.
    """

    def _compute_quantile_metric(self, y_true: pd.Series, q_pred: pd.DataFrame, quantile_levels: np.ndarray) -> float:
        values_true = y_true.values[:, None]  # shape [N, 1]
        values_pred = q_pred.values  # shape [N, len(quantile_levels)]

        return 2 * np.mean(
            np.abs((values_true - values_pred) * ((values_true <= values_pred) - quantile_levels)).sum(axis=0)
            / np.abs(values_true).sum()
        )
