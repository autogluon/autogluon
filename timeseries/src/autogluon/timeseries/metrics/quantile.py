from typing import List, Optional

import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame

from .abstract import TimeSeriesScorer


class QuantileForecastScorer(TimeSeriesScorer):
    """Base class for all quantile forecast metrics."""

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ) -> float:
        assert quantile_levels is not None, f"{self.name} expects `quantile_levels` to be provided"
        y_true = data_future[target]
        q_pred = predictions[[str(q) for q in quantile_levels]]
        quantile_levels = np.array(quantile_levels, dtype=float)
        return self._compute_quantile_metric(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)


class WQL(QuantileForecastScorer):
    def _compute_quantile_metric(self, y_true: pd.Series, q_pred: pd.DataFrame, quantile_levels: List[float]) -> float:
        values_true = y_true.values[:, None]  # shape [N, 1]
        values_pred = q_pred.values  # shape [N, len(quantile_levels)]

        return 2 * np.mean(
            np.abs((values_true - values_pred) * ((values_true <= values_pred) - quantile_levels)).sum(axis=0)
            / np.abs(values_true).sum()
        )
