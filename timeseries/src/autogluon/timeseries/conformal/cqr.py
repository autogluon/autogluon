from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame, ITEMID
from .abstract_conformalizer import AbstractConformalizer


class ConformalizedQuantileRegression(AbstractConformalizer):
    def __init__(self, prediction_length: int, quantile_levels: List[float], target_column: str = "target"):
        super().__init__(
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            target_column=target_column,
        )
        self.quantile_adjustments: Dict[str, float] = {}

    def fit(self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame):
        data_future = data.slice_by_timestep(-self.prediction_length, None)
        assert (data_future.index == predictions.index).all()
        assert all(str(q) in predictions.columns for q in self.quantile_levels)

        y_true = data_future[self.target_column]
        num_samples = len(y_true)
        self.quantile_adjustments = {}
        for q in self.quantile_levels:
            y_pred = predictions[str(q)]
            if q > 0.5:
                error_high = y_true - y_pred
                error_high = np.sort(error_high)
                index_high = int(np.ceil(q * (num_samples + 1))) - 1
                index_high = min(max(index_high, 0), num_samples - 1)
                correction = error_high[index_high]
            else:
                error_low = y_pred - y_true
                error_low = np.sort(error_low)
                index_low = int(np.ceil((1 - q) * (num_samples + 1))) - 1
                index_low = min(max(index_low, 0), num_samples - 1)
                correction = -error_low[index_low]
            self.quantile_adjustments[str(q)] = correction
        self._is_fit = True

    def transform(
        self, predictions: TimeSeriesDataFrame, data_past: Optional[TimeSeriesDataFrame] = None
    ) -> TimeSeriesDataFrame:
        assert self._is_fit, f"{self.__class__.__name__} has not been fit yet"
        assert all(str(q) in predictions.columns for q in self.quantile_levels)
        for q in self.quantile_levels:
            predictions[str(q)] = predictions[str(q)] + self.quantile_adjustments[str(q)]
        return predictions


class ScaledConformalizedQuantileRegression(ConformalizedQuantileRegression):
    def _compute_scale(self, data: TimeSeriesDataFrame, min_scale: float = 1e-5) -> pd.Series:
        return data.groupby(level=ITEMID, sort=False)[self.target_column].std().clip(lower=min_scale)

    def fit(self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame):
        # Avoid scaling the original data
        data = data.copy()
        data_past = data.slice_by_timestep(None, -self.prediction_length)
        scale_per_item = self._compute_scale(data_past)
        for col in predictions.columns:
            predictions[col] = predictions[col] / scale_per_item
        data[self.target_column] = data[self.target_column] / scale_per_item
        super().fit(data, predictions)

    def transform(
        self, predictions: TimeSeriesDataFrame, data_past: Optional[TimeSeriesDataFrame] = None
    ) -> TimeSeriesDataFrame:
        scale_per_item = self._compute_scale(data_past)
        for col in predictions.columns:
            predictions[col] = predictions[col] / scale_per_item
        predictions = super().transform(predictions=predictions)
        for col in predictions.columns:
            predictions[col] = predictions[col] * scale_per_item
        return predictions
