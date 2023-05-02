from typing import Optional

import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame

from .abstract_transformer import AbstractTransformer


class StdScaler(AbstractTransformer):
    """Divide all time series values by its standard deviation."""

    def __init__(self, target: str = "target", copy: bool = True, min_scale: float = 1e-2):
        super().__init__(target=target, copy=copy)
        self.min_scale = min_scale
        self.scale_per_item: Optional[pd.Series] = None

    def _fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        self.scale_per_item = data[self.target].groupby(level=ITEMID, sort=False).std().clip(lower=self.min_scale)
        data[self.target] = data[self.target] / self.scale_per_item
        return data

    def _inverse_transform_predictions(self, predictions: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        for col in predictions.columns:
            predictions[col] = predictions[col] * self.scale_per_item
        return predictions
