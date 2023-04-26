import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame

from .abstract_transform import AbstractTransform


class Scaler(AbstractTransform):
    def __init__(self, target: str = "target", min_scale: float = 1e-2):
        super().__init__(target=target)
        self.min_scale = min_scale
        self.scale_per_item: pd.Series = None

    def _fit(self, data: TimeSeriesDataFrame) -> None:
        self.scale_per_item = (
            data[self.target].abs().groupby(level=ITEMID, sort=False).mean().clip(lower=self.min_scale)
        )

    def _transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        assert data.item_ids.equals(self.scale_per_item.index), f"{self.name} was fit on data with different item_ids"
        data[self.target] = data[self.target] / self.scale_per_item
        return data

    def _inverse_transform_predictions(self, predictions: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        assert predictions.item_ids.equals(
            self.scale_per_item.index
        ), f"{self.name} was fit on data with different item_ids"
        for col in predictions.columns:
            predictions[col] = predictions[col] * self.scale_per_item
        return predictions


class StdScaler(Scaler):
    """Divide all time series values by its standard deviation."""

    def _fit(self, data: TimeSeriesDataFrame) -> None:
        self.scale_per_item = (
            data[self.target].abs().groupby(level=ITEMID, sort=False).mean().clip(lower=self.min_scale)
        )


class MeanAbsoluteScaler(Scaler):
    """Divide all time series values by its mean absolute value."""

    def _fit(self, data: TimeSeriesDataFrame) -> None:
        self.scale_per_item = data[self.target].groupby(level=ITEMID, sort=False).std().clip(lower=self.min_scale)
