import numpy as np
import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame

from .abstract_transform import AbstractTransform


class DetrenderSingle:
    """Detrend a single time series."""

    def __init__(self, order: int = 1):
        self.order = order
        self.nobs: int = None
        self.beta: np.ndarray = None

    def fit_transform(self, ts: pd.Series):
        self.nobs = len(ts)
        trends = np.vander(np.arange(len(ts)), N=self.order + 1)
        self.beta = np.linalg.pinv(trends).dot(ts)
        trends = np.vander(np.arange(len(ts)), N=self.order + 1)
        resid = ts - trends @ self.beta
        return resid

    def get_future_trend(self, prediction_length: int) -> np.ndarray:
        future_trends = np.vander(np.arange(self.nobs, self.nobs + prediction_length), N=self.order + 1)
        return future_trends @ self.beta


class Detrender(AbstractTransform):
    def __init__(self, order: int = 1, target: str = "target"):
        super().__init__(target)
        self.order = order
        self.detrender_per_series = {}

    def fit_transform(self, data: TimeSeriesDataFrame, inplace: bool = False) -> TimeSeriesDataFrame:
        if not inplace:
            data = data.copy()
        transformed_series = []
        for item_id, ts in data[self.target].groupby(level=ITEMID, sort=False):
            self.detrender_per_series[item_id] = DetrenderSingle(order=self.order)
            transformed_series.append(self.detrender_per_series[item_id].fit_transform(ts))
        data[self.target] = pd.concat(transformed_series)
        self._is_fit = True
        return data

    def _inverse_transform_predictions(self, predictions: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        future_trends = []
        for item_id, length in predictions.num_timesteps_per_item().iteritems():
            future_trends.append(self.detrender_per_series[item_id].get_future_trend(length))
        future_trends = np.concatenate(future_trends)
        for col in predictions.columns:
            predictions[col] = predictions[col] + future_trends
        return predictions
