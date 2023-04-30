import numpy as np
import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame

from .abstract_transformer import AbstractTransformer


class DetrenderSingle:
    """Detrend a single time series.

    Based on statsmodels.tsa.tsatools.detrend.
    """

    def __init__(self, order: int = 1):
        self.order = order
        self.nobs: int = None
        self.beta: np.ndarray = None

    def fit_transform(self, ts: pd.Series) -> pd.Series:
        self.nobs = len(ts)
        trends = np.vander(np.arange(len(ts)), N=self.order + 1)
        self.beta = np.linalg.pinv(trends).dot(ts)
        trends = np.vander(np.arange(len(ts)), N=self.order + 1)
        resid = ts - trends @ self.beta
        return resid

    def get_future_trend(self, prediction_length: int) -> np.ndarray:
        future_trends = np.vander(np.arange(self.nobs, self.nobs + prediction_length), N=self.order + 1)
        return future_trends @ self.beta


class Detrender(AbstractTransformer):
    """Remove trend from the time series by fitting polynomial regression.

    Parameters
    ----------
    order : int, default = 1
        Order of the polynomial used to remove trend. 0 = constant, 1 = linear, 2 = quadratic, ...
    """

    def __init__(self, target: str = "target", copy: bool = True, order: int = 1):
        super().__init__(target=target, copy=copy)
        self.order = order
        self.detrender_per_series = {}

    def _fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        transformed_series = []
        for item_id, ts in data[self.target].groupby(level=ITEMID, sort=False):
            self.detrender_per_series[item_id] = DetrenderSingle(order=self.order)
            transformed_series.append(self.detrender_per_series[item_id].fit_transform(ts))
        data[self.target] = pd.concat(transformed_series)
        return data

    def _inverse_transform_predictions(self, predictions: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        future_trends = []
        for item_id, length in predictions.num_timesteps_per_item().items:
            future_trends.append(self.detrender_per_series[item_id].get_future_trend(length))
        future_trends = np.concatenate(future_trends)
        for col in predictions.columns:
            predictions[col] = predictions[col] + future_trends
        return predictions
