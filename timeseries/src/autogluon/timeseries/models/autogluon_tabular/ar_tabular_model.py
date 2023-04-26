import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import autogluon.core as ag
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.transforms import Detrender, StdScaler
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
from autogluon.timeseries.utils.seasonality import get_seasonality

logger = logging.getLogger(__name__)


class RecurrentTabularModel(AbstractTimeSeriesModel):
    """Fit a quantile regression for 1-step-ahead forecasts.

    At prediction time, forecasts are generated one step at a time.

    Other Parameters
    ----------------
    detrend : bool, default = True
        If True, linear trend will be removed from each time series.
    scale_target : bool, default = True
        If True, each time series will be divided by its standard deviation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_lags = self._get_num_lags_from_freq(self.freq)
        if 0.5 not in self.quantile_levels:
            self.median_should_be_dropped = True
            self.quantile_levels = self.quantile_levels + [0.5]
        else:
            self.median_should_be_dropped = False

    @staticmethod
    def _get_num_lags_from_freq(freq: str, min_num_lags: int = 20, max_num_lags: int = 50) -> int:
        seasonality = get_seasonality(freq)
        return min(max(seasonality * 2, min_num_lags), max_num_lags)

    def _tsdf_to_list(self, data: TimeSeriesDataFrame) -> List[np.ndarray]:
        """Convert a TimeSeriesDataFrame into a list of numpy arrays."""
        target_series = data[self.target]
        return [ts.to_numpy() for _, ts in target_series.groupby(level=ITEMID, sort=False)]

    @property
    def _feature_column_names(self) -> List[str]:
        return [f"lag_{idx}" for idx in range(1, self.num_lags + 1)]

    def _get_features_dataframe(self, data: TimeSeriesDataFrame, last_k_values: Optional[int] = None) -> pd.DataFrame:
        """Construct a features dataframe from time series data."""

        def get_lags_and_target(ts: np.ndarray):
            """Get a sliding window view of the time series."""
            extended = np.concatenate([np.full(self.num_lags, np.nan), ts])
            result = sliding_window_view(extended, self.num_lags + 1)
            if last_k_values is not None:
                result = result[-last_k_values:]
            return result

        all_series = self._tsdf_to_list(data)
        results = [get_lags_and_target(ts) for ts in all_series]
        features = np.concatenate(results)
        features = features[~np.isnan(features).any(axis=1)]
        return pd.DataFrame(features, columns=self._feature_column_names + [self.target])

    def _fit_tabular_model(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], **kwargs) -> None:
        """Fit tabular model to the features."""
        raise NotImplementedError

    def _predict_tab_model(self, features_df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        **kwargs,
    ):
        model_params = self._get_model_params().copy()
        if model_params.get("detrend", True):
            self.transforms.append(Detrender(target=self.target))
        if model_params.get("scale_target", True):
            self.transforms.append(StdScaler(target=self.target))

        train_data = self.preprocess(train_data)
        train_df = self._get_features_dataframe(train_data)
        if val_data is not None:
            val_data = self.preprocess(val_data)
            val_df = self._get_features_dataframe(val_data, last_k_values=self.prediction_length)
        else:
            val_df = None

        self._fit_tabular_model(train_df, val_df, time_limit=time_limit)

    def _get_forecast_df(self, predictions: List[pd.DataFrame], data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Combine predictions made for each timestep into a single TimeSeriesDataFrame.

        Parameters
        ----------
        predictions : List[pd.DataFrame]
            Each entry is a pd.DataFrame of shape [num_items, len(quantile_levels)] that contains predicted quantiles
            for each item_id at the respective timestamp.
        data : TimeSeriesDataFrame
            Data for which the forecast is generated

        Returns
        -------
        forecast_df : TimeSeriesDataFrame
            Predictions in TimeSeriesDataFrame format with columns containing mean and quantile forecasts.
        """
        # Predictions for each timestep are interleaved, such that in forecast_df
        # the predictions for each item_id are stored contiguously
        forecast_df = pd.concat(predictions).sort_index(kind="stable").reset_index(drop=True)
        forecast_df.index = get_forecast_horizon_index_ts_dataframe(data, self.prediction_length)
        forecast_df["mean"] = forecast_df["0.5"]
        return forecast_df

    def _append_predictions(self, features_df: pd.DataFrame, next_preds: pd.Series) -> pd.DataFrame:
        """Append predictions as the last column of features_df, drop the first column of features_df.

        Example
        -------
        features_df
            [1, 2, 3]
            [4, 5, 6]

        next_preds
            [8]
            [9]

        _append_predictions(features_df, next_preds)
            [2, 3, 8]
            [5, 6, 9]
        """
        features_updated = np.concatenate([features_df.values[:, 1:], next_preds.values[:, None]], axis=1)
        return pd.DataFrame(features_updated, columns=self._feature_column_names)

    def predict(self, data: TimeSeriesDataFrame, **kwargs) -> TimeSeriesDataFrame:
        def get_last_lags(ts: np.ndarray) -> np.ndarray:
            """Get a vector containing the last num_lags values of the time series.

            If the time series is shorter than num_lags, then leading entries are set to NaN.
            """
            result = np.full(self.num_lags, np.nan)
            num_values = min(self.num_lags, len(ts))
            result[-num_values:] = ts[-num_values:]
            return result

        data = self.preprocess(data)
        all_series = self._tsdf_to_list(data)
        features = np.stack([get_last_lags(ts) for ts in all_series])
        features_df = pd.DataFrame(features, columns=self._feature_column_names)
        predictions = []
        for _ in range(self.prediction_length):
            next_preds = self._predict_tab_model(features_df)
            predictions.append(next_preds)
            features_df = self._append_predictions(features_df, next_preds["0.5"])
        predictions_df = TimeSeriesDataFrame(self._get_forecast_df(predictions, data))
        predictions_df = self.inverse_transform_predictions(predictions_df)
        if self.median_should_be_dropped:
            predictions_df.drop("0.5", axis=1, inplace=True)
        return predictions_df


class LGBMRecurrentQuantileModel(RecurrentTabularModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from lightgbm import LGBMRegressor

        self.models = {str(q): LGBMRegressor(objective="quantile", alpha=q) for q in self.quantile_levels}

    def _fit_tabular_model(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], **kwargs):
        """Fit tabular model to the features."""
        X = train_df.drop(self.target, axis=1).values
        y = train_df[self.target].values
        for q, model in self.models.items():
            model.fit(X, y)

    def _predict_tab_model(self, features_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({q: model.predict(features_df.values) for q, model in self.models.items()})


class TabularRecurrentQuantileModel(RecurrentTabularModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from lightgbm import LGBMRegressor

        self.models = {str(q): LGBMRegressor(objective="quantile", alpha=q) for q in self.quantile_levels}

    def _fit_tabular_model(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], **kwargs):
        """Fit tabular model to the features."""
        from autogluon.tabular import TabularPredictor

        self.tabular_predictor = TabularPredictor(label=self.target, problem_type=ag.constants.QUANTILE).fit(train_df)

    def _predict_tab_model(self, features_df: pd.DataFrame) -> pd.DataFrame:
        predictions = self.tabular_predictor.predict(features_df)
        predictions.columns = [str(q) for q in predictions.columns]
        return predictions
