import logging
import re
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

# TODO: Drop GluonTS dependency
from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str

import autogluon.core as ag
from autogluon.tabular import TabularPredictor
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class AutoGluonTabularModel(AbstractTimeSeriesModel):
    """Predict future time series values using autogluon.tabular.TabularPredictor.

    The forecasting is converted to a tabular problem using the following features:

    - lag features (observed time series values) based on ``freq`` of the data
    - time features (e.g., day of the week) based on the timestamp of the measurement
    - lagged known and past covariates (if available)
    - static features of each item (if available)

    Quantiles are obtained by assuming that the residuals follow zero-mean normal distribution, scale of which is
    estimated from the empirical distribution of the residuals.


    Other Parameters
    ----------------
    max_train_size : int, default = 1_000_000
        Maximum number of rows in the training and validation sets. If the number of rows in train or validation data
        exceeds ``max_train_size``, then ``max_train_size`` many rows are subsampled from the dataframe.
    tabular_hyperparmeters : Dict[Dict[str, Any]], optional
        Hyperparameters dictionary passed to `TabularPredictor.fit`. Contains the names of models that should be fit.
        Defaults to ``{"XGB": {}, "CAT": {}, "GBM" :{}}``.
    """

    default_tabular_hyperparameters = {
        "XGB": {},
        "CAT": {},
        "GBM": {},
    }

    PREDICTION_BATCH_SIZE = 100_000

    TIMESERIES_METRIC_TO_TABULAR_METRIC = {
        "MASE": "mean_absolute_error",
        "MAPE": "mean_absolute_percentage_error",
        "sMAPE": "mean_absolute_percentage_error",
        "mean_wQuantileLoss": "mean_absolute_error",
        "MSE": "mean_squared_error",
        "RMSE": "root_mean_squared_error",
    }

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: str = None,
        hyperparameters: Dict[str, Any] = None,
        **kwargs,  # noqa
    ):
        name = name or re.sub(r"Model$", "", self.__class__.__name__)  # TODO: look name up from presets
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        self._target_lag_indices: np.array = None
        self._known_covariates_lag_indices: np.array = None
        self._past_covariates_lag_indices: np.array = None
        self._time_features: List[Callable] = None
        self._available_features: pd.Index = None
        self.quantile_adjustments: Dict[str, float] = {}

        self.tabular_predictor = TabularPredictor(
            path=self.path,
            label=self.target,
            problem_type=ag.constants.REGRESSION,
            eval_metric=self.TIMESERIES_METRIC_TO_TABULAR_METRIC.get(self.eval_metric),
        )

    def _get_features_dataframe(
        self,
        data: TimeSeriesDataFrame,
        last_k_values: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate a feature matrix used by TabularPredictor.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Dataframe containing features derived from time index & past time series values, as well as the target.
        last_k_values: int, optional
            If provided, features will be generated only for the last `last_k_values` timesteps of each time series.
        """
        # TODO: Rethink the featurization process for time series based on SotA tree-based models (scaling, rolling feautres)
        # TODO: More efficient featurization with tsfresh? (currently sequential over time series => slow)

        def get_lags(df: pd.DataFrame, lag_indices: List[int]) -> pd.DataFrame:
            """Construct a dataframe consisting of shifted copies of the original df.

            Parameters
            ----------
            df
                Original dataframe, shape [N, D]
            lag_indices
                List of lag features to compute.

            Returns
            -------
            lag_df
                Dataframe with lag features, shape [N, D * len(lag_indices)]
            """
            shifted = [df.shift(idx).add_suffix(f"_lag_{idx}") for idx in lag_indices]
            return pd.concat(shifted, axis=1)

        def apply_mask(
            df: pd.DataFrame, num_hidden: np.ndarray, lag_indices: np.ndarray, num_columns: int = 1
        ) -> pd.DataFrame:
            """Apply a mask that mimics the situation at prediction time when target/covariates are unknown during the
            forecast horizon.

            Parameters
            ----------
            df
                Dataframe to mask, shape [N, D * len(lag_indices)]
            num_hidden
                Number of entries hidden in each row, shape [N]
            lag_indices
                Lag indices used to construct the dataframe
            num_columns
                D - number of columns in the original dataframe, before lag features were constructed

            Returns
            -------
            masked_df
                Dataframe with the masking applied, shape [N, D * len(lag_indices)]


            For example, given the following inputs

            df = [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
            num_hidden = [6, 0, 1]
            lag_indices = [1, 2, 5, 10]
            num_columns = 1

            The resulting masked dataframe will be

            masked_df = [
                [NaN, NaN, NaN, 1],
                [1, 1, 1, 1],
                [NaN, 1, 1, 1],
            ]

            """
            if num_columns > 1:
                lag_indices = np.repeat(lag_indices, num_columns)
            mask = num_hidden[:, None] >= lag_indices[None]  # shape [len(num_hidden), len(lag_indices) * num_columns]
            df[mask] = np.nan
            return df

        def get_lag_features_and_target(time_series: pd.DataFrame) -> pd.DataFrame:
            """Construct the dataframe with lagged features and prediction target for a single time series.

            Parameters
            ----------
            time_series
                Dataframe containing a single time series, including target & past/known covariates.
                Such time series can equivalently be obtained with data.loc[item_id] for some item_id in the dataset.

            Returns
            -------
            features
                Feature dataframe, where each row corresponds to a single timestep of the input time series.
            """
            target_lags = get_lags(time_series[[self.target]], self._target_lag_indices)

            # Starting from the end of the time series, mask the values as if the last `prediction_length` steps weren't observed
            # This mimics what will happen at test time, when we simultaneously predict the next `prediction_length` values
            num_windows = (len(time_series) - 1) // self.prediction_length
            # We don't hide any past values for the first `remainder` values, otherwise the features will be all empty
            remainder = len(time_series) - num_windows * self.prediction_length
            num_hidden = np.concatenate([np.zeros(remainder), np.tile(np.arange(self.prediction_length), num_windows)])
            target_lags = apply_mask(target_lags, num_hidden=num_hidden, lag_indices=self._target_lag_indices)
            feature_dfs = [target_lags, time_series[[self.target]]]

            if self.metadata.past_covariates_real:
                past_covariates_lags = get_lags(
                    time_series[self.metadata.past_covariates_real], self._past_covariates_lag_indices
                )
                past_covariates_lags = apply_mask(
                    past_covariates_lags,
                    num_hidden=num_hidden,
                    lag_indices=self._past_covariates_lag_indices,
                    num_columns=len(self.metadata.past_covariates_real),
                )
                feature_dfs.append(past_covariates_lags)

            if self.metadata.known_covariates_real:
                known_covariates_lags = get_lags(
                    time_series[self.metadata.known_covariates_real], self._known_covariates_lag_indices
                )
                feature_dfs.append(known_covariates_lags)

            features = pd.concat(feature_dfs, axis=1)
            return features

        df = pd.DataFrame(data).reset_index(level=TIMESTAMP)
        timestamps = pd.DatetimeIndex(df.pop(TIMESTAMP))
        features = df.groupby(level=ITEMID, sort=False, group_keys=False).apply(get_lag_features_and_target)

        for time_feat in self._time_features:
            features[time_feat.__name__] = time_feat(timestamps)

        if last_k_values is not None:
            features = features.groupby(level=ITEMID, sort=False, group_keys=False).tail(last_k_values)

        if data.static_features is not None:
            features = pd.merge(features, data.static_features, how="left", on=ITEMID, suffixes=(None, "_static_feat"))

        features.reset_index(inplace=True, drop=True)
        return features

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        self._check_fit_params()
        start_time = time.time()
        if self.tabular_predictor._learner.is_fit:
            raise AssertionError(f"{self.name} predictor has already been fit!")
        verbosity = kwargs.get("verbosity", 2)
        self._target_lag_indices = np.array(get_lags_for_frequency(train_data.freq), dtype=np.int64)
        self._past_covariates_lag_indices = self._target_lag_indices
        self._known_covariates_lag_indices = np.concatenate([[0], self._target_lag_indices])
        self._time_features = time_features_from_frequency_str(train_data.freq)

        train_data, _ = self._normalize_targets(train_data)
        train_df = self._get_features_dataframe(train_data)
        # Remove features that are completely missing in the training set
        train_df.dropna(axis=1, how="all", inplace=True)
        self._available_features = train_df.columns

        model_params = self._get_model_params()
        tabular_hyperparameters = model_params.get("tabular_hyperparameters", self.default_tabular_hyperparameters)
        max_train_size = model_params.get("max_train_size", 1_000_000)

        if len(train_df) > max_train_size:
            train_df = train_df.sample(max_train_size)
        logger.debug(f"Generated training dataframe with shape {train_df.shape}")

        if val_data is not None:
            if val_data.freq != train_data.freq:
                raise ValueError(
                    f"train_data and val_data must have the same freq (received {train_data.freq} and {val_data.freq})"
                )
            val_data, _ = self._normalize_targets(val_data)
            val_df = self._get_features_dataframe(val_data, last_k_values=self.prediction_length)
            val_df = val_df[self._available_features]

            if len(val_df) > max_train_size:
                val_df = val_df.sample(max_train_size)

            logger.debug(f"Generated validation dataframe with shape {val_df.shape}")
        else:
            logger.warning(
                f"No val_data was provided to {self.name}. "
                "TabularPredictor will generate a validation set without respecting the temporal ordering."
            )
            val_df = None

        time_elapsed = time.time() - start_time
        autogluon_logger = logging.getLogger("autogluon")
        logging_level = autogluon_logger.level
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tabular_predictor.fit(
                train_data=train_df,
                tuning_data=val_df,
                time_limit=time_limit - time_elapsed if time_limit else None,
                hyperparameters=tabular_hyperparameters,
                verbosity=verbosity - 2,
            )
        residuals = (train_df[self.target] - self.tabular_predictor.predict(train_df)).values
        for q in self.quantile_levels:
            self.quantile_adjustments[q] = np.quantile(residuals, q)
        # Logger level is changed inside .fit(), restore to the initial value
        autogluon_logger.setLevel(logging_level)

    def _extend_index(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Add self.prediction_length many time steps with dummy values to each timeseries in the dataset."""

        def extend_single_time_series(group):
            offset = pd.tseries.frequencies.to_offset(data.freq)
            cutoff = group.index.get_level_values(TIMESTAMP)[-1]
            new_index = pd.date_range(cutoff + offset, freq=offset, periods=self.prediction_length).rename(TIMESTAMP)
            new_values = np.full([self.prediction_length], fill_value=np.nan)
            new_df = pd.DataFrame(new_values, index=new_index, columns=[self.target])
            return pd.concat([group.droplevel(ITEMID), new_df])

        extended_data = data.groupby(level=ITEMID, sort=False).apply(extend_single_time_series)
        extended_data.static_features = data.static_features
        return extended_data

    def predict(self, data: TimeSeriesDataFrame, quantile_levels: List[float] = None, **kwargs) -> TimeSeriesDataFrame:
        self._check_predict_inputs(data=data, quantile_levels=quantile_levels)
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        data, scale_per_item = self._normalize_targets(data)
        data_extended = self._extend_index(data)
        features = self._get_features_dataframe(data_extended, last_k_values=self.prediction_length)
        features = features[self._available_features]

        # Predict for batches (instead of using full dataset) to avoid high memory usage
        batches = features.groupby(np.arange(len(features)) // self.PREDICTION_BATCH_SIZE, sort=False)
        predictions = pd.concat([self.tabular_predictor.predict(batch) for _, batch in batches])

        predictions = predictions.rename("mean").to_frame()
        preds_index = data_extended.slice_by_timestep(-self.prediction_length, None).index
        predictions.set_index(preds_index, inplace=True)

        for q in quantile_levels:
            predictions[str(q)] = predictions["mean"] + self.quantile_adjustments[q]

        predictions = self._rescale_targets(predictions, scale_per_item)
        return TimeSeriesDataFrame(predictions).loc[data.item_ids]

    def _normalize_targets(self, data: TimeSeriesDataFrame, min_scale=1e-5) -> Tuple[TimeSeriesDataFrame, pd.Series]:
        """Normalize data such that each the average absolute value of each time series is equal to 1."""
        # TODO: Implement other scalers (min/max)?
        # TODO: Don't include validation data when computing the scale
        scale_per_item = data.abs().groupby(level=ITEMID, sort=False)[self.target].mean().clip(lower=min_scale)
        normalized_data = data.copy()
        for col in normalized_data.columns:
            normalized_data[col] = normalized_data[col] / scale_per_item
        return normalized_data, scale_per_item

    def _rescale_targets(self, normalized_data: TimeSeriesDataFrame, scale_per_item: pd.Series) -> TimeSeriesDataFrame:
        """Scale all columns in the normalized dataframe back to original scale (inplace)."""
        data = normalized_data
        for col in data.columns:
            data[col] = data[col] * scale_per_item
        return data
