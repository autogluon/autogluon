import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# TODO: Drop GluonTS dependency
from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str
from joblib.parallel import Parallel, delayed

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
    tabular_hyperparameters : Dict[Dict[str, Any]], optional
        Hyperparameters dictionary passed to `TabularPredictor.fit`. Contains the names of models that should be fit.
        Defaults to ``{"CAT": {}, "GBM" :{}}``.
    """

    default_tabular_hyperparameters = {
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

        self.tabular_predictor: TabularPredictor = None

    def _get_features_dataframe(
        self,
        data: TimeSeriesDataFrame,
        max_rows_per_item: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate a feature matrix used by TabularPredictor.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Dataframe containing features derived from time index & past time series values, as well as the target.
        max_rows_per_item: int, optional
            If given, features will be generated only for the last `max_rows_per_item` timesteps of each time series.
        """

        def apply_mask(array: np.ndarray, num_hidden: np.ndarray, lag_indices: np.ndarray) -> pd.DataFrame:
            """Apply a mask that mimics the situation at prediction time when target/covariates are unknown during the
            forecast horizon.

            Parameters
            ----------
            array
                Array to mask, shape [N, len(lag_indices)]
            num_hidden
                Number of entries hidden in each row, shape [N]
            lag_indices
                Lag indices used to construct the dataframe

            Returns
            -------
            masked_array
                Array with the masking applied, shape [N, D * len(lag_indices)]


            For example, given the following inputs

            array = [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
            num_hidden = [6, 0, 1]
            lag_indices = [1, 2, 5, 10]
            num_columns = 1

            The resulting masked output will be

            masked_array = [
                [NaN, NaN, NaN, 1],
                [1, 1, 1, 1],
                [NaN, 1, 1, 1],
            ]

            """
            mask = num_hidden[:, None] >= lag_indices[None]  # shape [len(num_hidden), len(lag_indices)]
            array[mask] = np.nan
            return array

        def get_lags(
            ts: np.ndarray,
            lag_indices: np.ndarray,
            prediction_length: int,
            max_rows_per_item: int = 100_000,
            mask: bool = False,
        ) -> np.ndarray:
            """Generate the matrix of lag features for a single time series.

            Parameters
            ----------
            ts
                Array with target or covariate values, shape [N]
            lag_indices
                Array with the lag indices to use for feature generation.
            prediction_length
                Length of the forecast horizon.
            max_rows_per_item
                Maximum number of rows to include in the feature matrix.
                If max_rows_per_item < len(ts), the lag features will be generated only
                for the *last* max_rows_per_item entries of ts.
            mask
                If True, a mask will be applied to some entries of the feature matrix,
                mimicking the behavior at prediction time, when the ts values are not
                known during the forecast horizon.

            Returns
            -------
            features
                Array with lag features, shape [min(N, max_rows_per_item), len(lag_indices)]
            """
            num_rows = min(max_rows_per_item, len(ts))
            features = np.full([num_rows, len(lag_indices)], fill_value=np.nan)
            for i in range(1, num_rows + 1):
                target_idx = len(ts) - i
                selected_lags = lag_indices[lag_indices <= target_idx]
                features[num_rows - i, np.arange(len(selected_lags))] = ts[target_idx - selected_lags]
            if mask:
                num_windows = (len(ts) - 1) // prediction_length
                # We don't hide any past values for the first `remainder` values, otherwise the features will be all empty
                remainder = len(ts) - num_windows * prediction_length
                num_hidden = np.concatenate([np.zeros(remainder), np.tile(np.arange(prediction_length), num_windows)])
                features = apply_mask(features, num_hidden[-num_rows:], lag_indices)
            return features

        def get_lag_features(
            all_series: List[np.ndarray],
            lag_indices: np.ndarray,
            prediction_length: int,
            max_rows_per_item: int,
            mask: bool,
            name: str,
        ):
            """Generate lag features for all time series in the dataset.

            See the docstring of get_lags for the description of the parameters.
            """
            # TODO: Expose n_jobs to the user as a hyperparameter
            lags_per_item = Parallel(n_jobs=-1)(
                delayed(get_lags)(
                    ts,
                    lag_indices,
                    prediction_length=prediction_length,
                    max_rows_per_item=max_rows_per_item,
                    mask=mask,
                )
                for ts in all_series
            )
            features = np.concatenate(lags_per_item)
            return pd.DataFrame(features, columns=[f"{name}_lag_{idx}" for idx in lag_indices])

        df = pd.DataFrame(data)
        all_series = [ts for _, ts in df.droplevel(TIMESTAMP).groupby(level=ITEMID, sort=False)]
        if max_rows_per_item is None:
            max_rows_per_item = data.num_timesteps_per_item().max()

        feature_dfs = []
        for column_name in df.columns:
            if column_name == self.target:
                mask = True
                lag_indices = self._target_lag_indices
            elif column_name in self.metadata.past_covariates_real:
                mask = True
                lag_indices = self._past_covariates_lag_indices
            elif column_name in self.metadata.known_covariates_real:
                mask = False
                lag_indices = self._known_covariates_lag_indices
            else:
                raise ValueError(f"Unexpected column {column_name} is not among target or covariates.")

            feature_dfs.append(
                get_lag_features(
                    [ts[column_name].to_numpy() for ts in all_series],
                    lag_indices=lag_indices,
                    prediction_length=self.prediction_length,
                    max_rows_per_item=max_rows_per_item,
                    mask=mask,
                    name=column_name,
                )
            )

        # Only the last max_rows_per_item entries for each item will be included in the feature matrix
        target_with_index = df[self.target].groupby(level=ITEMID, sort=False).tail(max_rows_per_item)
        feature_dfs.append(target_with_index.reset_index(drop=True))

        timestamps = target_with_index.index.get_level_values(level=TIMESTAMP)
        feature_dfs.append(
            pd.DataFrame({time_feat.__name__: time_feat(timestamps) for time_feat in self._time_features})
        )

        features = pd.concat(feature_dfs, axis=1)

        if data.static_features is not None:
            features.index = target_with_index.index.get_level_values(level=ITEMID)
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
        if self.tabular_predictor is not None:
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
            val_df = self._get_features_dataframe(val_data, max_rows_per_item=self.prediction_length)
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

        self.tabular_predictor = TabularPredictor(
            path=self.path,
            label=self.target,
            problem_type=ag.constants.REGRESSION,
            eval_metric=self.TIMESERIES_METRIC_TO_TABULAR_METRIC.get(self.eval_metric),
        )

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
        features = self._get_features_dataframe(data_extended, max_rows_per_item=self.prediction_length)
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

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
