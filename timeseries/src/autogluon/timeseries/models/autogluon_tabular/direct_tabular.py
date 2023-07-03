import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# TODO: Drop GluonTS dependency
from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str
from joblib.parallel import Parallel, delayed

import autogluon.core as ag
from autogluon.tabular import TabularPredictor
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.models.local.abstract_local_model import AG_DEFAULT_N_JOBS
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
from autogluon.timeseries.utils.warning_filters import statsmodels_joblib_warning_filter, statsmodels_warning_filter

logger = logging.getLogger(__name__)


class DirectTabularModel(AbstractTimeSeriesModel):
    """Predict all future time series values simultaneously using TabularPredictor from AutoGluon-Tabular.

    A single TabularPredictor is used to forecast all future time series values using the following features:

    - lag features (observed time series values) based on ``freq`` of the data
    - time features (e.g., day of the week) based on the timestamp of the measurement
    - lagged known and past covariates (if available)
    - static features of each item (if available)

    Features not known during the forecast horizon (e.g., future target values) are replaced by NaNs.

    If ``eval_metric=="mean_wQuantileLoss"``, the TabularPredictor will be trained with ``"quantile"`` problem type.
    Otherwise, TabularPredictor will be trained with ``"regression"`` problem type, and dummy quantiles will be
    obtained by assuming that the residuals follow zero-mean normal distribution.


    Other Parameters
    ----------------
    max_num_samples : int, default = 1_000_000
        Maximum number of rows in the training and validation sets. If the number of rows in train or validation data
        exceeds ``max_num_samples``, then ``max_num_samples`` many rows are subsampled from the dataframe.
    tabular_hyperparameters : Dict[Dict[str, Any]], optional
        Hyperparameters dictionary passed to `TabularPredictor.fit`. Contains the names of models that should be fit.
        Defaults to ``{"GBM" :{}}``.
    """

    # TODO: Implemented detrending/differencing to allow extrapolation.

    default_tabular_hyperparameters = {
        "GBM": {},
    }

    PREDICTION_BATCH_SIZE = 100_000

    TIMESERIES_METRIC_TO_TABULAR_METRIC = {
        "MASE": "mean_absolute_error",
        "MAPE": "mean_absolute_percentage_error",
        "sMAPE": "mean_absolute_percentage_error",
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
        self.is_quantile_model = self.eval_metric == "mean_wQuantileLoss"
        if 0.5 not in self.quantile_levels:
            self.must_drop_median = True
            self.quantile_levels = sorted(set([0.5] + self.quantile_levels))
        else:
            self.must_drop_median = False
        self.residuals_std = 1.0
        self.tabular_predictor: TabularPredictor = None

    def _normalize_targets(self, data: TimeSeriesDataFrame, min_scale=1e-5) -> Tuple[TimeSeriesDataFrame, pd.Series]:
        """Normalize data such that each the average absolute value of each time series is equal to 1."""
        # TODO: Implement other scalers (min/max)?
        # TODO: Don't include validation data when computing the scale
        scale_per_item = data.abs().groupby(level=ITEMID, sort=False)[self.target].mean().clip(lower=min_scale)
        normalized_data = data.copy()
        normalized_data[self.target] = normalized_data[self.target] / scale_per_item
        return normalized_data, scale_per_item

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
            with statsmodels_joblib_warning_filter(), statsmodels_warning_filter():
                lags_per_item = Parallel(n_jobs=AG_DEFAULT_N_JOBS)(
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
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        self._check_fit_params()
        start_time = time.time()
        if self.tabular_predictor is not None:
            raise AssertionError(f"{self.name} predictor has already been fit!")
        self._target_lag_indices = np.array(get_lags_for_frequency(train_data.freq), dtype=np.int64)
        self._past_covariates_lag_indices = self._target_lag_indices
        self._known_covariates_lag_indices = np.concatenate([[0], self._target_lag_indices])
        self._time_features = time_features_from_frequency_str(train_data.freq)

        train_data, _ = self._normalize_targets(train_data)
        # Do not use external val_data as tuning_data to avoid overfitting
        train_subset, val_subset = train_data.train_test_split(self.prediction_length)
        train_df = self._get_features_dataframe(train_subset)
        val_df = self._get_features_dataframe(val_subset, max_rows_per_item=self.prediction_length)

        model_params = self._get_model_params()
        tabular_hyperparameters = model_params.get("tabular_hyperparameters", self.default_tabular_hyperparameters)
        max_num_samples = model_params.get("max_num_samples", 1_000_000)

        if len(train_df) > max_num_samples:
            train_df = train_df.sample(max_num_samples)
        logger.debug(f"Generated training dataframe with shape {train_df.shape}")

        time_elapsed = time.time() - start_time
        autogluon_logger = logging.getLogger("autogluon")
        logging_level = autogluon_logger.level

        if self.is_quantile_model:
            predictor_init_kwargs = {
                "problem_type": ag.constants.QUANTILE,
                "eval_metric": "pinball_loss",
                "quantile_levels": self.quantile_levels,
            }
        else:
            predictor_init_kwargs = {
                "problem_type": ag.constants.REGRESSION,
                "eval_metric": self.TIMESERIES_METRIC_TO_TABULAR_METRIC.get(self.eval_metric),
            }

        self.tabular_predictor = TabularPredictor(
            path=self.path,
            label=self.target,
            **predictor_init_kwargs,
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

        if not self.is_quantile_model:
            residuals = (train_df[self.target] - self.tabular_predictor.predict(train_df)).values
            self.residuals_std = np.sqrt(np.mean(residuals**2))

        # Logger level is changed inside .fit(), restore to the initial value
        autogluon_logger.setLevel(logging_level)

    def _postprocess_predictions(self, predictions: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Convert output of TabularPredictor to a dataframe with 'mean' and quantile forecast columns."""
        from scipy.stats import norm

        if self.is_quantile_model:
            # Ensure that quantiles are monotonic
            predictions.values.sort(axis=1)
            predictions.columns = [str(q) for q in predictions.columns]
            predictions["mean"] = predictions["0.5"]
        else:
            predictions = predictions.rename("mean").to_frame()
            for q in self.quantile_levels:
                predictions[str(q)] = predictions["mean"] + norm.ppf(q) * self.residuals_std

        column_order = ["mean"] + [col for col in predictions.columns if col != "mean"]
        return predictions[column_order]

    def predict(
        self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame] = None, **kwargs
    ) -> TimeSeriesDataFrame:
        data, scale_per_item = self._normalize_targets(data)
        if known_covariates is not None:
            data_future = known_covariates.copy()
            data_future[self.target] = np.nan
        else:
            future_index = get_forecast_horizon_index_ts_dataframe(data, self.prediction_length)
            data_future = pd.DataFrame(columns=[self.target], index=future_index)
        data_extended = pd.concat([data, data_future])
        data_extended.static_features = data.static_features

        features = self._get_features_dataframe(data_extended, max_rows_per_item=self.prediction_length)

        # Predict for batches (instead of using full dataset) to avoid high memory usage
        batches = features.groupby(np.arange(len(features)) // self.PREDICTION_BATCH_SIZE, sort=False)
        predictions = pd.concat([self.tabular_predictor.predict(batch) for _, batch in batches])
        predictions.index = data_future.index

        predictions = self._postprocess_predictions(predictions)

        for col in predictions.columns:
            predictions[col] = predictions[col] * scale_per_item
        return TimeSeriesDataFrame(predictions).reindex(data.item_ids, level=ITEMID)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
