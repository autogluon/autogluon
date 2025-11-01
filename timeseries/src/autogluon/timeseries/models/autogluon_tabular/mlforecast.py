import copy
import logging
import math
import time
import warnings
from typing import Any, Callable, Collection, Optional, Type, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import autogluon.core as ag
from autogluon.core.models import AbstractModel as AbstractTabularModel
from autogluon.features import AutoMLPipelineFeatureGenerator
from autogluon.tabular.registry import ag_model_registry
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.metrics.abstract import TimeSeriesScorer
from autogluon.timeseries.metrics.utils import in_sample_squared_seasonal_error
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.models.local import SeasonalNaiveModel
from autogluon.timeseries.utils.datetime import (
    get_lags_for_frequency,
    get_seasonality,
    get_time_features_for_frequency,
)
from autogluon.timeseries.utils.warning_filters import set_loggers_level, warning_filter

from .utils import MLF_ITEMID, MLF_TARGET, MLF_TIMESTAMP

logger = logging.getLogger(__name__)


class TabularModel(BaseEstimator):
    """A scikit-learn compatible wrapper for arbitrary autogluon.tabular models"""

    def __init__(self, model_class: Type[AbstractTabularModel], model_kwargs: Optional[dict] = None):
        self.model_class = model_class
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        self.feature_pipeline = AutoMLPipelineFeatureGenerator(verbosity=0)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, **kwargs):
        self.model = self.model_class(**self.model_kwargs)
        X = self.feature_pipeline.fit_transform(X=X)
        X_val = self.feature_pipeline.transform(X=X_val)
        self.model.fit(X=X, y=y, X_val=X_val, y_val=y_val, **kwargs)
        return self

    def predict(self, X: pd.DataFrame, **kwargs):
        X = self.feature_pipeline.transform(X=X)
        return self.model.predict(X=X, **kwargs)

    def get_params(self, deep=True):
        params = {"model_class": self.model_class, "model_kwargs": self.model_kwargs}
        if deep:
            return copy.deepcopy(params)
        else:
            return params


class AbstractMLForecastModel(AbstractTimeSeriesModel):
    _supports_known_covariates = True
    _supports_static_features = True

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: Optional[Union[str, TimeSeriesScorer]] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
        **kwargs,
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
        from mlforecast import MLForecast
        from mlforecast.target_transforms import BaseTargetTransform

        self._sum_of_differences: int = 0  # number of time steps removed from each series by differencing
        self._max_ts_length: Optional[int] = None
        self._target_lags: np.ndarray
        self._date_features: list[Callable]
        self._mlf: MLForecast
        self._scaler: Optional[BaseTargetTransform] = None
        self._residuals_std_per_item: pd.Series
        self._train_target_median: Optional[float] = None
        self._non_boolean_real_covariates: list[str] = []
        self._known_covariate_name_map: dict[str, str] = {}
        self._sanitized_to_original_covariate_map: dict[str, str] = {}

    def _initialize_transforms_and_regressor(self):
        super()._initialize_transforms_and_regressor()
        # Do not create a scaler in the model, scaler will be passed to MLForecast
        self.target_scaler = None

    def _ensure_known_covariate_name_map(self) -> None:
        if self.covariate_metadata is None or len(self.covariate_metadata.known_covariates) == 0:
            self._known_covariate_name_map = {}
            self._sanitized_to_original_covariate_map = {}
            return

        known_covariates = list(self.covariate_metadata.known_covariates)
        if set(self._known_covariate_name_map.keys()) == set(known_covariates):
            return

        prefix = "__known_covariate__"
        self._known_covariate_name_map = {col: f"{prefix}{col}" for col in known_covariates}
        self._sanitized_to_original_covariate_map = {
            sanitized: original for original, sanitized in self._known_covariate_name_map.items()
        }

    @property
    def allowed_hyperparameters(self) -> list[str]:
        return super().allowed_hyperparameters + [
            "lags",
            "date_features",
            "differences",
            "model_name",
            "model_hyperparameters",
            "max_num_items",
            "max_num_samples",
            "lag_transforms",
        ]

    def preprocess(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        is_train: bool = False,
        **kwargs,
    ) -> tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        if is_train:
            # All-NaN series are removed; partially-NaN series in train_data are handled inside _generate_train_val_dfs
            all_nan_items = data.item_ids[
                data[self.target].isna().groupby(TimeSeriesDataFrame.ITEMID, sort=False).all()
            ]
            if len(all_nan_items):
                data = data.query("item_id not in @all_nan_items")
        else:
            data = data.fill_missing_values()
            # Fill time series consisting of all NaNs with the median of target in train_data
            if data.isna().any(axis=None):
                data[self.target] = data[self.target].fillna(value=self._train_target_median)
        return data, known_covariates

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        return {
            "max_num_items": 20_000,
            "max_num_samples": 1_000_000,
            "model_name": "GBM",
            "model_hyperparameters": {},
        }

    def _create_tabular_model(self, model_name: str, model_hyperparameters: dict[str, Any]) -> TabularModel:
        raise NotImplementedError

    def _get_mlforecast_init_args(
        self, train_data: TimeSeriesDataFrame, model_params: dict[str, Any]
    ) -> dict[str, Any]:
        from mlforecast.target_transforms import Differences

        from .transforms import MLForecastScaler

        lags = model_params.get("lags")
        if lags is None:
            assert self.freq is not None
            lags = get_lags_for_frequency(self.freq)
        self._target_lags = np.array(sorted(set(lags)), dtype=np.int64)

        date_features = model_params.get("date_features")
        if date_features is None:
            date_features = get_time_features_for_frequency(self.freq)
        self._date_features = date_features

        target_transforms = []
        differences = model_params.get("differences")
        assert isinstance(differences, Collection)

        ts_lengths = train_data.num_timesteps_per_item()
        required_ts_length = sum(differences) + 1
        all_train_ts_are_long_enough = ts_lengths.min() >= required_ts_length
        some_ts_available_for_validation = ts_lengths.max() >= required_ts_length + self.prediction_length
        if not (all_train_ts_are_long_enough and some_ts_available_for_validation):
            logger.warning(
                f"\tTime series in the dataset are too short for chosen differences {differences}. "
                f"Setting differences to [1]."
            )
            differences = [1]

        if len(differences) > 0:
            target_transforms.append(Differences(differences))
            self._sum_of_differences = sum(differences)

        if "target_scaler" in model_params and "scaler" in model_params:
            warnings.warn(
                f"Both 'target_scaler' and 'scaler' hyperparameters are provided to {self.__class__.__name__}. "
                "Please only set the 'target_scaler' parameter."
            )
        # Support "scaler" for backward compatibility
        scaler_type = model_params.get("target_scaler", model_params.get("scaler"))
        if scaler_type is not None:
            self._scaler = MLForecastScaler(scaler_type=scaler_type)
            target_transforms.append(self._scaler)

        return {
            "lags": self._target_lags.tolist(),
            "date_features": self._date_features,
            "target_transforms": target_transforms,
            "lag_transforms": model_params.get("lag_transforms"),
        }

    def _mask_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a mask that mimics the situation at prediction time when target/covariates are unknown during the
        forecast horizon.

        This method is overridden by DirectTabularModel.
        """
        return df

    @staticmethod
    def _shorten_all_series(mlforecast_df: pd.DataFrame, max_length: int) -> pd.DataFrame:
        logger.debug(f"Shortening all series to at most {max_length}")
        return mlforecast_df.groupby(MLF_ITEMID, as_index=False, sort=False).tail(max_length)

    def _generate_train_val_dfs(
        self, data: TimeSeriesDataFrame, max_num_items: Optional[int] = None, max_num_samples: Optional[int] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Exclude items that are too short for chosen differences - otherwise exception will be raised
        if self._sum_of_differences > 0:
            ts_lengths = data.num_timesteps_per_item()
            items_to_exclude = ts_lengths.index[ts_lengths <= self._sum_of_differences]
            if len(items_to_exclude) > 0:
                logger.debug(f"Removing {len(items_to_exclude)} items that are too short for chosen differences")
                data = data.query("item_id not in @items_to_exclude")

        if max_num_items is not None and data.num_items > max_num_items:
            items_to_keep = data.item_ids.to_series().sample(n=int(max_num_items))  # noqa: F841
            data = data.query("item_id in @items_to_keep")

        # MLForecast.preprocess does not support missing values, but we will exclude them later from the training set
        missing_entries = data.index[data[self.target].isna()]
        data = data.fill_missing_values()

        num_items = data.num_items
        mlforecast_df = self._to_mlforecast_df(data, data.static_features)

        # Shorten time series before calling preprocess to avoid high memory usage
        if max_num_samples is not None:
            max_samples_per_ts = max(200, math.ceil(max_num_samples / num_items))
            self._max_ts_length = max_samples_per_ts + self.prediction_length + self._sum_of_differences
            mlforecast_df = self._shorten_all_series(mlforecast_df, self._max_ts_length)

        # Unless we set static_features=[], MLForecast interprets all known covariates as static features
        df = self._mlf.preprocess(mlforecast_df, dropna=False, static_features=[])
        # df.query results in 2x memory saving compared to df.dropna(subset="y")
        df = df.query("y.notnull()")  # type: ignore

        df = self._mask_df(df)

        # We remove originally missing values filled via imputation from the training set
        if len(missing_entries):
            df = df.set_index(["unique_id", "ds"]).drop(missing_entries, errors="ignore").reset_index()

        if max_num_samples is not None and len(df) > max_num_samples:
            df = df.sample(n=max_num_samples)

        grouped_df = df.groupby(MLF_ITEMID, sort=False)

        # Use up to `prediction_length` last rows as validation set (but no more than 50% of the rows)
        val_rows_per_item = min(self.prediction_length, math.ceil(0.5 * len(df) / num_items))
        train_df = grouped_df.nth(slice(None, -val_rows_per_item))
        val_df = grouped_df.tail(val_rows_per_item)
        logger.debug(f"train_df shape: {train_df.shape}, val_df shape: {val_df.shape}")

        return train_df.drop(columns=[MLF_TIMESTAMP]), val_df.drop(columns=[MLF_TIMESTAMP])  # type: ignore

    def _to_mlforecast_df(
        self,
        data: TimeSeriesDataFrame,
        static_features: Optional[pd.DataFrame],
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Convert TimeSeriesDataFrame to a format expected by MLForecast methods `predict` and `preprocess`.

        Each row contains unique_id, ds, y, and (optionally) known covariates & static features.
        """
        # TODO: Add support for past_covariates
        selected_columns = self.covariate_metadata.known_covariates.copy()
        column_name_mapping = {TimeSeriesDataFrame.ITEMID: MLF_ITEMID, TimeSeriesDataFrame.TIMESTAMP: MLF_TIMESTAMP}
        self._ensure_known_covariate_name_map()
        column_name_mapping.update(self._known_covariate_name_map)
        if include_target:
            selected_columns += [self.target]
            column_name_mapping[self.target] = MLF_TARGET

        df = pd.DataFrame(data)[selected_columns].reset_index()
        if static_features is not None:
            df = pd.merge(
                df, static_features, how="left", on=TimeSeriesDataFrame.ITEMID, suffixes=(None, "_static_feat")
            )

        for sanitized_col in self._non_boolean_real_covariates:
            original_col = self._sanitized_to_original_covariate_map.get(sanitized_col, sanitized_col)
            # Normalize non-boolean features using mean_abs scaling
            df[f"__scaled_{sanitized_col}"] = (
                df[original_col]
                / df[original_col]
                .abs()
                .groupby(df[TimeSeriesDataFrame.ITEMID])
                .mean()
                .reindex(df[TimeSeriesDataFrame.ITEMID])
                .values
            )

        # Convert float64 to float32 to reduce memory usage
        float64_cols = list(df.select_dtypes(include="float64"))
        df[float64_cols] = df[float64_cols].astype("float32")

        # We assume that df is sorted by 'unique_id' inside `TimeSeriesPredictor._check_and_prepare_data_frame`
        return df.rename(columns=column_name_mapping)

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        from mlforecast import MLForecast

        self._check_fit_params()
        self._log_unused_hyperparameters()
        fit_start_time = time.time()
        self._train_target_median = train_data[self.target].median()
        self._ensure_known_covariate_name_map()
        self._non_boolean_real_covariates = []
        for col in self.covariate_metadata.known_covariates_real:
            if not set(train_data[col].unique()) == set([0, 1]):
                sanitized_name = self._known_covariate_name_map.get(col, col)
                self._non_boolean_real_covariates.append(sanitized_name)
        model_params = self.get_hyperparameters()

        mlforecast_init_args = self._get_mlforecast_init_args(train_data, model_params)
        assert self.freq is not None
        self._mlf = MLForecast(models={}, freq=self.freq, **mlforecast_init_args)

        # We generate train/val splits from train_data and ignore val_data to avoid overfitting
        train_df, val_df = self._generate_train_val_dfs(
            train_data,
            max_num_items=model_params["max_num_items"],
            max_num_samples=model_params["max_num_samples"],
        )

        with set_loggers_level(regex=r"^autogluon\.(tabular|features).*", level=logging.ERROR):
            tabular_model = self._create_tabular_model(
                model_name=model_params["model_name"], model_hyperparameters=model_params["model_hyperparameters"]
            )
            tabular_model.fit(
                X=train_df.drop(columns=[MLF_TARGET, MLF_ITEMID]),
                y=train_df[MLF_TARGET],
                X_val=val_df.drop(columns=[MLF_TARGET, MLF_ITEMID]),
                y_val=val_df[MLF_TARGET],
                time_limit=(None if time_limit is None else time_limit - (time.time() - fit_start_time)),
                verbosity=verbosity - 1,
            )

        # We directly insert the trained model into models_ since calling _mlf.fit_models does not support X_val, y_val
        self._mlf.models_ = {"mean": tabular_model}

        self._save_residuals_std(val_df)

    def get_tabular_model(self) -> TabularModel:
        """Get the underlying tabular regression model."""
        assert "mean" in self._mlf.models_, "Call `fit` before calling `get_tabular_model`"
        mean_estimator = self._mlf.models_["mean"]
        assert isinstance(mean_estimator, TabularModel)
        return mean_estimator

    def _save_residuals_std(self, val_df: pd.DataFrame) -> None:
        """Compute standard deviation of residuals for each item using the validation set.

        Saves per-item residuals to `self.residuals_std_per_item`.
        """
        residuals_df = val_df[[MLF_ITEMID, MLF_TARGET]]
        mean_estimator = self.get_tabular_model()

        residuals_df = residuals_df.assign(y_pred=mean_estimator.predict(val_df))
        if self._scaler is not None:
            # Scaler expects to find column MLF_TIMESTAMP even though it's not used - fill with dummy
            residuals_df = residuals_df.assign(**{MLF_TIMESTAMP: np.datetime64("2010-01-01")})
            residuals_df = self._scaler.inverse_transform(residuals_df)

        assert isinstance(residuals_df, pd.DataFrame)
        residuals = residuals_df[MLF_TARGET] - residuals_df["y_pred"]
        self._residuals_std_per_item = (
            residuals.pow(2.0).groupby(val_df[MLF_ITEMID].values, sort=False).mean().pow(0.5)  # type: ignore
        )

    def _remove_short_ts_and_generate_fallback_forecast(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
    ) -> tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame], Optional[TimeSeriesDataFrame]]:
        """Remove series that are too short for chosen differencing from data and generate naive forecast for them.

        Returns
        -------
        data_long
            Data containing only time series that are long enough for the model to predict.
        known_covariates_long
            Future known covariates containing only time series that are long enough for the model to predict.
        forecast_for_short_series
            Seasonal naive forecast for short series, if there are any in the dataset.
        """
        ts_lengths = data.num_timesteps_per_item()
        short_series = ts_lengths.index[ts_lengths <= self._sum_of_differences]
        if len(short_series) > 0:
            logger.warning(
                f"Warning: {len(short_series)} time series ({len(short_series) / len(ts_lengths):.1%}) are shorter "
                f"than {self._sum_of_differences} and cannot be predicted by {self.name}. "
                "Fallback model SeasonalNaive is used for these time series."
            )
            data_short = data.query("item_id in @short_series")
            seasonal_naive = SeasonalNaiveModel(
                freq=self.freq,
                prediction_length=self.prediction_length,
                target=self.target,
                quantile_levels=self.quantile_levels,
            )
            seasonal_naive.fit(train_data=data_short)
            forecast_for_short_series = seasonal_naive.predict(data_short)

            data_long = data.query("item_id not in @short_series")
            if known_covariates is not None:
                known_covariates_long = known_covariates.query("item_id not in @short_series")
            else:
                known_covariates_long = None
        else:
            data_long = data
            known_covariates_long = known_covariates
            forecast_for_short_series = None
        return data_long, known_covariates_long, forecast_for_short_series

    def _add_gaussian_quantiles(
        self, predictions: pd.DataFrame, repeated_item_ids: pd.Series, past_target: pd.Series
    ) -> pd.DataFrame:
        """
        Add quantile levels assuming that residuals follow normal distribution
        """
        from scipy.stats import norm

        num_items = int(len(predictions) / self.prediction_length)
        sqrt_h = np.sqrt(np.arange(1, self.prediction_length + 1))
        # Series where normal_scale_per_timestep.loc[item_id].loc[N] = sqrt(1 + N) for N in range(prediction_length)
        normal_scale_per_timestep = pd.Series(np.tile(sqrt_h, num_items), index=repeated_item_ids)

        residuals_std_per_timestep = self._residuals_std_per_item.reindex(repeated_item_ids)
        # Use in-sample seasonal error in for items not seen during fit
        items_not_seen_during_fit = residuals_std_per_timestep.index[residuals_std_per_timestep.isna()].unique()
        if len(items_not_seen_during_fit) > 0:
            scale_for_new_items: pd.Series = in_sample_squared_seasonal_error(
                y_past=past_target.loc[items_not_seen_during_fit]
            ).pow(0.5)
            residuals_std_per_timestep = residuals_std_per_timestep.fillna(scale_for_new_items)

        std_per_timestep = residuals_std_per_timestep * normal_scale_per_timestep
        for q in self.quantile_levels:
            predictions[str(q)] = predictions["mean"] + norm.ppf(q) * std_per_timestep.to_numpy()
        return predictions

    def _more_tags(self) -> dict[str, Any]:
        return {"allow_nan": True, "can_refit_full": True}


class DirectTabularModel(AbstractMLForecastModel):
    """Predict all future time series values simultaneously using a regression model from AutoGluon-Tabular.

    A single tabular model is used to forecast all future time series values using the following features:

    - lag features (observed time series values) based on ``freq`` of the data
    - time features (e.g., day of the week) based on the timestamp of the measurement
    - known covariates (if available)
    - static features of each item (if available)

    Features not known during the forecast horizon (e.g., future target values) are replaced by NaNs.

    If ``eval_metric.needs_quantile``, the tabular regression model will be trained with ``"quantile"`` problem type.
    Otherwise, the model will be trained with ``"regression"`` problem type, and dummy quantiles will be
    obtained by assuming that the residuals follow zero-mean normal distribution.

    Based on the `mlforecast <https://github.com/Nixtla/mlforecast>`_ library.


    Other Parameters
    ----------------
    lags : list[int], default = None
        Lags of the target that will be used as features for predictions. If None, will be determined automatically
        based on the frequency of the data.
    date_features : list[Union[str, Callable]], default = None
        Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input.
        If None, will be determined automatically based on the frequency of the data.
    differences : list[int], default = []
        Differences to take of the target before computing the features. These are restored at the forecasting step.
        Defaults to no differencing.
    target_scaler : {"standard", "mean_abs", "min_max", "robust", None}, default = "mean_abs"
        Scaling applied to each time series. Scaling is applied after differencing.
    model_name : str, default = "GBM"
        Name of the tabular regression model. See ``autogluon.tabular.registry.ag_model_registry`` or
        `the documentation <https://auto.gluon.ai/stable/api/autogluon.tabular.models.html>`_ for the list of available
        tabular models.
    model_hyperparameters : dict[str, Any], optional
        Hyperparameters passed to the tabular regression model.
    max_num_items : int or None, default = 20_000
        If not None, the model will randomly select this many time series for training and validation.
    max_num_samples : int or None, default = 1_000_000
        If not None, training dataset passed to the tabular regression model will contain at most this many rows
        (starting from the end of each time series).
    """

    ag_priority = 85

    @property
    def is_quantile_model(self) -> bool:
        return self.eval_metric.needs_quantile

    def get_hyperparameters(self) -> dict[str, Any]:
        model_params = super().get_hyperparameters()
        # We don't set 'target_scaler' if user already provided 'scaler' to avoid overriding the user-provided value
        if "scaler" not in model_params:
            model_params.setdefault("target_scaler", "mean_abs")
        if "differences" not in model_params or model_params["differences"] is None:
            model_params["differences"] = []
        if "lag_transforms" in model_params:
            model_params.pop("lag_transforms")
            logger.warning(f"{self.name} does not support the 'lag_transforms' hyperparameter.")
        return model_params

    def _mask_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a mask that mimics the situation at prediction time when target/covariates are unknown during the
        forecast horizon.
        """
        # Fix seed to make the model deterministic
        rng = np.random.default_rng(seed=123)
        num_hidden = rng.integers(0, self.prediction_length, size=len(df))
        lag_cols = [f"lag{lag}" for lag in self._target_lags]
        mask = num_hidden[:, None] < self._target_lags[None]  # shape [len(num_hidden), len(_target_lags)]
        # use df.loc[:, lag_cols] instead of df[lag_cols] to avoid SettingWithCopyWarning
        df.loc[:, lag_cols] = df[lag_cols].where(mask, other=np.nan)
        return df

    def _save_residuals_std(self, val_df: pd.DataFrame) -> None:
        if self.is_quantile_model:
            # Quantile model does not require residuals to produce prediction intervals
            self._residuals_std_per_item = pd.Series(1.0, index=val_df[MLF_ITEMID].unique())
        else:
            super()._save_residuals_std(val_df=val_df)

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        from .transforms import apply_inverse_transform

        original_item_id_order = data.item_ids
        data, known_covariates, forecast_for_short_series = self._remove_short_ts_and_generate_fallback_forecast(
            data=data, known_covariates=known_covariates
        )
        if len(data) == 0:
            # All time series are too short for chosen differences
            assert forecast_for_short_series is not None
            return forecast_for_short_series

        if known_covariates is not None:
            data_future = known_covariates.copy()
        else:
            future_index = self.get_forecast_horizon_index(data)
            data_future = pd.DataFrame(columns=[self.target], index=future_index, dtype="float32")
        # MLForecast raises exception of target contains NaN. We use inf as placeholder, replace them by NaN afterwards
        data_future[self.target] = float("inf")
        data_extended = pd.concat([data, data_future])
        mlforecast_df = self._to_mlforecast_df(data_extended, data.static_features)  # type: ignore
        if self._max_ts_length is not None:
            # We appended `prediction_length` time steps to each series, so increase length
            mlforecast_df = self._shorten_all_series(mlforecast_df, self._max_ts_length + self.prediction_length)
        df = self._mlf.preprocess(mlforecast_df, dropna=False, static_features=[])
        assert isinstance(df, pd.DataFrame)

        df = df.groupby(MLF_ITEMID, sort=False).tail(self.prediction_length)
        df = df.replace(float("inf"), float("nan"))

        mean_estimator = self.get_tabular_model()
        raw_predictions = mean_estimator.predict(df)
        predictions = self._postprocess_predictions(raw_predictions, repeated_item_ids=df[MLF_ITEMID])
        # Paste columns one by one to preserve dtypes
        predictions[MLF_ITEMID] = df[MLF_ITEMID].values
        predictions[MLF_TIMESTAMP] = df[MLF_TIMESTAMP].values

        if hasattr(self._mlf.ts, "target_transforms"):
            # Ensure that transforms are fitted only on past data
            mlforecast_df_past = self._to_mlforecast_df(data, None)
            if self._max_ts_length is not None:
                mlforecast_df_past = self._shorten_all_series(mlforecast_df_past, self._max_ts_length)
            self._mlf.preprocess(mlforecast_df_past, static_features=[], dropna=False)
            assert self._mlf.ts.target_transforms is not None
            for tfm in self._mlf.ts.target_transforms[::-1]:
                predictions = apply_inverse_transform(predictions, transform=tfm)

        if not self.is_quantile_model:
            predictions = self._add_gaussian_quantiles(
                predictions, repeated_item_ids=predictions[MLF_ITEMID], past_target=data[self.target]
            )
        predictions_tsdf: TimeSeriesDataFrame = TimeSeriesDataFrame(
            predictions.rename(
                columns={MLF_ITEMID: TimeSeriesDataFrame.ITEMID, MLF_TIMESTAMP: TimeSeriesDataFrame.TIMESTAMP}
            )
        )

        if forecast_for_short_series is not None:
            predictions_tsdf = pd.concat([predictions_tsdf, forecast_for_short_series])  # type: ignore
            predictions_tsdf = predictions_tsdf.reindex(original_item_id_order, level=TimeSeriesDataFrame.ITEMID)

        return predictions_tsdf

    def _postprocess_predictions(
        self, predictions: Union[np.ndarray, pd.Series], repeated_item_ids: pd.Series
    ) -> pd.DataFrame:
        if self.is_quantile_model:
            predictions_df = pd.DataFrame(predictions, columns=[str(q) for q in self.quantile_levels])
            predictions_df.values.sort(axis=1)
            predictions_df["mean"] = predictions_df["0.5"]
        else:
            predictions_df = pd.DataFrame(predictions, columns=["mean"])

        column_order = ["mean"] + [col for col in predictions_df.columns if col != "mean"]
        return predictions_df[column_order]

    def _create_tabular_model(self, model_name: str, model_hyperparameters: dict[str, Any]) -> TabularModel:
        model_class = ag_model_registry.key_to_cls(model_name)
        if self.is_quantile_model:
            problem_type = ag.constants.QUANTILE
            eval_metric = "pinball_loss"
            model_hyperparameters["ag.quantile_levels"] = self.quantile_levels
        else:
            problem_type = ag.constants.REGRESSION
            eval_metric = self.eval_metric.equivalent_tabular_regression_metric or "mean_absolute_error"
        return TabularModel(
            model_class=model_class,
            model_kwargs={
                "path": "",
                "name": model_class.__name__,
                "hyperparameters": model_hyperparameters,
                "problem_type": problem_type,
                "eval_metric": eval_metric,
            },
        )


class RecursiveTabularModel(AbstractMLForecastModel):
    """Predict future time series values one by one using a regression model from AutoGluon-Tabular.

    A single tabular regression model is used to forecast the future time series values using the following features:

    - lag features (observed time series values) based on ``freq`` of the data
    - time features (e.g., day of the week) based on the timestamp of the measurement
    - known covariates (if available)
    - static features of each item (if available)

    The tabular model will always be trained with ``"regression"`` problem type, and dummy quantiles will be
    obtained by assuming that the residuals follow zero-mean normal distribution.

    Based on the `mlforecast <https://github.com/Nixtla/mlforecast>`_ library.


    Other Parameters
    ----------------
    lags : list[int], default = None
        Lags of the target that will be used as features for predictions. If None, will be determined automatically
        based on the frequency of the data.
    date_features : list[Union[str, Callable]], default = None
        Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input.
        If None, will be determined automatically based on the frequency of the data.
    differences : list[int], default = None
        Differences to take of the target before computing the features. These are restored at the forecasting step.
        If None, will be set to ``[seasonal_period]``, where seasonal_period is determined based on the data frequency.
    target_scaler : {"standard", "mean_abs", "min_max", "robust", None}, default = "standard"
        Scaling applied to each time series. Scaling is applied after differencing.
    lag_transforms : dict[int, list[Callable]], default = None
        Dictionary mapping lag periods to transformation functions applied to lagged target values (e.g., rolling mean).
        See `MLForecast documentation <https://nixtlaverse.nixtla.io/mlforecast/lag_transforms.html>`_ for more details.
    model_name : str, default = "GBM"
        Name of the tabular regression model. See ``autogluon.tabular.registry.ag_model_registry`` or
        `the documentation <https://auto.gluon.ai/stable/api/autogluon.tabular.models.html>`_ for the list of available
        tabular models.
    model_hyperparameters : dict[str, Any], optional
        Hyperparameters passed to the tabular regression model.
    max_num_items : int or None, default = 20_000
        If not None, the model will randomly select this many time series for training and validation.
    max_num_samples : int or None, default = 1_000_000
        If not None, training dataset passed to the tabular regression model will contain at most this many rows
        (starting from the end of each time series).
    """

    ag_priority = 90

    def get_hyperparameters(self) -> dict[str, Any]:
        model_params = super().get_hyperparameters()
        # We don't set 'target_scaler' if user already provided 'scaler' to avoid overriding the user-provided value
        if "scaler" not in model_params:
            model_params.setdefault("target_scaler", "standard")
        if "differences" not in model_params or model_params["differences"] is None:
            model_params["differences"] = [get_seasonality(self.freq)]
        return model_params

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        original_item_id_order = data.item_ids
        data, known_covariates, forecast_for_short_series = self._remove_short_ts_and_generate_fallback_forecast(
            data=data, known_covariates=known_covariates
        )
        if len(data) == 0:
            # All time series are too short for chosen differences
            assert forecast_for_short_series is not None
            return forecast_for_short_series

        new_df = self._to_mlforecast_df(data, data.static_features)
        if self._max_ts_length is not None:
            new_df = self._shorten_all_series(new_df, self._max_ts_length)
        if known_covariates is None:
            future_index = self.get_forecast_horizon_index(data)
            known_covariates = TimeSeriesDataFrame(
                pd.DataFrame(columns=[self.target], index=future_index, dtype="float32")
            )
        X_df = self._to_mlforecast_df(known_covariates, data.static_features, include_target=False)
        # If both covariates & static features are missing, set X_df = None to avoid exception from MLForecast
        if len(X_df.columns.difference([MLF_ITEMID, MLF_TIMESTAMP])) == 0:
            X_df = None
        with warning_filter():
            raw_predictions = self._mlf.predict(
                h=self.prediction_length,
                new_df=new_df,
                X_df=X_df,
            )
        assert isinstance(raw_predictions, pd.DataFrame)
        raw_predictions = raw_predictions.rename(
            columns={MLF_ITEMID: TimeSeriesDataFrame.ITEMID, MLF_TIMESTAMP: TimeSeriesDataFrame.TIMESTAMP}
        )

        predictions: TimeSeriesDataFrame = TimeSeriesDataFrame(
            self._add_gaussian_quantiles(
                raw_predictions,
                repeated_item_ids=raw_predictions[TimeSeriesDataFrame.ITEMID],
                past_target=data[self.target],
            )
        )
        if forecast_for_short_series is not None:
            predictions = pd.concat([predictions, forecast_for_short_series])  # type: ignore
        return predictions.reindex(original_item_id_order, level=TimeSeriesDataFrame.ITEMID)

    def _create_tabular_model(self, model_name: str, model_hyperparameters: dict[str, Any]) -> TabularModel:
        model_class = ag_model_registry.key_to_cls(model_name)
        return TabularModel(
            model_class=model_class,
            model_kwargs={
                "path": "",
                "name": model_class.__name__,
                "hyperparameters": model_hyperparameters,
                "problem_type": ag.constants.REGRESSION,
                "eval_metric": self.eval_metric.equivalent_tabular_regression_metric or "mean_absolute_error",
            },
        )
