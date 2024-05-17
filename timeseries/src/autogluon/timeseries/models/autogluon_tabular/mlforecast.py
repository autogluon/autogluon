import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import autogluon.core as ag
from autogluon.tabular import TabularPredictor
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.models.local import SeasonalNaiveModel
from autogluon.timeseries.utils.datetime import (
    get_lags_for_frequency,
    get_seasonality,
    get_time_features_for_frequency,
)
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
from autogluon.timeseries.utils.warning_filters import warning_filter

logger = logging.getLogger(__name__)

MLF_TARGET = "y"
MLF_ITEMID = "unique_id"
MLF_TIMESTAMP = "ds"


class TabularEstimator(BaseEstimator):
    """Scikit-learn compatible interface for TabularPredictor."""

    def __init__(self, predictor_init_kwargs: Optional[dict] = None, predictor_fit_kwargs: Optional[dict] = None):
        self.predictor_init_kwargs = predictor_init_kwargs if predictor_init_kwargs is not None else {}
        self.predictor_fit_kwargs = predictor_fit_kwargs if predictor_fit_kwargs is not None else {}

    def get_params(self, deep: bool = True) -> dict:
        return {
            "predictor_init_kwargs": self.predictor_init_kwargs,
            "predictor_fit_kwargs": self.predictor_fit_kwargs,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabularEstimator":
        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)
        df = pd.concat([X, y.rename(MLF_TARGET).to_frame()], axis=1)
        self.predictor = TabularPredictor(**self.predictor_init_kwargs)
        with warning_filter():
            self.predictor.fit(df, **self.predictor_fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert isinstance(X, pd.DataFrame)
        return self.predictor.predict(X).values


class AbstractMLForecastModel(AbstractTimeSeriesModel):
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
        from mlforecast import MLForecast
        from mlforecast.target_transforms import BaseTargetTransform

        self._sum_of_differences: int = 0  # number of time steps removed from each series by differencing
        self._max_ts_length: Optional[int] = None
        self._target_lags: Optional[List[int]] = None
        self._date_features: Optional[List[str]] = None
        self._mlf: Optional[MLForecast] = None
        self._scaler: Optional[BaseTargetTransform] = None
        self._residuals_std_per_item: Optional[pd.Series] = None
        self._avg_residuals_std: Optional[float] = None
        self._train_target_median: Optional[float] = None
        self._non_boolean_real_covariates: List[str] = []

    @property
    def tabular_predictor_path(self) -> str:
        return os.path.join(self.path, "tabular_predictor")

    def save(self, path: str = None, verbose: bool = True) -> str:
        assert "mean" in self._mlf.models_, "TabularPredictor must be trained before saving"
        tabular_predictor = self._mlf.models_["mean"].predictor
        self._mlf.models_["mean"].predictor = None
        save_path = super().save(path=path, verbose=verbose)
        self._mlf.models_["mean"].predictor = tabular_predictor
        return save_path

    @classmethod
    def load(
        cls, path: str, reset_paths: bool = True, load_oof: bool = False, verbose: bool = True
    ) -> "AbstractTimeSeriesModel":
        model = super().load(path=path, reset_paths=reset_paths, load_oof=load_oof, verbose=verbose)
        assert "mean" in model._mlf.models_, "Loaded model doesn't have a trained TabularPredictor"
        model._mlf.models_["mean"].predictor = TabularPredictor.load(model.tabular_predictor_path)
        return model

    def preprocess(self, data: TimeSeriesDataFrame, is_train: bool = False, **kwargs) -> Any:
        if is_train:
            # All-NaN series are removed; partially-NaN series in train_data are handled inside _generate_train_val_dfs
            all_nan_items = data.item_ids[data[self.target].isna().groupby(ITEMID, sort=False).all()]
            if len(all_nan_items):
                data = data.query("item_id not in @all_nan_items")
            return data
        else:
            data = data.fill_missing_values()
            # Fill time series consisting of all NaNs with the median of target in train_data
            if data.isna().any(axis=None):
                data[self.target] = data[self.target].fillna(value=self._train_target_median)
            return data

    def _get_extra_tabular_init_kwargs(self) -> dict:
        raise NotImplementedError

    def _get_model_params(self) -> dict:
        model_params = super()._get_model_params().copy()
        model_params.setdefault("max_num_items", 20_000)
        model_params.setdefault("max_num_samples", 1_000_000)
        model_params.setdefault("tabular_hyperparameters", {"GBM": {}})
        model_params.setdefault("tabular_fit_kwargs", {})
        return model_params

    def _get_mlforecast_init_args(self, train_data: TimeSeriesDataFrame, model_params: dict) -> dict:
        from mlforecast.target_transforms import Differences

        from .utils import MeanAbsScaler, StandardScaler

        lags = model_params.get("lags")
        if lags is None:
            lags = get_lags_for_frequency(self.freq)
        self._target_lags = np.array(sorted(set(lags)), dtype=np.int64)

        date_features = model_params.get("date_features")
        if date_features is None:
            date_features = get_time_features_for_frequency(self.freq)
        self._date_features = date_features

        target_transforms = []
        differences = model_params.get("differences")

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

        scaler_name = model_params.get("scaler")
        if scaler_name is None:
            pass
        elif scaler_name == "standard":
            self._scaler = StandardScaler()
        elif scaler_name == "mean_abs":
            self._scaler = MeanAbsScaler()
        else:
            logger.warning(
                f"Unrecognized `scaler` {scaler_name} (supported options: ['standard', 'mean_abs', None]). Scaling disabled."
            )

        if self._scaler is not None:
            target_transforms.append(self._scaler)

        return {
            "lags": self._target_lags.tolist(),
            "date_features": self._date_features,
            "target_transforms": target_transforms,
        }

    def _mask_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a mask that mimics the situation at prediction time when target/covariates are unknown during the
        forecast horizon.

        This method is overridden by DirectTabularModel.
        """
        return df

    @staticmethod
    def _shorten_all_series(mlforecast_df: pd.DataFrame, max_length: int):
        logger.debug(f"Shortening all series to at most {max_length}")
        return mlforecast_df.groupby(MLF_ITEMID, as_index=False, sort=False).tail(max_length)

    def _generate_train_val_dfs(
        self, data: TimeSeriesDataFrame, max_num_items: Optional[int] = None, max_num_samples: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        df = df.query("y.notnull()")

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

        return train_df.drop(columns=[MLF_TIMESTAMP]), val_df.drop(columns=[MLF_TIMESTAMP])

    def _to_mlforecast_df(
        self,
        data: TimeSeriesDataFrame,
        static_features: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Convert TimeSeriesDataFrame to a format expected by MLForecast methods `predict` and `preprocess`.

        Each row contains unique_id, ds, y, and (optionally) known covariates & static features.
        """
        # TODO: Add support for past_covariates
        selected_columns = self.metadata.known_covariates.copy()
        column_name_mapping = {ITEMID: MLF_ITEMID, TIMESTAMP: MLF_TIMESTAMP}
        if include_target:
            selected_columns += [self.target]
            column_name_mapping[self.target] = MLF_TARGET

        df = pd.DataFrame(data)[selected_columns].reset_index()
        if static_features is not None:
            df = pd.merge(df, static_features, how="left", on=ITEMID, suffixes=(None, "_static_feat"))

        for col in self._non_boolean_real_covariates:
            # Normalize non-boolean features using mean_abs scaling
            df[f"__scaled_{col}"] = df[col] / df[col].abs().groupby(df[ITEMID]).mean().reindex(df[ITEMID]).values

        # Convert float64 to float32 to reduce memory usage
        float64_cols = list(df.select_dtypes(include="float64"))
        df[float64_cols] = df[float64_cols].astype("float32")

        # We assume that df is sorted by 'unique_id' inside `TimeSeriesPredictor._check_and_prepare_data_frame`
        return df.rename(columns=column_name_mapping)

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        from mlforecast import MLForecast

        self._check_fit_params()
        fit_start_time = time.time()
        self._train_target_median = train_data[self.target].median()
        for col in self.metadata.known_covariates_real:
            if not train_data[col].isin([0, 1]).all():
                self._non_boolean_real_covariates.append(col)
        # TabularEstimator is passed to MLForecast later to include tuning_data
        model_params = self._get_model_params()

        mlforecast_init_args = self._get_mlforecast_init_args(train_data, model_params)
        self._mlf = MLForecast(models={}, freq=self.freq, **mlforecast_init_args)

        # We generate train/val splits from train_data and ignore val_data to avoid overfitting
        train_df, val_df = self._generate_train_val_dfs(
            train_data,
            max_num_items=model_params["max_num_items"],
            max_num_samples=model_params["max_num_samples"],
        )

        estimator = TabularEstimator(
            predictor_init_kwargs={
                "path": self.tabular_predictor_path,
                "verbosity": verbosity - 2,
                "label": MLF_TARGET,
                **self._get_extra_tabular_init_kwargs(),
            },
            predictor_fit_kwargs={
                "tuning_data": val_df.drop(columns=[MLF_ITEMID]),
                "time_limit": None if time_limit is None else time_limit - (time.time() - fit_start_time),
                "hyperparameters": model_params["tabular_hyperparameters"],
                **model_params["tabular_fit_kwargs"],
            },
        )
        self._mlf.models = {"mean": estimator}

        with warning_filter():
            self._mlf.fit_models(X=train_df.drop(columns=[MLF_TARGET, MLF_ITEMID]), y=train_df[MLF_TARGET])

        self._save_residuals_std(val_df)

    def _save_residuals_std(self, val_df: pd.DataFrame) -> None:
        """Compute standard deviation of residuals for each item using the validation set.

        Saves per-item residuals to `self.residuals_std_per_item` and average std to `self._avg_residuals_std`.
        """
        residuals = val_df[MLF_TARGET] - self._mlf.models_["mean"].predict(val_df)
        self._residuals_std_per_item = residuals.pow(2.0).groupby(val_df[MLF_ITEMID], sort=False).mean().pow(0.5)
        self._avg_residuals_std = np.sqrt(residuals.pow(2.0).mean())

    def _get_scale_per_item(self, item_ids: pd.Index) -> pd.Series:
        """Extract the '_scale' values from the scaler object, if available."""
        raise NotImplementedError

    def _remove_short_ts_and_generate_fallback_forecast(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
    ) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame], Optional[TimeSeriesDataFrame]]:
        """Remove series that are too short for chosen differencing from data and generate naive forecast for them.

        Returns
        -------
        data_long : TimeSeriesDataFrame
            Data containing only time series that are long enough for the model to predict.
        known_covariates_long : TimeSeriesDataFrame or None
            Future known covariates containing only time series that are long enough for the model to predict.
        forecast_for_short_series : TimeSeriesDataFrame or None
            Seasonal naive forecast for short series, if there are any in the dataset.
        """
        ts_lengths = data.num_timesteps_per_item()
        short_series = ts_lengths.index[ts_lengths <= self._sum_of_differences + 1]
        if len(short_series) > 0:
            logger.warning(
                f"Warning: {len(short_series)} time series ({len(short_series) / len(ts_lengths):.1%}) are shorter "
                f"than {self._sum_of_differences} and cannot be predicted by {self.name}. "
                "Fallback model SeasonalNaive is used for these time series."
            )
            data_short = data.query("item_id in @short_series")
            seasonal_naive = SeasonalNaiveModel(freq=self.freq, prediction_length=self.prediction_length)
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

    def _add_gaussian_quantiles(self, predictions: pd.DataFrame, repeated_item_ids: pd.Series):
        """
        Add quantile levels assuming that residuals follow normal distribution
        """
        from scipy.stats import norm

        scale_per_item = self._get_scale_per_item(repeated_item_ids.unique())
        num_items = int(len(predictions) / self.prediction_length)
        sqrt_h = np.sqrt(np.arange(1, self.prediction_length + 1))
        # Series where normal_scale_per_timestep.loc[item_id].loc[N] = sqrt(1 + N) for N in range(prediction_length)
        normal_scale_per_timestep = pd.Series(np.tile(sqrt_h, num_items), index=repeated_item_ids)

        residuals_std_per_timestep = self._residuals_std_per_item.reindex(repeated_item_ids)
        # Use avg_residuals_std in case unseen item received for prediction
        if residuals_std_per_timestep.isna().any():
            residuals_std_per_timestep = residuals_std_per_timestep.fillna(value=self._avg_residuals_std)
        std_per_timestep = residuals_std_per_timestep * scale_per_item * normal_scale_per_timestep
        for q in self.quantile_levels:
            predictions[str(q)] = predictions["mean"] + norm.ppf(q) * std_per_timestep.to_numpy()
        return predictions

    def _more_tags(self) -> dict:
        return {"allow_nan": True, "can_refit_full": True}


class DirectTabularModel(AbstractMLForecastModel):
    """Predict all future time series values simultaneously using TabularPredictor from AutoGluon-Tabular.

    A single TabularPredictor is used to forecast all future time series values using the following features:

    - lag features (observed time series values) based on ``freq`` of the data
    - time features (e.g., day of the week) based on the timestamp of the measurement
    - known covariates (if available)
    - static features of each item (if available)

    Features not known during the forecast horizon (e.g., future target values) are replaced by NaNs.

    If ``eval_metric.needs_quantile``, the TabularPredictor will be trained with ``"quantile"`` problem type.
    Otherwise, TabularPredictor will be trained with ``"regression"`` problem type, and dummy quantiles will be
    obtained by assuming that the residuals follow zero-mean normal distribution.

    Based on the `mlforecast <https://github.com/Nixtla/mlforecast>`_ library.


    Other Parameters
    ----------------
    lags : List[int], default = None
        Lags of the target that will be used as features for predictions. If None, will be determined automatically
        based on the frequency of the data.
    date_features : List[Union[str, Callable]], default = None
        Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input.
        If None, will be determined automatically based on the frequency of the data.
    differences : List[int], default = []
        Differences to take of the target before computing the features. These are restored at the forecasting step.
        If None, will be set to ``[seasonal_period]``, where seasonal_period is determined based on the data frequency.
        Defaults to no differencing.
    scaler : {"standard", "mean_abs", None}, default = "mean_abs"
        Scaling applied to each time series.
    tabular_hyperparameters : Dict[Dict[str, Any]], optional
        Hyperparameters dictionary passed to ``TabularPredictor.fit``. Contains the names of models that should be fit.
        Defaults to ``{"GBM": {}}``.
    tabular_fit_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to ``TabularPredictor.fit``. Defaults to an empty dict.
    max_num_items : int or None, default = 20_000
        If not None, the model will randomly select this many time series for training and validation.
    max_num_samples : int or None, default = 1_000_000
        If not None, training dataset passed to TabularPredictor will contain at most this many rows (starting from the
        end of each time series).
    """

    supports_known_covariates = True
    supports_static_features = True

    @property
    def is_quantile_model(self) -> bool:
        return self.eval_metric.needs_quantile

    def _get_model_params(self) -> dict:
        model_params = super()._get_model_params()
        model_params.setdefault("scaler", "mean_abs")
        model_params.setdefault("differences", [])
        return model_params

    def _mask_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a mask that mimics the situation at prediction time when target/covariates are unknown during the
        forecast horizon.
        """
        num_hidden = np.random.randint(0, self.prediction_length, size=len(df))
        lag_cols = [f"lag{lag}" for lag in self._target_lags]
        mask = num_hidden[:, None] < self._target_lags[None]  # shape [len(num_hidden), len(_target_lags)]
        # use df.loc[:, lag_cols] instead of df[lag_cols] to avoid SettingWithCopyWarning
        df.loc[:, lag_cols] = df[lag_cols].where(mask, other=np.nan)
        return df

    def _save_residuals_std(self, val_df: pd.DataFrame) -> None:
        if self.is_quantile_model:
            # Quantile model does not require residuals to produce prediction intervals
            self._residuals_std_per_item = pd.Series(1.0, index=val_df[MLF_ITEMID].unique())
            self._avg_residuals_std = 1.0
        else:
            super()._save_residuals_std(val_df=val_df)

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
            return forecast_for_short_series

        if known_covariates is not None:
            data_future = known_covariates.copy()
        else:
            future_index = get_forecast_horizon_index_ts_dataframe(data, self.prediction_length, freq=self.freq)
            data_future = pd.DataFrame(columns=[self.target], index=future_index, dtype="float32")
        # MLForecast raises exception of target contains NaN. We use inf as placeholder, replace them by NaN afterwards
        data_future[self.target] = float("inf")
        data_extended = pd.concat([data, data_future])
        mlforecast_df = self._to_mlforecast_df(data_extended, data.static_features)
        if self._max_ts_length is not None:
            # We appended `prediction_length` time steps to each series, so increase length
            mlforecast_df = self._shorten_all_series(mlforecast_df, self._max_ts_length + self.prediction_length)
        df = self._mlf.preprocess(mlforecast_df, dropna=False, static_features=[])
        df = df.groupby(MLF_ITEMID, sort=False).tail(self.prediction_length)
        df = df.replace(float("inf"), float("nan"))

        raw_predictions = self._mlf.models_["mean"].predict(df)
        predictions = self._postprocess_predictions(raw_predictions, repeated_item_ids=df[MLF_ITEMID])
        # Paste columns one by one to preserve dtypes
        predictions[MLF_ITEMID] = df[MLF_ITEMID].values
        predictions[MLF_TIMESTAMP] = df[MLF_TIMESTAMP].values

        if hasattr(self._mlf.ts, "target_transforms"):
            # Ensure that transforms are fitted only on past data
            mlforecast_df_past = self._to_mlforecast_df(data, None)
            if self._max_ts_length is not None:
                mlforecast_df_past = self._shorten_all_series(mlforecast_df_past, self._max_ts_length)
            self._mlf.preprocess(mlforecast_df_past, static_features=[])
            for tfm in self._mlf.ts.target_transforms[::-1]:
                predictions = tfm.inverse_transform(predictions)
        predictions = TimeSeriesDataFrame(predictions.rename(columns={MLF_ITEMID: ITEMID, MLF_TIMESTAMP: TIMESTAMP}))

        if forecast_for_short_series is not None:
            predictions = pd.concat([predictions, forecast_for_short_series])
            predictions = predictions.reindex(original_item_id_order, level=ITEMID)
        return predictions

    def _postprocess_predictions(self, predictions: np.ndarray, repeated_item_ids: pd.Series) -> pd.DataFrame:
        if self.is_quantile_model:
            predictions = pd.DataFrame(predictions, columns=[str(q) for q in self.quantile_levels])
            predictions.values.sort(axis=1)
            predictions["mean"] = predictions["0.5"]
        else:
            predictions = pd.DataFrame(predictions, columns=["mean"])
            predictions = self._add_gaussian_quantiles(predictions, repeated_item_ids=repeated_item_ids)

        column_order = ["mean"] + [col for col in predictions.columns if col != "mean"]
        return predictions[column_order]

    def _get_scale_per_item(self, item_ids: pd.Index) -> pd.Series:
        # Rescaling is applied in the inverse_transform step, no need to scale predictions
        return pd.Series(1.0, index=item_ids)

    def _get_extra_tabular_init_kwargs(self) -> dict:
        if self.is_quantile_model:
            return {
                "problem_type": ag.constants.QUANTILE,
                "quantile_levels": self.quantile_levels,
                "eval_metric": "pinball_loss",
            }
        else:
            return {
                "problem_type": ag.constants.REGRESSION,
                "eval_metric": self.eval_metric.equivalent_tabular_regression_metric or "mean_absolute_error",
            }


class RecursiveTabularModel(AbstractMLForecastModel):
    """Predict future time series values one by one using TabularPredictor from AutoGluon-Tabular.

    A single TabularPredictor is used to forecast the future time series values using the following features:

    - lag features (observed time series values) based on ``freq`` of the data
    - time features (e.g., day of the week) based on the timestamp of the measurement
    - known covariates (if available)
    - static features of each item (if available)

    TabularPredictor will always be trained with ``"regression"`` problem type, and dummy quantiles will be
    obtained by assuming that the residuals follow zero-mean normal distribution.

    Based on the `mlforecast <https://github.com/Nixtla/mlforecast>`_ library.


    Other Parameters
    ----------------
    lags : List[int], default = None
        Lags of the target that will be used as features for predictions. If None, will be determined automatically
        based on the frequency of the data.
    date_features : List[Union[str, Callable]], default = None
        Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input.
        If None, will be determined automatically based on the frequency of the data.
    differences : List[int], default = None
        Differences to take of the target before computing the features. These are restored at the forecasting step.
        If None, will be set to ``[seasonal_period]``, where seasonal_period is determined based on the data frequency.
    scaler : {"standard", "mean_abs", None}, default = "standard"
        Scaling applied to each time series.
    tabular_hyperparameters : Dict[Dict[str, Any]], optional
        Hyperparameters dictionary passed to ``TabularPredictor.fit``. Contains the names of models that should be fit.
        Defaults to ``{"GBM": {}}``.
    tabular_fit_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to ``TabularPredictor.fit``. Defaults to an empty dict.
    max_num_items : int or None, default = 20_000
        If not None, the model will randomly select this many time series for training and validation.
    max_num_samples : int or None, default = 1_000_000
        If not None, training dataset passed to TabularPredictor will contain at most this many rows (starting from the
        end of each time series).
    """

    supports_known_covariates = True
    supports_static_features = True

    def _get_model_params(self) -> dict:
        model_params = super()._get_model_params()
        model_params.setdefault("scaler", "standard")
        model_params.setdefault("differences", [get_seasonality(self.freq)])
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
            return forecast_for_short_series

        new_df = self._to_mlforecast_df(data, data.static_features)
        if self._max_ts_length is not None:
            new_df = self._shorten_all_series(new_df, self._max_ts_length)
        if known_covariates is None:
            future_index = get_forecast_horizon_index_ts_dataframe(data, self.prediction_length, freq=self.freq)
            known_covariates = pd.DataFrame(columns=[self.target], index=future_index, dtype="float32")
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
        predictions = raw_predictions.rename(columns={MLF_ITEMID: ITEMID, MLF_TIMESTAMP: TIMESTAMP})
        predictions = TimeSeriesDataFrame(
            self._add_gaussian_quantiles(predictions, repeated_item_ids=predictions[ITEMID])
        )

        if forecast_for_short_series is not None:
            predictions = pd.concat([predictions, forecast_for_short_series])
        return predictions.reindex(original_item_id_order, level=ITEMID)

    def _get_extra_tabular_init_kwargs(self) -> dict:
        return {
            "problem_type": ag.constants.REGRESSION,
            "eval_metric": self.eval_metric.equivalent_tabular_regression_metric or "mean_absolute_error",
        }

    def _get_scale_per_item(self, item_ids: pd.Index) -> pd.Series:
        if self._scaler is not None:
            return self._scaler.stats_["_scale"].copy().reindex(item_ids)
        else:
            return pd.Series(1.0, index=item_ids)
