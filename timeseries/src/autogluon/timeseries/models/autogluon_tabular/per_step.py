import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import QUANTILE
from autogluon.tabular.models import AbstractModel as AbstractTabularModel
from autogluon.tabular.registry import ag_model_registry
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.datetime import get_lags_for_frequency, get_time_features_for_frequency
from autogluon.timeseries.utils.warning_filters import set_loggers_level

from .utils import MLF_ITEMID, MLF_TARGET, MLF_TIMESTAMP

logger = logging.getLogger(__name__)

DUMMY_FREQ = "D"


class PerStepTabularModel(AbstractTimeSeriesModel):
    """Fit a separate tabular regression model for each time step in the forecast horizon.

    Each model has access to the following features:

    - lag features (observed time series values) based on ``freq`` of the data
    - time features (e.g., day of the week) based on the timestamp of the measurement
    - known covariates (if available)
    - static features of each item (if available)

    This model is typically much slower to fit compared to other tabular forecasting models.

    This model uses `mlforecast <https://github.com/Nixtla/mlforecast>`_ under the hood for efficient preprocessing,
    but the implementation of the per-step forecasting strategy is different from the `max_horizon` in `mlforecast`.


    Other Parameters
    ----------------
    trailing_lags : List[int], default = None
        Trailing window lags of the target that will be used as features for predictions.
        Trailing lags are shifted per forecast step: model for step `h` uses `[lag+h for lag in trailing_lags]`.
        If None, defaults to [1, 2, ..., 12].
    seasonal_lags: List[int], default = None
        Seasonal lags of the target used as features. Unlike trailing lags, seasonal lags are not shifted
        but filtered by availability: model for step `h` uses `[lag for lag in seasonal_lags if lag > h]`.
        If None, determined automatically based on data frequency.
    date_features : List[Union[str, Callable]], default = None
        Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input.
        If None, will be determined automatically based on the frequency of the data.
    target_scaler : {"standard", "mean_abs", "min_max", "robust", None}, default = "mean_abs"
        Scaling applied to each time series.
    model_name : str, default = "CAT"
        Name of the tabular regression model. See `autogluon.tabular.registry.ag_model_registry` or
        `the documentation <https://auto.gluon.ai/stable/api/autogluon.tabular.models.html>`_ for the list of available
        tabular models.
    model_hyperparameters : Dict[str, Any], optional
        Hyperparameters passed to the tabular regression model.
    max_num_items : int or None, default = 20_000
        If not None, the model will randomly select this many time series for training and validation.
    max_num_samples : int or None, default = 1_000_000
        If not None, training dataset passed to TabularPredictor will contain at most this many rows (starting from the
        end of each time series).
    n_jobs : int or None, default = None
        Number of parallel jobs for fitting models across forecast horizons.
        If None, automatically determined based on available memory to prevent OOM errors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We save the relative paths to per-step models. Each worker process independently saves/loads the model.
        # This is much more efficient than passing around model objects that can get really large
        self._relative_paths_to_models: list[str]
        self._trailing_lags: list[int]
        self._seasonal_lags: list[int]
        self._date_features: list[Callable]
        self._model_cls: Type[AbstractTabularModel]
        self._n_jobs: int
        self._non_boolean_real_covariates: List[str] = []
        self._max_ts_length: Optional[int] = None

    @property
    def allowed_hyperparameters(self) -> List[str]:
        # TODO: Differencing is currently not supported because it greatly complicates the preprocessing logic
        return super().allowed_hyperparameters + [
            "trailing_lags",
            "seasonal_lags",
            "date_features",
            # "differences",
            "validation_fraction",
            "model_name",
            "model_hyperparameters",
            "max_num_items",
            "max_num_samples",
            "n_jobs",
        ]

    @property
    def _ag_to_nixtla(self) -> dict:
        return {self.target: MLF_TARGET, ITEMID: MLF_ITEMID, TIMESTAMP: MLF_TIMESTAMP}

    def _get_default_hyperparameters(self):
        return {
            "model_name": "CAT",
            "model_hyperparameters": {},
            "target_scaler": "mean_abs",
            "validation_fraction": 0.1,
            "max_num_samples": 1_000_000,
            "max_num_items": 20_000,
        }

    @staticmethod
    def _fit_single_model(
        train_df: pd.DataFrame,
        path_root: str,
        step: int,
        model_cls: Type[AbstractTabularModel],
        model_hyperparameters: dict,
        validation_fraction: Optional[float],
        quantile_levels: list[float],
        lags: list[int],
        date_features: list[Callable],
        time_limit: Optional[float],
        num_cpus: int,
        verbosity: int,
    ) -> str:
        from mlforecast import MLForecast

        start_time = time.monotonic()

        mlf = MLForecast(models=[], freq=DUMMY_FREQ, lags=lags, date_features=date_features)

        features_df = mlf.preprocess(train_df, static_features=[], dropna=False)
        del train_df
        del mlf
        # Sort chronologically for efficient train/test split
        features_df = features_df.sort_values(by=MLF_TIMESTAMP)
        X = features_df.drop(columns=[MLF_ITEMID, MLF_TIMESTAMP, MLF_TARGET])
        y = features_df[MLF_TARGET]
        del features_df

        y_is_valid = np.isfinite(y)
        X, y = X[y_is_valid], y[y_is_valid]
        X = X.replace(float("inf"), float("nan"))
        if validation_fraction is None or validation_fraction == 0.0:
            X_val = None
            y_val = None
        else:
            assert 0 < validation_fraction < 1, "validation_fraction must be between 0.0 and 1.0"
            num_val = math.ceil(len(X) * validation_fraction)
            X_val, y_val = X.iloc[-num_val:], y.iloc[-num_val:]
            X, y = X.iloc[:-num_val], y.iloc[:-num_val]
        if len(y) == 0:
            raise ValueError("Not enough valid target values to fit model")

        elapsed = time.monotonic() - start_time
        time_left = time_limit - elapsed if time_limit is not None else None
        try:
            with set_loggers_level(regex=r"^autogluon.tabular.*", level=logging.ERROR):
                model = model_cls(
                    path=os.path.join(path_root, f"step_{step}"),
                    name=model_cls.__name__,  # explicitly provide name to avoid warnings
                    problem_type=QUANTILE,
                    eval_metric="pinball_loss",
                    hyperparameters={**model_hyperparameters, "ag.quantile_levels": quantile_levels},
                )
                model.fit(
                    X=X,
                    y=y,
                    X_val=X_val,
                    y_val=y_val,
                    time_limit=time_left,
                    num_cpus=num_cpus,
                    num_gpus=0,  # num_cpus is only used if num_gpus is set as well
                    verbosity=verbosity,
                )
        except Exception as e:
            raise RuntimeError(f"Failed when fitting model for {step=}") from e
        model.save()
        relative_path = os.path.relpath(path=model.path, start=path_root)
        return relative_path

    @staticmethod
    def _get_n_jobs(
        train_df: pd.DataFrame,
        num_extra_dynamic_features: int,
        model_cls: Type[AbstractTabularModel],
        model_hyperparameters: dict,
        overhead_factor: float = 2.0,
    ) -> int:
        """Estimate the maximum number of jobs that can be run in parallel without encountering OOM errors."""
        mem_usage_per_column = get_approximate_df_mem_usage(train_df)
        num_columns = len(train_df.columns)
        mem_usage_per_job = mem_usage_per_column.sum()
        try:
            mem_usage_per_job += model_cls.estimate_memory_usage_static(
                X=train_df, hyperparameters=model_hyperparameters, problem_type="regression"
            )
        except NotImplementedError:
            mem_usage_per_job *= 2
        # Extra scaling factor because the preprocessed DF will have more columns for lags + date features
        mem_usage_per_job *= overhead_factor + (num_extra_dynamic_features + num_columns) / num_columns
        max_jobs_by_memory = int(ResourceManager.get_available_virtual_mem() / mem_usage_per_job)
        return max(1, max_jobs_by_memory)

    def preprocess(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        is_train: bool = False,
        **kwargs,
    ):
        # TODO: Make this toggleable with a hyperparameter
        # We add a scaled version of non-boolean known real covariates, same as in MLForecast models
        if is_train:
            for col in self.covariate_metadata.known_covariates_real:
                if not set(data[col].unique()) == set([0, 1]):
                    self._non_boolean_real_covariates.append(col)

        if len(self._non_boolean_real_covariates) > 0:
            item_ids = data.index.get_level_values(level=ITEMID)
            scale_per_column: dict[str, pd.Series] = {}
            columns_grouped = data[self._non_boolean_real_covariates].abs().groupby(item_ids)
            for col in self._non_boolean_real_covariates:
                scale_per_column[col] = columns_grouped[col].mean()
            data = data.assign(**{f"__scaled_{col}": data[col] / scale for col, scale in scale_per_column.items()})
            if known_covariates is not None:
                known_covariates = known_covariates.assign(
                    **{f"__scaled_{col}": known_covariates[col] / scale for col, scale in scale_per_column.items()}
                )
        data = data.astype({self.target: "float32"})
        return data, known_covariates

    def _get_train_df(
        self, train_data: TimeSeriesDataFrame, max_num_items: Optional[int], max_num_samples: Optional[int]
    ) -> pd.DataFrame:
        if max_num_items is not None and train_data.num_items > max_num_items:
            items_to_keep = train_data.item_ids.to_series().sample(n=int(max_num_items))  # noqa: F841
            train_data = train_data.query("item_id in @items_to_keep")

        if max_num_samples is not None and len(train_data) > max_num_samples:
            max_samples_per_ts = max(200, math.ceil(max_num_samples / train_data.num_items))
            self._max_ts_length = max_samples_per_ts + self.prediction_length
            train_data = train_data.slice_by_timestep(-self._max_ts_length, None)

        if len(self.covariate_metadata.past_covariates) > 0:
            train_data = train_data.drop(columns=self.covariate_metadata.past_covariates)

        train_df = train_data.to_data_frame().reset_index()
        if train_data.static_features is not None:
            train_df = pd.merge(
                left=train_df, right=train_data.static_features, left_on=ITEMID, right_index=True, how="left"
            )
        train_df = train_df.rename(columns=self._ag_to_nixtla)
        train_df = train_df.assign(**{MLF_TARGET: train_df[MLF_TARGET].fillna(float("inf"))})
        return train_df

    @staticmethod
    def _get_lags_for_step(
        trailing_lags: List[int],
        seasonal_lags: List[int],
        step: int,
    ) -> List[int]:
        """Get the list of lags that can be used by the model for the given step."""
        shifted_trailing_lags = [lag + step for lag in trailing_lags]
        # Only keep lags that are available for model predicting `step` values ahead at prediction time
        valid_lags = [lag for lag in shifted_trailing_lags + seasonal_lags if lag > step]
        return sorted(set(valid_lags))

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        self._check_fit_params()
        self._log_unused_hyperparameters()
        model_params = self.get_hyperparameters()

        train_df = self._get_train_df(
            train_data,
            max_num_items=model_params["max_num_items"],
            max_num_samples=model_params["max_num_samples"],
        )

        # Initialize MLForecast arguments
        assert self.freq is not None
        trailing_lags = model_params.get("trailing_lags")
        if trailing_lags is None:
            trailing_lags = list(range(1, 13))
        # Ensure that lags have type list[int] and not, e.g., np.ndarray
        self._trailing_lags = [int(lag) for lag in trailing_lags]
        assert all(lag >= 1 for lag in self._trailing_lags), "trailing_lags must be >= 1"

        seasonal_lags = model_params.get("seasonal_lags")
        if seasonal_lags is None:
            median_ts_length = int(train_df[MLF_ITEMID].value_counts(sort=False).median())
            seasonal_lags = get_lags_for_frequency(self.freq, num_default_lags=0, lag_ub=median_ts_length)
        self._seasonal_lags = [int(lag) for lag in seasonal_lags]
        assert all(lag >= 1 for lag in self._seasonal_lags), "seasonal_lags must be >= 1"

        date_features = model_params.get("date_features")
        if date_features is None:
            date_features = get_time_features_for_frequency(self.freq)
        self._date_features = date_features

        self._model_cls = ag_model_registry.key_to_cls(model_params["model_name"])
        supported_problem_types = self._model_cls.supported_problem_types()
        if supported_problem_types is not None and QUANTILE not in supported_problem_types:
            raise ValueError(
                f"Chosen model_name='{model_params['model_name']}' cannot be used by {self.name} because it does not "
                f"support problem_type='quantile' ({supported_problem_types=})"
            )
        model_hyperparameters = model_params["model_hyperparameters"]
        # User-provided n_jobs takes priority over the automatic estimate
        if model_params.get("n_jobs") is not None:
            self._n_jobs = model_params["n_jobs"]
        else:
            self._n_jobs = self._get_n_jobs(
                train_df,
                num_extra_dynamic_features=len(set(self._seasonal_lags + self._trailing_lags))
                + len(self._date_features),
                model_cls=self._model_cls,
                model_hyperparameters=model_hyperparameters,
            )
        n_jobs = min(self._n_jobs, self.prediction_length, cpu_count(only_physical_cores=True))

        num_cpus_per_model = max(cpu_count(only_physical_cores=True) // n_jobs, 1)
        if time_limit is not None:
            time_limit_per_model = time_limit / math.ceil(self.prediction_length / n_jobs)
        else:
            time_limit_per_model = None
        model_fit_kwargs = dict(
            train_df=train_df,
            path_root=self.path,
            model_cls=self._model_cls,
            quantile_levels=self.quantile_levels,
            validation_fraction=model_params["validation_fraction"],
            date_features=self._date_features,
            time_limit=time_limit_per_model,
            num_cpus=num_cpus_per_model,
            model_hyperparameters=model_hyperparameters.copy(),
            verbosity=verbosity - 1,
        )
        logger.debug(f"Fitting models in parallel with {n_jobs=}, {num_cpus_per_model=}, {time_limit_per_model=}")
        self._relative_paths_to_models = Parallel(n_jobs=n_jobs)(  # type: ignore
            delayed(self._fit_single_model)(
                step=step,
                lags=self._get_lags_for_step(
                    seasonal_lags=self._seasonal_lags, trailing_lags=self._trailing_lags, step=step
                ),
                **model_fit_kwargs,
            )
            for step in range(self.prediction_length)
        )

    @staticmethod
    def _predict_with_single_model(
        full_df: pd.DataFrame,
        path_to_model: str,
        model_cls: Type[AbstractTabularModel],
        step: int,
        prediction_length: int,
        lags: list[int],
        date_features: list[Callable],
    ) -> np.ndarray:
        """Make predictions with the model for the given step.

        Returns
        -------
        predictions :
            Predictions of the model for the given step. Shape: (num_items, len(quantile_levels)).
        """
        from mlforecast import MLForecast

        mlf = MLForecast(models=[], freq=DUMMY_FREQ, lags=lags, date_features=date_features)

        features_df = mlf.preprocess(full_df, static_features=[], dropna=False)
        del mlf

        end_idx_per_item = np.cumsum(features_df[MLF_ITEMID].value_counts(sort=False).to_numpy(dtype="int32"))
        features_for_step = features_df.iloc[end_idx_per_item - (prediction_length - step)]
        try:
            model: AbstractTabularModel = model_cls.load(path_to_model)  # type: ignore
        except:
            logger.error(f"Could not load model for {step=} from {path_to_model}")
            raise
        predictions = model.predict(features_for_step)
        return predictions

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if known_covariates is not None:
            X_df = known_covariates
        else:
            X_df = TimeSeriesDataFrame(
                pd.DataFrame(float("inf"), index=self.get_forecast_horizon_index(data), columns=[self.target])
            )
        full_df = pd.concat([data, X_df])
        if self._max_ts_length is not None:
            full_df = full_df.slice_by_timestep(-(self._max_ts_length + self.prediction_length), None)
        full_df = full_df.to_data_frame().reset_index()
        if data.static_features is not None:
            full_df = pd.merge(full_df, data.static_features, left_on=ITEMID, right_index=True, how="left")

        full_df = (
            full_df.rename(columns=self._ag_to_nixtla)
            .sort_values(by=[MLF_ITEMID, MLF_TIMESTAMP])
            .reset_index(drop=True)
        )
        full_df = full_df.assign(**{MLF_TARGET: full_df[MLF_TARGET].fillna(float("inf"))})

        model_predict_kwargs = dict(
            full_df=full_df,
            prediction_length=self.prediction_length,
            model_cls=self._model_cls,
            date_features=self._date_features,
        )
        n_jobs = min(self._n_jobs, self.prediction_length, cpu_count(only_physical_cores=True))
        predictions_per_step = Parallel(n_jobs=n_jobs)(
            delayed(self._predict_with_single_model)(
                step=step,
                lags=self._get_lags_for_step(
                    seasonal_lags=self._seasonal_lags, trailing_lags=self._trailing_lags, step=step
                ),
                path_to_model=os.path.join(self.path, suffix),
                **model_predict_kwargs,
            )
            for step, suffix in enumerate(self._relative_paths_to_models)
        )
        predictions = pd.DataFrame(
            np.stack(predictions_per_step, axis=1).reshape([-1, len(self.quantile_levels)]),
            columns=[str(q) for q in self.quantile_levels],
            index=self.get_forecast_horizon_index(data),
        )
        predictions["mean"] = predictions["0.5"]
        return TimeSeriesDataFrame(predictions)

    def _more_tags(self) -> Dict[str, Any]:
        return {"allow_nan": True, "can_refit_full": True}
