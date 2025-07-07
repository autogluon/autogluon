import logging
import math
import time
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models import AbstractModel as AbstractTabularModel
from autogluon.tabular.registry import ag_model_registry
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.datetime import get_lags_for_frequency, get_time_features_for_frequency
from autogluon.timeseries.utils.forecast import make_future_data_frame

from .utils import MLF_ITEMID, MLF_TARGET, MLF_TIMESTAMP

logger = logging.getLogger(__name__)


class PerStepTabularModel(AbstractTimeSeriesModel):
    """Fit a separate tabular regression model for each time step in the forecast horizon.

    Each model has access to the following features:

    - lag features (observed time series values) based on ``freq`` of the data
    - time features (e.g., day of the week) based on the timestamp of the measurement
    - known covariates (if available)
    - static features of each item (if available)

    Based on the `mlforecast <https://github.com/Nixtla/mlforecast>`_ library.


    Other Parameters
    ----------------
    lags : List[int], default = None
        Lags of the target that will be used as features for predictions. If None, will be determined automatically
        based on the frequency of the data.
        Lags are shifted per forecast step: model for step `h` uses `[lag+h for lag in lags]`.
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
        self.model_per_step: list[AbstractTabularModel]
        self.lags: list[int]
        self.date_features: list[Callable]
        self._non_boolean_real_covariates: List[str] = []

    @property
    def allowed_hyperparameters(self) -> List[str]:
        # TODO: Differencing is currently not supported because it greatly complicates the preprocessing logic
        return super().allowed_hyperparameters + [
            "lags",
            "date_features",
            # "differences",
            "validation_fraction",
            "model_name",
            "model_hyperparameters",
            "max_num_items",
            "max_num_samples",
            "lag_transforms",
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
        step: int,
        model_name: str,
        model_hyperparameters: dict,
        validation_fraction: Optional[float],
        quantile_levels: list[float],
        lags: list[int],
        date_features: list[Callable],
        time_limit: Optional[float],
        num_cpus: int,
    ) -> AbstractTabularModel:
        from mlforecast import MLForecast

        start_time = time.monotonic()

        # Ensure that lags have type list[int] (and not e.g. np.ndarray)
        lags = sorted([int(lag) + step for lag in lags])
        mlf = MLForecast(models=[], freq="D", lags=lags, date_features=date_features)

        features_df = mlf.preprocess(train_df, static_features=[], dropna=False)
        del mlf
        # Sort chronologically for efficient train/test split
        features_df.sort_values(by=MLF_TIMESTAMP)
        X = features_df.drop(columns=[MLF_TARGET])
        y = features_df[MLF_TARGET]

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

        model_cls = ag_model_registry.key_to_cls(model_name)
        model = model_cls(
            path="",
            problem_type="quantile",
            hyperparameters={**model_hyperparameters, "ag.quantile_levels": quantile_levels},
        )
        elapsed = time.monotonic() - start_time
        # print(f"Preprocessing time for {step=}: {elapsed:.1f}s")
        time_left = time_limit - elapsed if time_limit is not None else None
        model.fit(X=X, y=y, X_val=X_val, y_val=y_val, time_limit=time_left, num_cpus=num_cpus, num_gpus=0)
        return model

    @staticmethod
    def _get_n_jobs(
        train_df: pd.DataFrame,
        prediction_length: int,
        model_name: str,
        model_hyperparameters: dict,
        overhead_factor: float = 2.0,
    ) -> int:
        """Estimate the maximum number of jobs that can be run in parallel without encountering OOM errors."""
        mem_usage_per_job = overhead_factor * train_df.memory_usage().sum()
        model_cls = ag_model_registry.key_to_cls(model_name)
        try:
            mem_usage_per_job += model_cls.estimate_memory_usage_static(
                X=train_df, hyperparameters=model_hyperparameters, problem_type="regression"
            )
        except NotImplementedError:
            mem_usage_per_job += 0.5 * mem_usage_per_job
        max_jobs_by_memory = int(ResourceManager.get_available_virtual_mem() / mem_usage_per_job)
        return max(1, min(max_jobs_by_memory, cpu_count(only_physical_cores=True), prediction_length))

    def preprocess(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        is_train: bool = False,
        **kwargs,
    ):
        # We add a scaled version of non-boolean known real covariates
        if is_train:
            for col in self.covariate_metadata.known_covariates_real:
                if not set(data[col].unique()) == set([0, 1]):
                    self._non_boolean_real_covariates.append(col)

        if len(self._non_boolean_real_covariates) > 0:
            item_ids = data.index.get_level_values(level=ITEMID)
            scale_per_column: dict[str, pd.Series] = {}
            for col in self._non_boolean_real_covariates:
                scale_per_column[col] = data[col].abs().groupby(item_ids).mean()
            data = data.assign(**{f"__scaled_{col}": data[col] / scale for col, scale in scale_per_column.items()})
            if known_covariates is not None:
                known_covariates = known_covariates.assign(
                    **{f"__scaled_{col}": known_covariates[col] / scale for col, scale in scale_per_column.items()}
                )
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
        lags = model_params.get("lags")
        if lags is None:
            lags = get_lags_for_frequency(self.freq, lag_ub=int(train_data.num_timesteps_per_item().max()))
        self.lags = lags

        date_features = model_params.get("date_features")
        if date_features is None:
            date_features = get_time_features_for_frequency(self.freq)
        self.date_features = date_features

        model_name = model_params["model_name"]
        model_hyperparameters = model_params["model_hyperparameters"]
        # User-provided n_jobs takes priority over the automatic estimate
        if model_params.get("n_jobs") is not None:
            n_jobs = model_params["n_jobs"]
        else:
            n_jobs = self._get_n_jobs(
                train_df,
                prediction_length=self.prediction_length,
                model_name=model_name,
                model_hyperparameters=model_hyperparameters,
            )

        num_cpus_per_model = max(cpu_count(only_physical_cores=True) // n_jobs, 1)
        if time_limit is not None:
            time_limit_per_model = time_limit / math.ceil(self.prediction_length / n_jobs)
        else:
            time_limit_per_model = None
        model_fit_kwargs = dict(
            train_df=train_df,
            model_name=model_name,
            quantile_levels=self.quantile_levels,
            validation_fraction=model_params["validation_fraction"],
            lags=lags,
            date_features=date_features,
            time_limit=time_limit_per_model,
            num_cpus=num_cpus_per_model,
            model_hyperparameters=model_hyperparameters.copy(),
        )
        logger.debug(f"Fitting models in parallel with {n_jobs=}, {num_cpus_per_model=}, {time_limit_per_model=:.1f}")
        self.model_per_step = Parallel(n_jobs=n_jobs)(  # type: ignore
            delayed(self._fit_single_model)(step=step, **model_fit_kwargs) for step in range(self.prediction_length)
        )

    @staticmethod
    def _predict_with_single_model(
        full_df: pd.DataFrame,
        step: int,
        prediction_length: int,
        model: AbstractTabularModel,
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

        lags = sorted([int(lag) + step for lag in lags])
        mlf = MLForecast(models=[], freq="D", lags=lags, date_features=date_features)

        features_df = mlf.preprocess(full_df, static_features=[], dropna=False)
        del mlf
        features_for_step = features_df.groupby(MLF_ITEMID, sort=False, as_index=False).nth(
            -(prediction_length - step)
        )
        return model.predict(features_for_step)

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        df = data.to_data_frame().reset_index().rename(columns=self._ag_to_nixtla)
        if known_covariates is not None:
            X_df = known_covariates.to_data_frame().reset_index()
        else:
            X_df = make_future_data_frame(data, prediction_length=self.prediction_length, freq=self.freq)
        if data.static_features is not None:
            X_df = pd.merge(X_df, data.static_features, left_on=ITEMID, right_index=True, how="left")
        X_df = X_df.rename(columns=self._ag_to_nixtla)

        full_df = pd.concat([df, X_df]).sort_values(by=[MLF_ITEMID, MLF_TIMESTAMP])
        full_df = full_df.assign(**{MLF_TARGET: full_df[MLF_TARGET].fillna(float("inf"))})

        predictions_per_step = [
            self._predict_with_single_model(
                full_df,
                step=step,
                prediction_length=self.prediction_length,
                model=model,
                lags=self.lags,
                date_features=self.date_features,
            )
            for step, model in enumerate(self.model_per_step)
        ]
        predictions = pd.DataFrame(
            np.stack(predictions_per_step, axis=1).reshape([-1, len(self.quantile_levels)]),
            columns=[str(q) for q in self.quantile_levels],
            index=self.get_forecast_horizon_index(data),
        )
        predictions["mean"] = predictions["0.5"]
        return TimeSeriesDataFrame(predictions)
