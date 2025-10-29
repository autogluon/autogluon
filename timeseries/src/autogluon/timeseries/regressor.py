import logging
import time
from typing import Any, Optional, Protocol, Union, overload, runtime_checkable

import numpy as np
import pandas as pd

from autogluon.core.models import AbstractModel
from autogluon.tabular.registry import ag_model_registry as tabular_ag_model_registry
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.utils.features import CovariateMetadata

logger = logging.getLogger(__name__)


@runtime_checkable
class CovariateRegressor(Protocol):
    def is_fit(self) -> bool: ...

    def fit(self, data: TimeSeriesDataFrame, time_limit: Optional[float] = None, **kwargs) -> "CovariateRegressor": ...

    def transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame: ...

    def fit_transform(
        self, data: TimeSeriesDataFrame, time_limit: Optional[float] = None, **kwargs
    ) -> TimeSeriesDataFrame: ...

    def inverse_transform(
        self,
        predictions: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame,
        static_features: Optional[pd.DataFrame],
    ) -> TimeSeriesDataFrame: ...


class GlobalCovariateRegressor(CovariateRegressor):
    """Predicts target values from the covariates for the same observation.

    The model construct the feature matrix using known_covariates and static_features.

    Parameters
    ----------
    model_name
        Name of the tabular regression model. See ``autogluon.tabular.registry.ag_model_registry`` or
        `the documentation <https://auto.gluon.ai/stable/api/autogluon.tabular.models.html>`_ for the list of available
        tabular models.
    model_hyperparameters
        Hyperparameters passed to the tabular regression model.
    eval_metric
        Metric provided as ``eval_metric`` to the tabular regression model. Must be compatible with `problem_type="regression"`.
    refit_during_predict
        If True, the model will be re-trained every time ``fit_transform`` is called. If False, the model will only be
        trained the first time that ``fit_transform`` is called, and future calls to ``fit_transform`` will only perform a
        ``transform``.
    max_num_samples
        If not None, training dataset passed to regression model will contain at most this many rows.
    covariate_metadata
        Metadata object describing the covariates available in the dataset.
    target
        Name of the target column.
    validation_fraction
        Fraction of observations that are reserved as the validation set during training (starting from the end of each
        time series).
    fit_time_fraction
        The fraction of the time_limit that will be reserved for model training. The remainder (1 - fit_time_fraction)
        will be reserved for prediction.

        If the estimated prediction time exceeds ``(1 - fit_time_fraction) * time_limit``, the regressor will be disabled.
    include_static_features
        If True, static features will be included as features for the regressor.
    include_item_id
        If True, item_id will be included as a categorical feature for the regressor.
    """

    def __init__(
        self,
        model_name: str = "CAT",
        model_hyperparameters: Optional[dict[str, Any]] = None,
        eval_metric: str = "mean_absolute_error",
        refit_during_predict: bool = False,
        max_num_samples: Optional[int] = 500_000,
        covariate_metadata: Optional[CovariateMetadata] = None,
        target: str = "target",
        validation_fraction: Optional[float] = 0.1,
        fit_time_fraction: float = 0.5,
        include_static_features: bool = True,
        include_item_id: bool = False,
    ):
        tabular_model_types = tabular_ag_model_registry.key_to_cls_map()
        if model_name not in tabular_model_types:
            raise ValueError(
                f"Tabular model {model_name} not supported. Available models: {list(tabular_model_types)}"
            )
        self.target = target
        self.model_type = tabular_model_types[model_name]
        self.model_name = model_name
        self.model_hyperparameters = model_hyperparameters or {}
        self.refit_during_predict = refit_during_predict
        self.tabular_eval_metric = eval_metric
        self.max_num_samples = max_num_samples
        self.validation_fraction = validation_fraction
        self.fit_time_fraction = fit_time_fraction
        self.include_static_features = include_static_features
        self.include_item_id = include_item_id

        self.model: Optional[AbstractModel] = None
        self.disabled = False
        self.covariate_metadata = covariate_metadata or CovariateMetadata()

    def is_fit(self) -> bool:
        return self.model is not None

    def fit(self, data: TimeSeriesDataFrame, time_limit: Optional[float] = None, **kwargs) -> "CovariateRegressor":
        """Fit the tabular regressor on the target column using covariates as features."""
        start_time = time.monotonic()
        tabular_df = self._get_tabular_df(data, static_features=data.static_features, include_target=True)
        tabular_df = tabular_df.query(f"{self.target}.notnull()")

        median_ts_length = data.num_timesteps_per_item().median()
        features_to_drop = [self.target]
        if not self.include_item_id:
            features_to_drop += [TimeSeriesDataFrame.ITEMID]
        if self.validation_fraction is not None:
            grouped_df = tabular_df.groupby(TimeSeriesDataFrame.ITEMID, observed=False, sort=False)
            val_size = max(int(self.validation_fraction * median_ts_length), 1)
            train_df = self._subsample_df(grouped_df.head(-val_size))
            val_df = self._subsample_df(grouped_df.tail(val_size))
            X = train_df.drop(columns=features_to_drop)
            y = train_df[self.target]
            X_val = val_df.drop(columns=features_to_drop)
            y_val = val_df[self.target]
        else:
            tabular_df = self._subsample_df(tabular_df)
            X = tabular_df.drop(columns=features_to_drop)
            y = tabular_df[self.target]
            X_val = None
            y_val = None

        self.model = self.model_type(
            problem_type="regression",
            hyperparameters={
                **self.model_hyperparameters,
                "ag_args_fit": {"predict_1_batch_size": 10000},  # needed to compute predict_1_time
            },
            eval_metric=self.tabular_eval_metric,
            # Has no effect since the model won't be saved to disk.
            # We provide path to avoid https://github.com/autogluon/autogluon/issues/4832
            path="",
            name=self.model_type.__name__,
        )
        if time_limit is not None:
            time_limit_fit = self.fit_time_fraction * (time_limit - (time.monotonic() - start_time))
        else:
            time_limit_fit = None
        # Don't fit if all features are constant to avoid autogluon.core.utils.exceptions.NoValidFeatures
        if (X.nunique() <= 1).all():
            logger.warning("\tDisabling the covariate_regressor since all features are constant.")
            self.disabled = True
        else:
            self.model.fit(X=X, y=y, X_val=X_val, y_val=y_val, time_limit=time_limit_fit, **kwargs)

            if time_limit is not None:
                time_left = time_limit - (time.monotonic() - start_time)
                assert self.model.predict_1_time is not None
                estimated_predict_time = self.model.predict_1_time * len(data)
                if estimated_predict_time > time_left:
                    logger.warning(
                        f"\tDisabling the covariate_regressor since {estimated_predict_time=:.1f} exceeds {time_left=:.1f}."
                    )
                    self.disabled = True
        return self

    def transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Subtract the tabular regressor predictions from the target column."""
        if not self.disabled:
            y_pred = self._predict(data, static_features=data.static_features)
            data = data.assign(**{self.target: data[self.target] - y_pred})
        return data

    def fit_transform(
        self, data: TimeSeriesDataFrame, time_limit: Optional[float] = None, **kwargs
    ) -> TimeSeriesDataFrame:
        if not self.is_fit() or self.refit_during_predict:
            self.fit(data=data, time_limit=time_limit, **kwargs)
        return self.transform(data=data)

    def inverse_transform(
        self,
        predictions: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame,
        static_features: Optional[pd.DataFrame],
    ) -> TimeSeriesDataFrame:
        """Add the tabular regressor predictions to the target column."""
        if not self.disabled:
            y_pred = self._predict(known_covariates, static_features=static_features)
            predictions = predictions.assign(**{col: predictions[col] + y_pred for col in predictions.columns})
        return predictions

    def _predict(self, data: TimeSeriesDataFrame, static_features: Optional[pd.DataFrame]) -> np.ndarray:
        """Construct the tabular features matrix and make predictions"""
        assert self.model is not None, "CovariateRegressor must be fit before calling predict."
        tabular_df = self._get_tabular_df(data, static_features=static_features)
        if not self.include_item_id:
            tabular_df = tabular_df.drop(columns=[TimeSeriesDataFrame.ITEMID])
        return self.model.predict(X=tabular_df)

    def _get_tabular_df(
        self,
        data: TimeSeriesDataFrame,
        static_features: Optional[pd.DataFrame] = None,
        include_target: bool = False,
    ) -> pd.DataFrame:
        """Construct a tabular dataframe from known covariates and static features."""
        available_columns = [TimeSeriesDataFrame.ITEMID] + self.covariate_metadata.known_covariates
        if include_target:
            available_columns += [self.target]
        tabular_df = (
            pd.DataFrame(data).reset_index()[available_columns].astype({TimeSeriesDataFrame.ITEMID: "category"})
        )
        if static_features is not None and self.include_static_features:
            tabular_df = pd.merge(tabular_df, static_features, on=TimeSeriesDataFrame.ITEMID)
        return tabular_df

    def _subsample_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Randomly subsample the dataframe if it contains more than self.max_num_samples rows."""
        if self.max_num_samples is not None and len(df) > self.max_num_samples:
            df = df.sample(n=self.max_num_samples)
        return df


@overload
def get_covariate_regressor(covariate_regressor: None, target: str, covariate_metadata: CovariateMetadata) -> None: ...
@overload
def get_covariate_regressor(
    covariate_regressor: Union[str, dict], target: str, covariate_metadata: CovariateMetadata
) -> CovariateRegressor: ...
def get_covariate_regressor(
    covariate_regressor: Optional[Union[str, dict]], target: str, covariate_metadata: CovariateMetadata
) -> Optional[CovariateRegressor]:
    """Create a CovariateRegressor object based on the value of the `covariate_regressor` hyperparameter."""
    if covariate_regressor is None:
        return None
    elif len(covariate_metadata.known_covariates + covariate_metadata.static_features) == 0:
        logger.info(
            "\tSkipping covariate_regressor since the dataset contains no known_covariates or static_features."
        )
        return None
    else:
        if isinstance(covariate_regressor, str):
            return GlobalCovariateRegressor(covariate_regressor, target=target, covariate_metadata=covariate_metadata)
        elif isinstance(covariate_regressor, dict):
            return GlobalCovariateRegressor(
                **covariate_regressor, target=target, covariate_metadata=covariate_metadata
            )
        else:
            raise ValueError(
                f"Invalid value for covariate_regressor {covariate_regressor} of type {type(covariate_regressor)}"
            )
