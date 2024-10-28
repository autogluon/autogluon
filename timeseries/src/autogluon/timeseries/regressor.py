from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from autogluon.core.models import AbstractModel
from autogluon.tabular.trainer.model_presets.presets import MODEL_TYPES as TABULAR_MODEL_TYPES
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame
from autogluon.timeseries.utils.features import CovariateMetadata


class CovariateRegressor:
    """Predicts target values from the covariates for the same observation.

    The model construct the feature matrix using known_covariates and static_features.

    Parameters
    ----------
    model_name : str
        Name of the tabular regression model. See `autogluon.tabular.trainer.model_presets.presets.MODEL_TYPES` for the
        list of available models.
    model_hyperparameters : dict or None
        Hyperparameters passed to the tabular regression model.
    eval_metric : str
        Metric provided as `eval_metric` to the tabular regression model. Must be compatible with `problem_type="regression"`.
    refit_during_predict : bool
        If True, the model will be re-trained every time `fit_transform` is called. If False, the model will only be
        trained the first time that `fit_transform` is called, and future calls to `fit_transform` will only perform a
        `transform`.
    max_num_samples : int or None
        If not None, training dataset passed to regression model will contain at most this many rows.
    metadata : CovariateMetadata
        Metadata object describing the covariates available in the dataset.
    target : str
        Name of the target column.
    validation_frac : float, optional
        Fraction of observations that are reserved as the validation set during training (starting from the end of each
        time series).
    """

    def __init__(
        self,
        model_name: str = "GBM",
        model_hyperparameters: Optional[Dict[str, Any]] = None,
        eval_metric: str = "mean_absolute_error",
        refit_during_predict: bool = False,
        max_num_samples: Optional[int] = 500_000,
        metadata: Optional[CovariateMetadata] = None,
        target: str = "target",
        validation_fraction: Optional[float] = 0.1,
    ):
        if model_name not in TABULAR_MODEL_TYPES:
            raise ValueError(
                f"Tabular model {model_name} not supported. Available models: {list(TABULAR_MODEL_TYPES)}"
            )
        self.target = target
        self.model_type = TABULAR_MODEL_TYPES[model_name]
        self.model_name = model_name
        self.model_hyperparameters = model_hyperparameters or {}
        self.refit_during_predict = refit_during_predict
        self.tabular_eval_metric = eval_metric
        self.max_num_samples = max_num_samples
        self.validation_fraction = validation_fraction
        self.model: Optional[AbstractModel] = None
        self.metadata = metadata or CovariateMetadata()

    def is_fit(self) -> bool:
        return self.model is not None

    def fit(self, data: TimeSeriesDataFrame, time_limit: Optional[float] = None, **kwargs) -> "CovariateRegressor":
        """Fit the tabular regressor on the target column using covariates as features."""
        tabular_df = self._get_tabular_df(data, static_features=data.static_features, include_target=True)
        tabular_df = tabular_df.query(f"{self.target}.notnull()")

        median_ts_length = data.num_timesteps_per_item().median()
        if self.validation_fraction is not None:
            grouped_df = tabular_df.groupby(ITEMID)
            val_size = max(int(self.validation_fraction * median_ts_length), 1)
            train_df = self._subsample_df(grouped_df.head(-val_size))
            val_df = self._subsample_df(grouped_df.tail(val_size))
            X = train_df.drop(columns=[self.target])
            y = train_df[self.target]
            X_val = val_df.drop(columns=[self.target])
            y_val = val_df[self.target]
        else:
            tabular_df = self._subsample_df(tabular_df)
            X = tabular_df.drop(columns=[self.target])
            y = tabular_df[self.target]
            X_val = None
            y_val = None

        self.model = self.model_type(
            problem_type="regression",
            hyperparameters=self.model_hyperparameters,
            eval_metric=self.tabular_eval_metric,
        )
        self.model.fit(X=X, y=y, X_val=X_val, y_val=y_val, time_limit=time_limit, **kwargs)
        return self

    def transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Subtract the tabular regressor predictions from the target column."""
        y_pred = self._predict(data, static_features=data.static_features)
        return data.assign(**{self.target: data[self.target] - y_pred})

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
        y_pred = self._predict(known_covariates, static_features=static_features)
        return predictions.assign(**{col: predictions[col] + y_pred for col in predictions.columns})

    def _predict(self, data: TimeSeriesDataFrame, static_features: Optional[pd.DataFrame]) -> np.ndarray:
        """Construct the tabular features matrix and make predictions"""
        tabular_df = self._get_tabular_df(data, static_features=static_features)
        return self.model.predict(X=tabular_df)

    def _get_tabular_df(
        self,
        data: TimeSeriesDataFrame,
        static_features: Optional[pd.DataFrame] = None,
        include_target: bool = False,
    ) -> pd.DataFrame:
        """Construct a tabular dataframe from known covariates and static features."""
        available_columns = [ITEMID] + self.metadata.known_covariates
        if include_target:
            available_columns += [self.target]
        tabular_df = pd.DataFrame(data).reset_index()[available_columns]
        if static_features is not None:
            tabular_df = pd.merge(tabular_df, static_features, on=ITEMID)
        return tabular_df

    def _subsample_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Randomly subsample the dataframe if it contains more than self.max_num_samples rows."""
        if self.max_num_samples is not None and len(df) > self.max_num_samples:
            df = df.sample(n=self.max_num_samples)
        return df
