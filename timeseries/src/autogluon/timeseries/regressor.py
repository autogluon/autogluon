from typing import Any, Dict, Optional, Tuple
import pandas as pd
from autogluon.core.models import AbstractModel
from autogluon.tabular.trainer.model_presets.presets import (
    MODEL_TYPES as TABULAR_MODEL_TYPES,
)
from autogluon.timeseries.dataset.ts_dataframe import (
    ITEMID,
    TIMESTAMP,
    TimeSeriesDataFrame,
)
from autogluon.timeseries.utils.features import CovariateMetadata


class CovariatesRegressor:
    """Predicts y values from the covariates."""

    def __init__(
        self,
        model_name: str = "GBM",
        model_hyperparameters: Optional[Dict[str, Any]] = None,
        tabular_eval_metric: str = "mean_absolute_error",
        refit_during_predict: bool = False,
        max_num_samples: Optional[int] = 500_000,
        metadata: Optional[CovariateMetadata] = None,
        target: str = "target",
        validation_fraction: float = 0.1,
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
        self.tabular_eval_metric = tabular_eval_metric
        self.max_num_samples = max_num_samples
        self.validation_fraction = validation_fraction
        self.model: Optional[AbstractModel] = None
        self.metadata = metadata or CovariateMetadata()

    def _subsample_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.max_num_samples is not None and len(df) > self.max_num_samples:
            df = df.sample(n=self.max_num_samples)
        return df

    def fit(self, data: TimeSeriesDataFrame, time_limit: Optional[float] = None) -> "CovariatesRegressor":
        tabular_df = self._get_tabular_df(data, static_features=data.static_features)
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

        self.model = self._get_tabular_model()
        self.model.fit(time_limit=time_limit, X=X, y=y, X_val=X_val, y_val=y_val)
        return self

    def transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        tabular_df = self._get_tabular_df(data, static_features=data.static_features)
        y_pred = self.model.predict(X=tabular_df)
        return data.assign(**{self.target: data[self.target] - y_pred})

    def fit_transform(self, data: TimeSeriesDataFrame, time_limit: Optional[float] = None) -> TimeSeriesDataFrame:
        return self.fit(data=data, time_limit=time_limit).transform(data=data)

    def inverse_transform(
        self,
        predictions: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame,
        static_features: Optional[pd.DataFrame],
    ) -> TimeSeriesDataFrame:
        tabular_df = self._get_tabular_df(known_covariates, static_features=static_features)
        y_pred_future = self.model.predict(X=tabular_df)
        return predictions.assign(**{col: predictions[col] + y_pred_future for col in predictions.columns})

    def _get_tabular_df(
        self, data: TimeSeriesDataFrame, static_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        tabular_df = pd.DataFrame(data).reset_index().drop(columns=[TIMESTAMP] + self.metadata.past_covariates)
        if static_features is not None:
            tabular_df = pd.merge(tabular_df, static_features, on=ITEMID)
        return tabular_df

    def _get_tabular_model(self) -> AbstractModel:
        return self.model_type(
            problem_type="regression",
            hyperparameters=self.model_hyperparameters,
            eval_metric=self.tabular_eval_metric,
        )
