import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from mlforecast.target_transforms import BaseTargetTransform
from sklearn.base import BaseEstimator

import autogluon.core as ag
from autogluon.tabular import TabularPredictor


class TabularRegressor(BaseEstimator):
    """Scikit-learn compatible interface for TabularPredictor."""

    _label_column_name = "y"

    def __init__(self, predictor_init_kwargs: Optional[dict] = None, predictor_fit_kwargs: Optional[dict] = None):
        self.predictor_init_kwargs = predictor_init_kwargs if predictor_init_kwargs is not None else {}
        self.predictor_fit_kwargs = predictor_fit_kwargs if predictor_fit_kwargs is not None else {}

    def get_params(self, deep: bool = True) -> dict:
        return {
            "predictor_init_kwargs": self.predictor_init_kwargs,
            "predictor_fit_kwargs": self.predictor_fit_kwargs,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabularRegressor":
        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)
        df = pd.concat([X, y.rename(self._label_column_name).to_frame()], axis=1)
        self.predictor = TabularPredictor(label=self._label_column_name, **self.predictor_init_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.predictor.fit(df, **self.predictor_fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert isinstance(X, pd.DataFrame)
        return self.predictor.predict(X).values


class TabularQuantileRegressor(TabularRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stored_predictions: List[pd.DataFrame] = []

    def reset_stored_predictions(self) -> None:
        self._stored_predictions = []

    def get_stored_predictions(self) -> List[pd.DataFrame]:
        return self._stored_predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert isinstance(X, pd.DataFrame)
        predictions = self.predictor.predict(X)  # (num_rows, len(quantile_levels))
        self._stored_predictions.append(predictions)
        return predictions[0.5].values


class StandardScaler(BaseTargetTransform):
    """Standardizes the series by dividing by their standard deviation."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.stats_ = (
            df.groupby(self.id_col)[self.target_col]
            .agg(["mean", "std"])
            .rename(columns={"mean": "_mean", "std": "_std"})
        )
        df = df.merge(self.stats_, on=self.id_col)
        df[self.target_col] = (df[self.target_col] - df["_mean"]) / df["_std"].clip(lower=1e-2)
        df = df.drop(columns=["_mean", "_std"])
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(self.stats_, on=self.id_col)
        for col in df.columns.drop([self.id_col, self.time_col, "_mean", "_std"]):
            df[col] = df[col] * df["_std"] + df["_mean"]
        df = df.drop(columns=["_mean", "_std"])
        return df
