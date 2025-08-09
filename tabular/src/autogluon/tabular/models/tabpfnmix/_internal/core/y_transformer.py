import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import QuantileTransformer

from .enums import Task


def create_y_transformer(y_train: np.ndarray, task: Task) -> TransformerMixin:
    # The y_transformer transforms the target variable to a normal distribution
    # This should be used for the y variable when training a regression model,
    # but when testing the model, we want to inverse transform the predictions

    if task == Task.REGRESSION:
        y_transformer = QuantileTransformer1D(output_distribution="uniform")
        y_transformer.fit(y_train)
        return y_transformer
    else:
        # Identity
        return FunctionTransformer()


class QuantileTransformer1D(BaseEstimator, TransformerMixin):

    def __init__(self, output_distribution="normal") -> None:
        self.quantile_transformer = QuantileTransformer(output_distribution=output_distribution)

    def fit(self, x: np.ndarray):
        self.quantile_transformer.fit(x[:, None])
        return self
    
    def transform(self, x: np.ndarray):
        return self.quantile_transformer.transform(x[:, None])[:, 0]
    
    def inverse_transform(self, x: np.ndarray):
        return self.quantile_transformer.inverse_transform(x[:, None])[:, 0]