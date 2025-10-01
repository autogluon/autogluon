from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Self


class EnsembleRegressor(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, base_model_predictions: np.ndarray, labels: np.ndarray, **kwargs) -> Self:
        """
        Parameters
        ----------
        base_model_predictions
            Predictions of base models. Array of shape
            (windows, items, prediction_length, quantiles, model)

        labels
            Ground truth array of shape
            (windows, items, prediction_length)
        """
        pass

    @abstractmethod
    def predict(self, base_model_predictions: np.ndarray) -> np.ndarray:
        """Predict with the fitted ensemble regressor for a single window.

        Parameters
        ----------
        base_model_predictions
            Predictions of base models. Array of shape
            (1, items, prediction_length, quantiles, model)

        Returns
        -------
        ensemble_predictions
            Array of shape (1, items, prediction_length, quantiles)
        """
        pass

    def predict_cold(self, base_model_predictions: np.ndarray) -> np.ndarray:
        """Predict with the fitted ensemble regressor for a single window, where
        the items do not refer to the same item indices used when fitting the model.

        Parameters
        ----------
        base_model_predictions
            Predictions of base models. Array of shape
            (1, items, prediction_length, quantiles, model)

        Returns
        -------
        ensemble_predictions
            Array of shape (1, items, prediction_length, quantiles)
        """
        raise NotImplementedError


class SimpleAverageEnsembleRegressor(EnsembleRegressor):
    def fit(self, base_model_predictions: np.ndarray, labels: np.ndarray, **kwargs) -> Self:
        return self

    def predict(self, base_model_predictions: np.ndarray) -> np.ndarray:
        return np.mean(base_model_predictions, axis=-1)
