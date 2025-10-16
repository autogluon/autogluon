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
            (num_windows, num_items, prediction_length, num_quantiles, num_models)

        labels
            Ground truth array of shape
            (num_windows, num_items, prediction_length, 1)
        """
        pass

    @abstractmethod
    def predict(self, base_model_predictions: np.ndarray) -> np.ndarray:
        """Predict with the fitted ensemble regressor for a single window.
        The items do not have to refer to the same item indices used when fitting
        the model.

        Parameters
        ----------
        base_model_predictions
            Predictions of base models. Array of shape
            (1, num_items, prediction_length, num_quantiles, num_models)

        Returns
        -------
        ensemble_predictions
            Array of shape (1, num_items, prediction_length, num_quantiles)
        """
        pass


class MedianEnsembleRegressor(EnsembleRegressor):
    def fit(self, base_model_predictions: np.ndarray, labels: np.ndarray, **kwargs) -> Self:
        return self

    def predict(self, base_model_predictions: np.ndarray) -> np.ndarray:
        return np.nanmedian(base_model_predictions, axis=-1)
