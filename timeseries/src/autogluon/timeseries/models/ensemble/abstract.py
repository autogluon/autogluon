import functools
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from typing_extensions import final

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import TimeSeriesModelBase

logger = logging.getLogger(__name__)


class AbstractTimeSeriesEnsembleModel(TimeSeriesModelBase, ABC):
    """Abstract class for time series ensemble models."""

    @property
    @abstractmethod
    def model_names(self) -> List[str]:
        """Names of base models included in the ensemble."""
        pass

    @final
    def fit(
        self,
        predictions_per_window: Dict[str, List[TimeSeriesDataFrame]],
        data_per_window: List[TimeSeriesDataFrame],
        model_scores: Optional[Dict[str, float]] = None,
        time_limit: Optional[float] = None,
    ):
        """Fit ensemble model given predictions of candidate base models and the true data.

        Parameters
        ----------
        predictions_per_window : Dict[str, List[TimeSeriesDataFrame]]
            Dictionary that maps the names of component models to their respective predictions for each validation
            window.
        data_per_window : List[TimeSeriesDataFrame]
            Observed ground truth data used to train the ensemble for each validation window. Each entry in the list
            includes both the forecast horizon (for which the predictions are given in ``predictions``), as well as the
            "history".
        model_scores : Optional[Dict[str, float]]
            Scores (higher is better) for the models that will constitute the ensemble.
        time_limit : Optional[float]
            Maximum allowed time for training in seconds.
        """
        if time_limit is not None and time_limit <= 0:
            logger.warning(
                f"\tWarning: Model has no time left to train, skipping model... (Time Left = {round(time_limit, 1)}s)"
            )
            raise TimeLimitExceeded
        if isinstance(data_per_window, TimeSeriesDataFrame):
            raise ValueError("When fitting ensemble, `data` should contain ground truth for each validation window")
        num_val_windows = len(data_per_window)
        for model, preds in predictions_per_window.items():
            if len(preds) != num_val_windows:
                raise ValueError(f"For model {model} predictions are unavailable for some validation windows")
        self._fit(
            predictions_per_window=predictions_per_window,
            data_per_window=data_per_window,
            model_scores=model_scores,
            time_limit=time_limit,
        )
        return self

    def _fit(
        self,
        predictions_per_window: Dict[str, List[TimeSeriesDataFrame]],
        data_per_window: List[TimeSeriesDataFrame],
        model_scores: Optional[Dict[str, float]] = None,
        time_limit: Optional[float] = None,
    ):
        """Private method for `fit`. See `fit` for documentation of arguments. Apart from the model
        training logic, `fit` additionally implements other logic such as keeping track of the time limit.
        """
        raise NotImplementedError

    @final
    def predict(self, data: Dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        if not set(self.model_names).issubset(set(data.keys())):
            raise ValueError(
                f"Set of models given for prediction in {self.name} differ from those provided during initialization."
            )
        for model_name, model_pred in data.items():
            if model_pred is None:
                raise RuntimeError(f"{self.name} cannot predict because base model {model_name} failed.")

        # Make sure that all predictions have same shape
        assert len(set(pred.shape for pred in data.values())) == 1

        return self._predict(data=data, **kwargs)

    @abstractmethod
    def _predict(self, data: Dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        pass

    @abstractmethod
    def remap_base_models(self, model_refit_map: Dict[str, str]) -> None:
        """Update names of the base models based on the mapping in model_refit_map.

        This method should be called after performing refit_full to point to the refitted base models, if necessary.
        """
        pass


class AbstractWeightedTimeSeriesEnsembleModel(AbstractTimeSeriesEnsembleModel, ABC):
    """Abstract class for weighted ensembles which assign one (global) weight per model."""

    def __init__(self, name: Optional[str] = None, **kwargs):
        if name is None:
            name = "WeightedEnsemble"
        super().__init__(name=name, **kwargs)
        self.model_to_weight: Dict[str, float] = {}

    @property
    def model_names(self) -> List[str]:
        return list(self.model_to_weight.keys())

    @property
    def model_weights(self) -> np.ndarray:
        return np.array(list(self.model_to_weight.values()), dtype=np.float64)

    def _predict(self, data: Dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        weighted_predictions = [data[model_name] * weight for model_name, weight in self.model_to_weight.items()]
        return functools.reduce(lambda x, y: x + y, weighted_predictions)

    def get_info(self) -> dict:
        info = super().get_info()
        info["model_weights"] = self.model_to_weight.copy()
        return info

    def remap_base_models(self, model_refit_map: Dict[str, str]) -> None:
        updated_weights = {}
        for model, weight in self.model_to_weight.items():
            model_full_name = model_refit_map.get(model, model)
            updated_weights[model_full_name] = weight
        self.model_to_weight = updated_weights
