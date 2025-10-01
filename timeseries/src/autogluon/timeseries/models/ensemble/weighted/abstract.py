import functools
from abc import ABC
from typing import Optional

import numpy as np

from autogluon.timeseries.dataset import TimeSeriesDataFrame

from ..abstract import AbstractTimeSeriesEnsembleModel


class AbstractWeightedTimeSeriesEnsembleModel(AbstractTimeSeriesEnsembleModel, ABC):
    """Abstract class for weighted ensembles which assign one (global) weight per model."""

    _default_model_name = "WeightedEnsemble"

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model_to_weight: dict[str, float] = {}

    @property
    def model_names(self) -> list[str]:
        return list(self.model_to_weight.keys())

    @property
    def model_weights(self) -> np.ndarray:
        return np.array(list(self.model_to_weight.values()), dtype=np.float64)

    def _predict(self, data: dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        weighted_predictions = [data[model_name] * weight for model_name, weight in self.model_to_weight.items()]
        return functools.reduce(lambda x, y: x + y, weighted_predictions)

    def get_info(self) -> dict:
        info = super().get_info()
        info["model_weights"] = self.model_to_weight.copy()
        return info

    def remap_base_models(self, model_refit_map: dict[str, str]) -> None:
        updated_weights = {}
        for model, weight in self.model_to_weight.items():
            model_full_name = model_refit_map.get(model, model)
            updated_weights[model_full_name] = weight
        self.model_to_weight = updated_weights
