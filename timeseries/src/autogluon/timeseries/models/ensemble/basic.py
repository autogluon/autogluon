from typing import Dict, List, Literal, Optional

import numpy as np

from autogluon.timeseries.dataset import TimeSeriesDataFrame

from .abstract import AbstractWeightedTimeSeriesEnsembleModel


class TimeSeriesSimpleAverageEnsemble(AbstractWeightedTimeSeriesEnsembleModel):
    """Constructs a weighted ensemble using a simple average of the constituent
    models' predictions.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        if name is None:
            name = "SimpleAverageEnsemble"
        super().__init__(name=name, **kwargs)

    def _fit(
        self,
        predictions_per_window: Dict[str, List[TimeSeriesDataFrame]],
        data_per_window: List[TimeSeriesDataFrame],
        model_scores: Optional[Dict[str, float]] = None,
        time_limit: Optional[float] = None,
    ):
        self.model_to_weight = {}
        num_models = len(predictions_per_window)
        for model_name in predictions_per_window.keys():
            self.model_to_weight[model_name] = 1.0 / num_models


class TimeSeriesPerformanceWeightedEnsemble(AbstractWeightedTimeSeriesEnsembleModel):
    """Constructs a weighted ensemble, where the weights are assigned in proportion to the
    (inverse) validation scores, using the method followed by Pawlikowski and Chorowska [PC2020]_.

    References
    ----------
    .. [PC2020] Pawlikowski, Maciej, and Agata Chorowska.
        "Weighted ensemble of statistical models." International Journal of Forecasting
        36.1 (2020): 93-97.
    """

    def __init__(self, name: Optional[str] = None, weight_scheme: Literal["sq", "exp"] = "sq", **kwargs):
        if name is None:
            name = "PerformanceWeightedEnsemble"
        super().__init__(name=name, **kwargs)
        self.weight_scheme = weight_scheme

    def _fit(
        self,
        predictions_per_window: Dict[str, List[TimeSeriesDataFrame]],
        data_per_window: List[TimeSeriesDataFrame],
        model_scores: Optional[Dict[str, float]] = None,
        time_limit: Optional[float] = None,
    ):
        assert model_scores is not None

        self.model_to_weight = {}
        score_transform = np.square if self.weight_scheme == "sq" else np.exp

        for model_name in predictions_per_window.keys():
            self.model_to_weight[model_name] = 1.0 / (score_transform(model_scores[model_name]) + 1e-5)

        total_weight = sum(self.model_to_weight.values())
        self.model_to_weight = {k: v / total_weight for k, v in self.model_to_weight.items()}
