from typing import Dict, List, Optional

import numpy as np

from autogluon.timeseries.dataset import TimeSeriesDataFrame

from .abstract import AbstractWeightedTimeSeriesEnsembleModel


class SimpleAverageEnsemble(AbstractWeightedTimeSeriesEnsembleModel):
    """Constructs a weighted ensemble using a simple average of the constituent models' predictions."""

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


class PerformanceWeightedEnsemble(AbstractWeightedTimeSeriesEnsembleModel):
    """Constructs a weighted ensemble, where the weights are assigned in proportion to the
    (inverse) validation scores.

    Other Parameters
    ----------------
    weight_scheme: Literal["sq", "inv", "loginv"], default = "loginv"
        Method used to compute the weights as a function of the validation scores.
        - "sqrt" computes weights in proportion to `sqrt(1 / S)`. This is the default.
        - "inv" computes weights in proportion to `(1 / S)`.
        - "sq" computes the weights in proportion to `(1 / S)^2` as outlined in [PC2020]_.

    References
    ----------
    .. [PC2020] Pawlikowski, Maciej, and Agata Chorowska.
        "Weighted ensemble of statistical models." International Journal of Forecasting
        36.1 (2020): 93-97.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        if name is None:
            name = "PerformanceWeightedEnsemble"
        super().__init__(name=name, **kwargs)

    def _get_default_hyperparameters(self) -> Dict:
        return {"weight_scheme": "sqrt"}

    def _fit(
        self,
        predictions_per_window: Dict[str, List[TimeSeriesDataFrame]],
        data_per_window: List[TimeSeriesDataFrame],
        model_scores: Optional[Dict[str, float]] = None,
        time_limit: Optional[float] = None,
    ):
        assert model_scores is not None

        weight_scheme = self.get_hyperparameters()["weight_scheme"]

        # drop NaNs
        model_scores = {k: v for k, v in model_scores.items() if np.isfinite(v)}
        assert len(model_scores) > 0, (
            "All models have NaN scores. At least one model must score successfully to fit an ensemble"
        )
        assert all(s <= 0 for s in model_scores.values()), (
            "All model scores must be negative, in higher-is-better format."
        )

        score_transform = {
            "sq": lambda x: np.square(np.reciprocal(x)),
            "inv": lambda x: np.reciprocal(x),
            "sqrt": lambda x: np.sqrt(np.reciprocal(x)),
        }[weight_scheme]

        self.model_to_weight = {
            model_name: score_transform(-model_scores[model_name] + 1e-5) for model_name in model_scores.keys()
        }
        total_weight = sum(self.model_to_weight.values())
        self.model_to_weight = {k: v / total_weight for k, v in self.model_to_weight.items()}
