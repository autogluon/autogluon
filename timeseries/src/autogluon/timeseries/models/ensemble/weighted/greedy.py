import logging
import pprint
from typing import Any

from autogluon.timeseries import TimeSeriesDataFrame

from ..ensemble_selection import fit_time_series_ensemble_selection
from .abstract import AbstractWeightedTimeSeriesEnsembleModel

logger = logging.getLogger(__name__)


class GreedyEnsemble(AbstractWeightedTimeSeriesEnsembleModel):
    """Constructs a weighted ensemble using the greedy Ensemble Selection algorithm by
    Caruana et al. [Car2004]

    Other Parameters
    ----------------
    ensemble_size: int, default = 100
        Number of models (with replacement) to include in the ensemble.

    References
    ----------
    .. [Car2024] Caruana, Rich, et al. "Ensemble selection from libraries of models."
        Proceedings of the twenty-first international conference on Machine learning. 2004.
    """

    def __init__(self, name: str | None = None, **kwargs):
        if name is None:
            # FIXME: the name here is kept for backward compatibility. it will be called
            # GreedyEnsemble in v1.4 once ensemble choices are exposed
            name = "WeightedEnsemble"
        super().__init__(name=name, **kwargs)

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        return {"ensemble_size": 100}

    def _fit(
        self,
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        data_per_window: list[TimeSeriesDataFrame],
        model_scores: dict[str, float] | None = None,
        time_limit: float | None = None,
    ):
        model_to_weight = fit_time_series_ensemble_selection(
            data_per_window=data_per_window,
            predictions_per_window=predictions_per_window,
            ensemble_size=self.get_hyperparameter("ensemble_size"),
            eval_metric=self.eval_metric,
            prediction_length=self.prediction_length,
            target=self.target,
            time_limit=time_limit,
        )
        self.model_to_weight = {model: weight for model, weight in model_to_weight.items() if weight > 0}

        weights_for_printing = {model: round(float(weight), 2) for model, weight in self.model_to_weight.items()}
        logger.info(f"\tEnsemble weights: {pprint.pformat(weights_for_printing, width=200)}")
