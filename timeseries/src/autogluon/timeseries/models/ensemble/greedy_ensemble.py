import copy
import logging
import pprint
from typing import Dict, List, Optional

import numpy as np

import autogluon.core as ag
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer
from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel
from autogluon.timeseries.utils.datetime import get_seasonality

logger = logging.getLogger(__name__)


class TimeSeriesEnsembleSelection(EnsembleSelection):
    def __init__(
        self,
        ensemble_size: int,
        metric: TimeSeriesScorer,
        problem_type: str = ag.constants.QUANTILE,
        sorted_initialization: bool = False,
        bagging: bool = False,
        tie_breaker: str = "random",
        random_state: np.random.RandomState = None,
        prediction_length: int = 1,
        target: str = "target",
        eval_metric_seasonal_period: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            metric=metric,
            problem_type=problem_type,
            sorted_initialization=sorted_initialization,
            bagging=bagging,
            tie_breaker=tie_breaker,
            random_state=random_state,
            **kwargs,
        )
        self.prediction_length = prediction_length
        self.target = target
        self.eval_metric_seasonal_period = eval_metric_seasonal_period

    def _fit(
        self,
        predictions: List[List[TimeSeriesDataFrame]],  # first dim: model, second dim: val window index
        labels: List[TimeSeriesDataFrame],
        time_limit: Optional[int] = None,
        sample_weight=None,
    ):
        # Stack predictions for each model into a 3d tensor of shape [num_val_windows, num_rows, num_cols]
        stacked_predictions = [np.stack(preds) for preds in predictions]

        self.dummy_pred_per_window = []
        self.scorer_per_window = []
        self.data_future_per_window = []

        for window_idx, data in enumerate(labels):
            dummy_pred = copy.deepcopy(predictions[0][window_idx])
            # This should never happen; sanity check to make sure that all predictions have the same index
            assert all(dummy_pred.index.equals(pred[window_idx].index) for pred in predictions)

            self.dummy_pred_per_window.append(dummy_pred)

            scorer = copy.deepcopy(self.metric)
            # Split the observed time series once to avoid repeated computations inside the evaluator
            data_past = data.slice_by_timestep(None, -self.prediction_length)
            data_future = data.slice_by_timestep(-self.prediction_length, None)
            scorer.save_past_metrics(data_past, target=self.target, seasonal_period=self.eval_metric_seasonal_period)
            self.scorer_per_window.append(scorer)
            self.data_future_per_window.append(data_future)

        super()._fit(
            predictions=stacked_predictions,
            labels=data_future,
            time_limit=time_limit,
        )
        self.dummy_pred_per_window = None
        self.evaluator_per_window = None
        self.data_future_per_window = None

    def _calculate_regret(self, y_true, y_pred_proba, metric=None, sample_weight=None):  # noqa
        # Compute average score across all validation windows
        total_score = 0.0
        for window_idx, data_future in enumerate(self.data_future_per_window):
            dummy_pred = self.dummy_pred_per_window[window_idx]
            dummy_pred[list(dummy_pred.columns)] = y_pred_proba[window_idx]
            # We use scorer.compute_metric instead of scorer.score to avoid repeated calls to scorer.save_past_metrics
            metric_value = self.scorer_per_window[window_idx].compute_metric(
                data_future, dummy_pred, target=self.target
            )
            total_score += metric.sign * metric_value
        avg_score = total_score / len(self.data_future_per_window)
        # score: higher is better, regret: lower is better, so we flip the sign
        return -avg_score


class TimeSeriesGreedyEnsemble(AbstractTimeSeriesEnsembleModel):
    """Constructs a weighted ensemble using the greedy Ensemble Selection algorithm."""

    def __init__(self, name: str, ensemble_size: int = 100, **kwargs):
        super().__init__(name=name, **kwargs)
        self.ensemble_size = ensemble_size
        self.model_to_weight: Dict[str, float] = {}

    def _fit_ensemble(
        self,
        predictions_per_window: Dict[str, List[TimeSeriesDataFrame]],
        data_per_window: List[TimeSeriesDataFrame],
        time_limit: Optional[int] = None,
        verbosity: int = 2,
        **kwargs,
    ):
        set_logger_verbosity(verbosity, logger=logger)
        if self.eval_metric_seasonal_period is None:
            self.eval_metric_seasonal_period = get_seasonality(self.freq)
        ensemble_selection = TimeSeriesEnsembleSelection(
            ensemble_size=self.ensemble_size,
            metric=self.eval_metric,
            prediction_length=self.prediction_length,
            target=self.target,
            eval_metric_seasonal_period=self.eval_metric_seasonal_period,
        )
        ensemble_selection.fit(
            predictions=list(predictions_per_window.values()),
            labels=data_per_window,
            time_limit=time_limit,
        )
        self.model_to_weight = {}
        for model_name, weight in zip(predictions_per_window.keys(), ensemble_selection.weights_):
            if weight != 0:
                self.model_to_weight[model_name] = weight

        weights_for_printing = {model: round(weight, 2) for model, weight in self.model_to_weight.items()}
        logger.info(f"\tEnsemble weights: {pprint.pformat(weights_for_printing, width=200)}")

    @property
    def model_names(self) -> List[str]:
        return list(self.model_to_weight.keys())

    @property
    def model_weights(self) -> np.ndarray:
        return np.array(list(self.model_to_weight.values()), dtype=np.float64)

    def predict(self, data: Dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        if set(data.keys()) != set(self.model_names):
            raise ValueError(
                f"Set of models given for prediction in {self.name} differ from those provided during initialization."
            )
        for model_name, model_pred in data.items():
            if model_pred is None:
                raise RuntimeError(f"{self.name} cannot predict because base model {model_name} failed.")

        # Make sure that all predictions have same shape
        assert len(set(pred.shape for pred in data.values())) == 1

        return sum(data[model_name] * weight for model_name, weight in self.model_to_weight.items())

    def get_info(self) -> dict:
        info = super().get_info()
        info["model_weights"] = self.model_to_weight
        return info

    def remap_base_models(self, model_refit_map: Dict[str, str]) -> None:
        updated_weights = {}
        for model, weight in self.model_to_weight.items():
            model_full_name = model_refit_map.get(model, model)
            updated_weights[model_full_name] = weight
        self.model_to_weight = updated_weights
