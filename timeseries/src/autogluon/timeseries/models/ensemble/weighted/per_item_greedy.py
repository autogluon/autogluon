import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.local.abstract_local_model import AG_DEFAULT_N_JOBS

from ..abstract import AbstractTimeSeriesEnsembleModel
from .greedy import TimeSeriesEnsembleSelection

logger = logging.getLogger(__name__)


class PerItemGreedyEnsemble(AbstractTimeSeriesEnsembleModel):
    """Fits a separate greedy weighted ensemble for each individual time series in the dataset.
    Constructs a weighted ensemble using the greedy Ensemble Selection algorithm by Caruana et al. [Car2004]

    Other Parameters
    ----------------
    ensemble_size: int, default = 100
        Number of models (with replacement) to include in the ensemble.

    References
    ----------
    .. [Car2024] Caruana, Rich, et al. "Ensemble selection from libraries of models."
        Proceedings of the twenty-first international conference on Machine learning. 2004.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        if name is None:
            name = "PerItemWeightedEnsemble"
        super().__init__(name=name, **kwargs)
        self.weights_df: Optional[pd.DataFrame] = None
        self.average_weight: Optional[pd.Series] = None

    @property
    def model_names(self) -> list[str]:
        if self.weights_df is None:
            return []
        return list(self.weights_df.columns)

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        return {"ensemble_size": 100, "n_jobs": AG_DEFAULT_N_JOBS}

    def _fit(
        self,
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        data_per_window: list[TimeSeriesDataFrame],
        model_scores: Optional[dict[str, float]] = None,
        time_limit: Optional[float] = None,
    ) -> None:
        model_names = list(predictions_per_window.keys())
        all_items = data_per_window[0].item_ids
        n_jobs = self.get_hyperparameters()["n_jobs"]

        # Split predictions and labels by item using indptr
        predictions_by_item = self._split_predictions_by_item(predictions_per_window, model_names)
        labels_by_item = self._split_data_by_item(data_per_window)

        # Fit ensemble for each item in parallel
        executor = Parallel(n_jobs=n_jobs)
        weights_per_item = executor(
            delayed(self._fit_item_ensemble)(predictions_by_item[i], labels_by_item[i]) for i in range(len(all_items))
        )
        self.weights_df = pd.DataFrame(weights_per_item, index=all_items, columns=model_names)
        self.average_weight = self.weights_df.mean(axis=0)

        # Drop models with zero average weight
        models_to_keep = self.average_weight[self.average_weight > 0].index
        self.weights_df = self.weights_df[models_to_keep]
        self.average_weight = self.average_weight[models_to_keep]

    def _split_predictions_by_item(
        self, predictions_per_window: dict[str, list[TimeSeriesDataFrame]], model_names: list[str]
    ) -> list[list[list[TimeSeriesDataFrame]]]:
        """Split predictions by item using prediction_length slicing."""
        num_items = len(predictions_per_window[model_names[0]][0].item_ids)
        predictions_by_item = []
        for item_idx in range(num_items):
            item_predictions = []
            for model_name in model_names:
                item_preds_per_window = [
                    TimeSeriesDataFrame(
                        pred.iloc[item_idx * self.prediction_length : (item_idx + 1) * self.prediction_length]
                    )
                    for pred in predictions_per_window[model_name]
                ]
                item_predictions.append(item_preds_per_window)
            predictions_by_item.append(item_predictions)
        return predictions_by_item

    def _split_data_by_item(self, data_per_window: list[TimeSeriesDataFrame]) -> list[list[TimeSeriesDataFrame]]:
        """Return ground truth values corresponding to each item."""
        labels_by_item = [[] for _ in range(data_per_window[0].num_items)]
        for data in data_per_window:
            indptr = data.get_indptr()
            for item_idx in range(data.num_items):
                new_slice = data.iloc[indptr[item_idx] : indptr[item_idx + 1]]
                labels_by_item[item_idx].append(new_slice)
        return labels_by_item

    def _fit_item_ensemble(
        self, predictions: list[list[TimeSeriesDataFrame]], labels: list[TimeSeriesDataFrame]
    ) -> np.ndarray:
        """Fit ensemble for a single item."""
        ensemble_selection = TimeSeriesEnsembleSelection(
            ensemble_size=self.get_hyperparameters()["ensemble_size"],
            metric=self.eval_metric,
            prediction_length=self.prediction_length,
            target=self.target,
        )
        ensemble_selection.fit(
            predictions=predictions,
            labels=labels,
            # TODO: Implement the time_limit
            time_limit=None,
        )
        return ensemble_selection.weights_

    def _predict(self, data: dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        first_model = next(iter(data.keys()))
        item_ids = data[first_model].item_ids

        weights = self.weights_df.reindex(item_ids).fillna(self.average_weight)

        result = None
        for model_name, model_pred in data.items():
            model_weights = weights[model_name].to_numpy().repeat(self.prediction_length)
            weighted_pred = model_pred.to_data_frame().multiply(model_weights, axis=0)
            result = weighted_pred if result is None else result + weighted_pred

        return result

    def remap_base_models(self, model_refit_map: dict[str, str]) -> None:
        self.weights_df.rename(columns=model_refit_map, inplace=True)
