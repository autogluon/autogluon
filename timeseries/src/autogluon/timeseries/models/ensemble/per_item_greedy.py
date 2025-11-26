import logging
from typing import Any, Optional

import pandas as pd
from joblib import Parallel, delayed

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.constants import AG_DEFAULT_N_JOBS

from .abstract import AbstractTimeSeriesEnsembleModel
from .ensemble_selection import fit_time_series_ensemble_selection

logger = logging.getLogger(__name__)


class PerItemGreedyEnsemble(AbstractTimeSeriesEnsembleModel):
    """Fits a separate greedy weighted ensemble for each individual time series in the dataset.
    Constructs a weighted ensemble using the greedy Ensemble Selection algorithm by Caruana et al. [Car2004]

    Other Parameters
    ----------------
    ensemble_size: int, default = 100
        Number of models (with replacement) to include in the ensemble.
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the ensembles in parallel.

    References
    ----------
    .. [Car2024] Caruana, Rich, et al. "Ensemble selection from libraries of models."
        Proceedings of the twenty-first international conference on Machine learning. 2004.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        if name is None:
            name = "PerItemWeightedEnsemble"
        super().__init__(name=name, **kwargs)
        self.weights_df: pd.DataFrame
        self.average_weight: pd.Series

    @property
    def model_names(self) -> list[str]:
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
        item_ids = data_per_window[0].item_ids
        n_jobs = self.get_hyperparameters()["n_jobs"]

        predictions_per_item = self._split_predictions_per_item(predictions_per_window)
        data_per_item = self._split_data_per_item(data_per_window)

        ensemble_selection_kwargs = dict(
            ensemble_size=self.get_hyperparameters()["ensemble_size"],
            eval_metric=self.eval_metric,
            prediction_length=self.prediction_length,
            target=self.target,
        )

        # Fit ensemble for each item in parallel
        executor = Parallel(n_jobs=n_jobs)
        # TODO: add time_limit
        weights_per_item = executor(
            delayed(fit_time_series_ensemble_selection)(
                data_per_item[item_id], predictions_per_item[item_id], **ensemble_selection_kwargs
            )
            for item_id in item_ids
        )
        self.weights_df = pd.DataFrame(weights_per_item, index=item_ids, columns=model_names)
        self.average_weight = self.weights_df.mean(axis=0)

        # Drop models with zero average weight
        if (self.average_weight == 0).any():
            models_to_keep = self.average_weight[self.average_weight > 0].index
            self.weights_df = self.weights_df[models_to_keep]
            self.average_weight = self.average_weight[models_to_keep]

    def _split_predictions_per_item(
        self, predictions_per_window: dict[str, list[TimeSeriesDataFrame]]
    ) -> dict[str, dict[str, list[TimeSeriesDataFrame]]]:
        """Build a dictionary mapping item_id -> dict[model_name, list[TimeSeriesDataFrame]]."""
        first_model_preds = list(predictions_per_window.values())[0]
        item_ids = first_model_preds[0].item_ids

        predictions_per_item = {}
        for i, item_id in enumerate(item_ids):
            item_predictions = {}
            for model_name, preds_per_window in predictions_per_window.items():
                item_preds_per_window = [
                    pred.iloc[i * self.prediction_length : (i + 1) * self.prediction_length]
                    for pred in preds_per_window
                ]
                item_predictions[model_name] = item_preds_per_window
            predictions_per_item[item_id] = item_predictions
        return predictions_per_item

    def _split_data_per_item(self, data_per_window: list[TimeSeriesDataFrame]) -> dict[str, list[TimeSeriesDataFrame]]:
        """Build a dictionary mapping item_id -> ground truth values across all windows."""
        item_ids = data_per_window[0].item_ids
        data_per_item = {item_id: [] for item_id in item_ids}

        for data in data_per_window:
            indptr = data.get_indptr()
            for item_idx, item_id in enumerate(item_ids):
                new_slice = data.iloc[indptr[item_idx] : indptr[item_idx + 1]]
                data_per_item[item_id].append(new_slice)
        return data_per_item

    @staticmethod
    def _fit_item_ensemble(
        data_per_window: list[TimeSeriesDataFrame],
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        **ensemble_selection_kwargs,
    ) -> dict[str, float]:
        """Fit ensemble for a single item."""
        return fit_time_series_ensemble_selection(data_per_window, predictions_per_window, **ensemble_selection_kwargs)

    def _predict(self, data: dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        first_model = next(iter(data.keys()))
        item_ids = data[first_model].item_ids

        assert all(model in data for model in self.weights_df.columns)
        weights = self.weights_df.reindex(item_ids).fillna(self.average_weight)

        result = None
        for model_name, model_pred in data.items():
            model_weights = weights[model_name].to_numpy().repeat(self.prediction_length)
            weighted_pred = model_pred.to_data_frame().multiply(model_weights, axis=0)
            result = weighted_pred if result is None else result + weighted_pred

        return TimeSeriesDataFrame(result)

    def remap_base_models(self, model_refit_map: dict[str, str]) -> None:
        self.weights_df.rename(columns=model_refit_map, inplace=True)
