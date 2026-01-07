import logging
import pprint
import time
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.constants import AG_DEFAULT_N_JOBS

from .abstract import AbstractTimeSeriesEnsembleModel
from .ensemble_selection import fit_time_series_ensemble_selection

logger = logging.getLogger(__name__)


class PerItemGreedyEnsemble(AbstractTimeSeriesEnsembleModel):
    """Per-item greedy ensemble that fits separate weighted ensembles for each individual time series.

    This ensemble applies the greedy Ensemble Selection algorithm by Caruana et al. [Car2004]_ independently
    to each time series in the dataset, allowing for customized model combinations that adapt to the
    specific characteristics of individual series. Each time series gets its own optimal ensemble weights
    based on predictions for that particular series. If items not seen during training are provided at prediction
    time, average model weight across the training items will be used for their predictions.

    The per-item approach is particularly effective for datasets with heterogeneous time series that
    exhibit different patterns, seasonalities, or noise characteristics.

    The algorithm uses parallel processing to efficiently fit ensembles across all time series.

    Other Parameters
    ----------------
    ensemble_size : int, default = 100
        Number of models (with replacement) to include in the ensemble.
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the ensembles in parallel.

    References
    ----------
    .. [Car2004] Caruana, Rich, et al. "Ensemble selection from libraries of models."
        Proceedings of the twenty-first international conference on Machine learning. 2004.
    """

    def __init__(self, name: str | None = None, **kwargs):
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
        model_scores: dict[str, float] | None = None,
        time_limit: float | None = None,
    ) -> None:
        model_names = list(predictions_per_window.keys())
        item_ids = data_per_window[0].item_ids
        n_jobs = min(self.get_hyperparameter("n_jobs"), len(item_ids))

        predictions_per_item = self._split_predictions_per_item(predictions_per_window)
        data_per_item = self._split_data_per_item(data_per_window)

        ensemble_selection_kwargs = dict(
            ensemble_size=self.get_hyperparameter("ensemble_size"),
            eval_metric=self.eval_metric,
            prediction_length=self.prediction_length,
            target=self.target,
        )

        time_limit_per_item = None if time_limit is None else time_limit * n_jobs / len(item_ids)
        end_time = None if time_limit is None else time.time() + time_limit

        # Fit ensemble for each item in parallel
        executor = Parallel(n_jobs=n_jobs)
        weights_per_item = executor(
            delayed(self._fit_item_ensemble)(
                data_per_item[item_id],
                predictions_per_item[item_id],
                time_limit_per_item=time_limit_per_item,
                end_time=end_time,
                **ensemble_selection_kwargs,
            )
            for item_id in item_ids
        )
        self.weights_df = pd.DataFrame(weights_per_item, index=item_ids, columns=model_names)  # type: ignore
        self.average_weight = self.weights_df.mean(axis=0)

        # Drop models with zero average weight
        if (self.average_weight == 0).any():
            models_to_keep = self.average_weight[self.average_weight > 0].index
            self.weights_df = self.weights_df[models_to_keep]
            self.average_weight = self.average_weight[models_to_keep]

        weights_for_printing = {model: round(float(weight), 2) for model, weight in self.average_weight.items()}
        logger.info(f"\tAverage ensemble weights: {pprint.pformat(weights_for_printing, width=1000)}")

    def _split_predictions_per_item(
        self, predictions_per_window: dict[str, list[TimeSeriesDataFrame]]
    ) -> dict[str, dict[str, list[TimeSeriesDataFrame]]]:
        """Build a dictionary mapping item_id -> dict[model_name, list[TimeSeriesDataFrame]]."""
        item_ids = list(predictions_per_window.values())[0][0].item_ids

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
        time_limit_per_item: float | None = None,
        end_time: float | None = None,
        **ensemble_selection_kwargs,
    ) -> dict[str, float]:
        """Fit ensemble for a single item."""
        if end_time is not None:
            assert time_limit_per_item is not None
            time_left = end_time - time.time()
            time_limit_per_item = min(time_limit_per_item, time_left)
        return fit_time_series_ensemble_selection(
            data_per_window, predictions_per_window, time_limit=time_limit_per_item, **ensemble_selection_kwargs
        )

    def _predict(self, data: dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        assert all(model in data for model in self.weights_df.columns)
        item_ids = list(data.values())[0].item_ids
        unseen_item_ids = set(item_ids) - set(self.weights_df.index)
        if unseen_item_ids:
            logger.debug(f"Using average weights for {len(unseen_item_ids)} unseen items")
        weights = self.weights_df.reindex(item_ids).fillna(self.average_weight)

        result = None
        for model_name in self.weights_df.columns:
            model_pred = data[model_name]
            model_weights = weights[model_name].to_numpy().repeat(self.prediction_length)
            weighted_pred = model_pred.to_data_frame().multiply(model_weights, axis=0)
            result = weighted_pred if result is None else result + weighted_pred

        return TimeSeriesDataFrame(result)  # type: ignore

    def remap_base_models(self, model_refit_map: dict[str, str]) -> None:
        self.weights_df.rename(columns=model_refit_map, inplace=True)
