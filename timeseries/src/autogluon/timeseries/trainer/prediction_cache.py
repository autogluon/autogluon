import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from autogluon.common.utils.utils import hash_pandas_df
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.timeseries import TimeSeriesDataFrame

logger = logging.getLogger(__name__)


class PredictionCache(ABC):
    """A prediction cache is an abstract key-value store for time series predictions. The storage is keyed by
    (data, known_covariates) pairs and stores (model_pred_dict, pred_time_dict) pair values. In this stored pair,
    (model_pred_dict, pred_time_dict), both dictionaries are keyed by model names.
    """

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)

    @abstractmethod
    def get(
        self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame]
    ) -> tuple[dict[str, Optional[TimeSeriesDataFrame]], dict[str, float]]:
        pass

    @abstractmethod
    def put(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame],
        model_pred_dict: dict[str, Optional[TimeSeriesDataFrame]],
        pred_time_dict: dict[str, float],
    ) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


def get_prediction_cache(use_cache: bool, root_path: str) -> PredictionCache:
    if use_cache:
        return FileBasedPredictionCache(root_path=root_path)
    else:
        return NoOpPredictionCache(root_path=root_path)


def compute_dataset_hash(data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame] = None) -> str:
    """Compute a unique string that identifies the time series dataset."""
    combined_hash = hash_pandas_df(data) + hash_pandas_df(known_covariates) + hash_pandas_df(data.static_features)
    return combined_hash


class NoOpPredictionCache(PredictionCache):
    """A dummy (no-op) prediction cache."""

    def get(
        self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame]
    ) -> tuple[dict[str, Optional[TimeSeriesDataFrame]], dict[str, float]]:
        return {}, {}

    def put(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame],
        model_pred_dict: dict[str, Optional[TimeSeriesDataFrame]],
        pred_time_dict: dict[str, float],
    ) -> None:
        pass

    def clear(self) -> None:
        pass


class FileBasedPredictionCache(PredictionCache):
    """A file-backed cache of model predictions."""

    _cached_predictions_filename = "cached_predictions.pkl"

    @property
    def path(self) -> Path:
        return Path(self.root_path) / self._cached_predictions_filename

    def get(
        self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame]
    ) -> tuple[dict[str, Optional[TimeSeriesDataFrame]], dict[str, float]]:
        dataset_hash = compute_dataset_hash(data, known_covariates)
        return self._get_cached_pred_dicts(dataset_hash)

    def put(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame],
        model_pred_dict: dict[str, Optional[TimeSeriesDataFrame]],
        pred_time_dict: dict[str, float],
    ) -> None:
        dataset_hash = compute_dataset_hash(data, known_covariates)
        self._save_cached_pred_dicts(dataset_hash, model_pred_dict, pred_time_dict)

    def clear(self) -> None:
        if self.path.exists():
            logger.debug(f"Removing existing cached predictions file {self.path}")
            self.path.unlink()

    def _load_cached_predictions(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self.path.exists():
            try:
                cached_predictions = load_pkl.load(str(self.path))
            except Exception:
                cached_predictions = {}
        else:
            cached_predictions = {}
        return cached_predictions

    def _get_cached_pred_dicts(
        self, dataset_hash: str
    ) -> tuple[dict[str, Optional[TimeSeriesDataFrame]], dict[str, float]]:
        """Load cached predictions for given dataset_hash from disk, if possible.

        If loading fails for any reason, empty dicts are returned.
        """
        cached_predictions = self._load_cached_predictions()
        if dataset_hash in cached_predictions:
            try:
                model_pred_dict = cached_predictions[dataset_hash]["model_pred_dict"]
                pred_time_dict = cached_predictions[dataset_hash]["pred_time_dict"]
                assert model_pred_dict.keys() == pred_time_dict.keys()
                return model_pred_dict, pred_time_dict
            except Exception:
                logger.warning("Cached predictions are corrupted. Predictions will be made from scratch.")
        return {}, {}

    def _save_cached_pred_dicts(
        self,
        dataset_hash: str,
        model_pred_dict: dict[str, Optional[TimeSeriesDataFrame]],
        pred_time_dict: dict[str, float],
    ) -> None:
        cached_predictions = self._load_cached_predictions()
        # Do not save results for models that failed
        cached_predictions[dataset_hash] = {
            "model_pred_dict": {k: v for k, v in model_pred_dict.items() if v is not None},
            "pred_time_dict": {k: v for k, v in pred_time_dict.items() if v is not None},
        }
        save_pkl.save(str(self.path), object=cached_predictions)
        logger.debug(f"Cached predictions saved to {self.path}")
