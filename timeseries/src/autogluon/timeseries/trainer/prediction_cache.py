import hashlib
import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from autogluon.common.utils.utils import hash_pandas_df
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.timeseries import TimeSeriesDataFrame

logger = logging.getLogger(__name__)


def compute_dataset_hash(data: TimeSeriesDataFrame, known_covariates: TimeSeriesDataFrame | None = None) -> str:
    """Compute a unique string that identifies the time series dataset."""
    static_hash = hash_pandas_df(data.static_features) if data.static_features is not None else "no_static"
    covar_hash = hash_pandas_df(known_covariates) if known_covariates is not None else "no_covar"
    combined_hash = hash_pandas_df(data) + covar_hash + static_hash
    return hashlib.sha256(combined_hash.encode("utf-8")).hexdigest()


def compute_model_hash(full_model_path: Path, root_path: Path) -> str:
    """
    Compute a hash based on the model's relative path and modification time.
    """
    hash_sha256 = hashlib.sha256()
    try:
        path_key = str(full_model_path.relative_to(root_path))
    except ValueError:
        path_key = str(full_model_path)
    hash_sha256.update(path_key.encode("utf-8"))
    try:
        stat = full_model_path.stat()
        hash_sha256.update(f"{stat.st_size}-{stat.st_mtime}".encode("utf-8"))
    except OSError:
        logger.warning(
            f"Cannot compute model hash for '{full_model_path}' as it does not exist or is inaccessible. "
            "Predictions for this model will not be cached correctly."
        )
        return "missing"
    return hash_sha256.hexdigest()


class PredictionCache(ABC):
    """An abstract key-value store for time series predictions.

    Maps (dataset, model) pairs to their corresponding predictions and inference times. The cache key
    consists of a hash of the dataset and a model hash generated from the model's file path and
    modification time. This prevents cache collisions even if different models share the same name.

    The `get` and `put` methods are designed to operate on batches of models for a given dataset.
    """

    def __init__(self, root_path: str) -> None:
        self.root_path = Path(root_path)

    @abstractmethod
    def get(
        self, data: TimeSeriesDataFrame, known_covariates: TimeSeriesDataFrame | None, model_path_map: dict[str, str]
    ) -> tuple[dict[str, TimeSeriesDataFrame | None], dict[str, float]]:
        pass

    @abstractmethod
    def put(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None,
        model_pred_dict: dict[str, TimeSeriesDataFrame | None],
        pred_time_dict: dict[str, float],
        model_path_map: dict[str, str],
    ) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class NoOpPredictionCache(PredictionCache):
    """A dummy (no-op) prediction cache."""

    def get(
        self, data: TimeSeriesDataFrame, known_covariates: TimeSeriesDataFrame | None, model_path_map: dict[str, str]
    ) -> tuple[dict[str, TimeSeriesDataFrame | None], dict[str, float]]:
        return {}, {}

    def put(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None,
        model_pred_dict: dict[str, TimeSeriesDataFrame | None],
        pred_time_dict: dict[str, float],
        model_path_map: dict[str, str],
    ) -> None:
        pass

    def clear(self) -> None:
        pass


class FileBasedPredictionCache(PredictionCache):
    """A file-backed cache of model predictions using atomic writes."""

    _CACHE_DIR_NAME = "predictions_cache"

    def __init__(self, root_path: str) -> None:
        super().__init__(root_path)
        self.cache_dir = self.root_path / self._CACHE_DIR_NAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_file(self, dataset_hash: str, model_name: str, rel_model_path: str) -> Path:
        full_model_path = self.root_path / rel_model_path
        model_hash = compute_model_hash(full_model_path, self.root_path)
        dataset_dir = self.cache_dir / dataset_hash
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir / f"{model_name}__{model_hash}.pkl"

    def get(
        self, data: TimeSeriesDataFrame, known_covariates: TimeSeriesDataFrame | None, model_path_map: dict[str, str]
    ) -> tuple[dict[str, TimeSeriesDataFrame | None], dict[str, float]]:
        dataset_hash = compute_dataset_hash(data, known_covariates)
        cached_preds: dict[str, TimeSeriesDataFrame | None] = {}
        cached_times: dict[str, float] = {}

        for model_name, model_path in model_path_map.items():
            cache_file = self._get_cache_file(dataset_hash, model_name, model_path)
            if cache_file.exists():
                try:
                    content = load_pkl.load(str(cache_file))
                    cached_preds[model_name] = content["pred"]
                    cached_times[model_name] = content["time"]
                except Exception:
                    logger.warning(f"Failed to load cache for {model_name}, skipping.")
                    continue

        return cached_preds, cached_times

    def put(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None,
        model_pred_dict: dict[str, TimeSeriesDataFrame | None],
        pred_time_dict: dict[str, float],
        model_path_map: dict[str, str],
    ) -> None:
        dataset_hash = compute_dataset_hash(data, known_covariates)

        for model_name, pred in model_pred_dict.items():
            if pred is None or model_name not in model_path_map:
                continue

            cache_file = self._get_cache_file(dataset_hash, model_name, model_path_map[model_name])
            temp_file = cache_file.with_suffix(f".tmp.{uuid.uuid4().hex}")
            content = {"pred": pred, "time": pred_time_dict.get(model_name, 0.0)}

            try:
                save_pkl.save(path=str(temp_file), object=content, verbose=False)
                os.replace(str(temp_file), str(cache_file))

            except Exception as e:
                logger.warning(f"Could not cache prediction for {model_name}: {e}")
                if temp_file.exists():
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass

    def clear(self) -> None:
        shutil.rmtree(self.cache_dir, ignore_errors=True)


def get_prediction_cache(use_cache: bool, root_path: str) -> PredictionCache:
    return FileBasedPredictionCache(root_path) if use_cache else NoOpPredictionCache(root_path)
