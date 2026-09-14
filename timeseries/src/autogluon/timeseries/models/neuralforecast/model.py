"""AutoGluon adapter for the complete, pinned NeuralForecast fork."""

import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

from ._catalog import NEURALFORECAST_MODELS, NEURALFORECAST_REVISION

logger = logging.getLogger(__name__)


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Backend hyperparameters must be JSON-compatible. Specify losses by name and kwargs, not objects.")


def _run_worker(action: str, request: Path, executable: str, timeout: float | None) -> None:
    """Enforce one budget over imports, training/calibration, and checkpoint writing."""
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise TimeLimitExceeded("No finite positive time budget remains for NeuralForecast.")
    script = Path(__file__).with_name("_worker.py")
    with (request.parent / f"{action}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [executable, str(script), action, str(request)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            # NF's Moirai backend may itself launch a child interpreter.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            if sys.exc_info()[0] is subprocess.TimeoutExpired:
                raise TimeLimitExceeded(f"NeuralForecast {action} exceeded its time budget.") from None
            raise
    if status:
        tail = (request.parent / f"{action}.log").read_text(encoding="utf-8", errors="replace")[-12000:]
        raise RuntimeError(f"NeuralForecast {action} failed (exit {status}). Backend log:\n{tail}")


class NeuralForecastModel(AbstractTimeSeriesModel):
    """Run a NeuralForecast model in a configurable Python environment.

    Flat hyperparameters are passed to the NF constructor, so AutoGluon HPO can
    search them. Reserved bridge options are documented in neuralforecast.md.
    The backend is imported only in the subprocess; existing AutoGluon models
    do not acquire a NeuralForecast dependency.
    """

    nf_model_name: str | None = None
    _supports_known_covariates = True
    _supports_past_covariates = True
    _supports_static_features = True
    ag_priority = 0

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        return {
            "model_name": self.nf_model_name or "NHITS",
            "python_executable": sys.executable,
            "validation_size": 0,
            "calibration_windows": 2,
            "calibration_step_size": None,
            "predict_time_limit": None,
            "hierarchy_item_ids": None,
        }

    def _configuration(self, num_cpus: int | None = None) -> dict:
        parameters = self.get_hyperparameters().copy()
        name = parameters.pop("model_name")
        if name not in NEURALFORECAST_MODELS or (self.nf_model_name and name != self.nf_model_name):
            raise ValueError(f"Invalid model_name={name!r} for {type(self).__name__}.")
        executable = os.path.expanduser(str(parameters.pop("python_executable")))
        executable = shutil.which(executable) or executable
        if not Path(executable).is_file():
            raise ValueError(f"python_executable does not exist: {executable}")
        # absolute(), unlike resolve(), preserves the venv's Python symlink.
        self._python_executable = str(Path(executable).absolute())
        validation_size = parameters.pop("validation_size")
        calibration_windows = parameters.pop("calibration_windows")
        step_size = parameters.pop("calibration_step_size")
        if type(validation_size) is not int or validation_size < 0:
            raise ValueError("validation_size must be a non-negative integer.")
        if type(calibration_windows) is not int or calibration_windows < 2:
            raise ValueError("calibration_windows must be an integer >= 2.")
        if step_size is not None and (type(step_size) is not int or step_size < 1):
            raise ValueError("calibration_step_size must be a positive integer or None.")
        self._predict_time_limit = parameters.pop("predict_time_limit")
        if self._predict_time_limit is not None and (not math.isfinite(self._predict_time_limit) or self._predict_time_limit <= 0):
            raise ValueError("predict_time_limit must be finite and positive or None.")
        hierarchy_ids = parameters.pop("hierarchy_item_ids")
        if hierarchy_ids is not None and name != "HINT":
            raise ValueError("hierarchy_item_ids is only used by NFHINT.")
        for key in ("target_scaler", "covariate_scaler", "covariate_regressor"):
            parameters.pop(key, None)  # Already applied by AbstractTimeSeriesModel.
        metadata = self.covariate_metadata
        features = {
            "past": list(metadata.past_covariates_real),
            "known": list(metadata.known_covariates_real),
            "static": list(metadata.static_features_real),
        }
        reserved = {"unique_id", "ds", "y"}
        if reserved.intersection(sum(features.values(), [])):
            raise ValueError("Rename covariates named unique_id, ds, or y before using the NeuralForecast bridge.")
        if metadata.covariates_cat or metadata.static_features_cat:
            logger.warning("%s uses numeric covariates only; categorical columns are not sent to NeuralForecast.", self.name)
        self._hierarchy_ids = hierarchy_ids
        return {
            "revision": NEURALFORECAST_REVISION, "expected_models": list(NEURALFORECAST_MODELS),
            "model_name": name, "model_parameters": parameters, "features": features,
            "horizon": self.prediction_length, "freq": self.freq, "quantiles": self.quantile_levels,
            "validation_size": validation_size, "calibration_windows": calibration_windows,
            "calibration_step_size": step_size, "num_cpus": max(1, int(num_cpus or 1)),
            "random_seed": int(parameters.get("random_seed", 1)),
        }

    def _write_inputs(self, directory: Path, data: TimeSeriesDataFrame, known_covariates=None) -> None:
        if not data.index.is_unique:
            raise ValueError("Duplicate (item_id, timestamp) rows are not supported.")
        features = self._backend_config["features"]
        temporal = features["past"] + features["known"]
        frame = data.to_data_frame()[[self.target] + temporal].copy()
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("NeuralForecast requires finite history; impute missing values within each training fold.")
        frame = frame.reset_index()
        frame.columns = ["unique_id", "ds", "y"] + temporal
        unknown = set(frame["unique_id"]) - set(self._item_ids)
        if unknown:
            raise ValueError(f"Refit this adapter before predicting new items: {unknown}.")
        mapping = {item: index for index, item in enumerate(self._item_ids)}
        frame["unique_id"] = frame["unique_id"].map(mapping)
        frame["ds"] = pd.to_datetime(frame["ds"]).astype("datetime64[ns]").astype("int64")
        frame.sort_values(["unique_id", "ds"]).to_csv(directory / "history.csv", index=False, float_format="%.17g")
        if features["static"]:
            if data.static_features is None:
                raise ValueError("Missing static_features.")
            current_items = list(data.item_ids)
            static = data.static_features.reindex(current_items)[features["static"]].copy()
            if not np.isfinite(static.to_numpy(dtype=float)).all():
                raise ValueError("Static covariates must be finite and present for every item.")
            static.insert(0, "unique_id", [mapping[item] for item in current_items])
            static.to_csv(directory / "static.csv", index=False, float_format="%.17g")
        if known_covariates is not None:
            if self.target in known_covariates.columns:
                raise ValueError("known_covariates must not contain the target.")
            expected = self.get_forecast_horizon_index(data)
            if not known_covariates.index.is_unique or len(expected.difference(known_covariates.index)):
                raise ValueError("known_covariates must cover every item and forecast timestamp without duplicates.")
            future = known_covariates.to_data_frame().reindex(expected)[features["known"]].copy()
            if not np.isfinite(future.to_numpy(dtype=float)).all():
                raise ValueError("Known future covariates must be finite over the full horizon.")
            future = future.reset_index()
            future.columns = ["unique_id", "ds"] + features["known"]
            future["unique_id"] = future["unique_id"].map(mapping)
            future["ds"] = pd.to_datetime(future["ds"]).astype("datetime64[ns]").astype("int64")
            future.to_csv(directory / "future.csv", index=False, float_format="%.17g")

    def _request(self, directory: Path, config: dict) -> Path:
        request = directory / "request.json"
        request.write_text(json.dumps(config, default=_json_value, allow_nan=False), encoding="utf-8")
        return request

    def _fit(self, train_data, val_data=None, time_limit=None, num_cpus=None, num_gpus=None, **kwargs) -> None:
        started = time.monotonic()
        self._backend_config = self._configuration(num_cpus=num_cpus)
        items = list(train_data.item_ids)
        if self._backend_config["model_name"] == "HINT":
            if self._hierarchy_ids is None:
                raise ValueError("NFHINT requires hierarchy_item_ids in the exact row order of S.")
            if len(self._hierarchy_ids) != len(items) or set(self._hierarchy_ids) != set(items):
                raise ValueError("hierarchy_item_ids must contain every training item exactly once.")
            items = list(self._hierarchy_ids)
        self._item_ids = items
        Path(self.path).mkdir(parents=True, exist_ok=True)
        artifacts = Path(tempfile.mkdtemp(prefix="neuralforecast-", dir=self.path))
        try:
            self._write_inputs(artifacts, train_data)
            request = self._request(artifacts, self._backend_config)
            remaining = None if time_limit is None else time_limit - (time.monotonic() - started)
            _run_worker("fit", request, self._python_executable, remaining)
            self.backend_info = json.loads((artifacts / "backend.json").read_text(encoding="utf-8"))
            self._artifact_path = str(artifacts)
            self._artifact_name = artifacts.name
            if self.backend_info["ignored_features"]:
                logger.warning("%s did not use these covariates: %s", self.name, self.backend_info["ignored_features"])
            # Checkpoints contain any calibration scores; raw input is not needed at inference.
            for name in ("history.csv", "static.csv", "request.json"):
                (artifacts / name).unlink(missing_ok=True)
        except BaseException:
            shutil.rmtree(artifacts)
            raise

    def _predict(self, data, known_covariates=None, **kwargs) -> TimeSeriesDataFrame:
        required = self.backend_info["features"]["futr_exog_list"]
        if required and known_covariates is None:
            raise ValueError(f"Missing known future covariates: {required}.")
        with tempfile.TemporaryDirectory(prefix="nf-predict-") as temporary:
            directory = Path(temporary)
            self._write_inputs(directory, data, known_covariates)
            config = self._backend_config | {"artifacts": self._artifact_path}
            request = self._request(directory, config)
            _run_worker("predict", request, self._python_executable, self._predict_time_limit)
            predictions = pd.read_csv(directory / "predictions.csv")
        indices = predictions.pop("unique_id")
        if not np.isfinite(indices).all() or ((indices < 0) | (indices >= len(self._item_ids))).any():
            raise ValueError("Backend returned unknown item indices.")
        if not np.equal(indices, indices.astype(int)).all():
            raise ValueError("Backend returned non-integer item indices.")
        predictions.index = pd.MultiIndex.from_arrays(
            [[self._item_ids[index] for index in indices.astype(int)], pd.to_datetime(predictions.pop("ds"), unit="ns")],
            names=["item_id", "timestamp"],
        )
        expected = self.get_forecast_horizon_index(data)
        if not predictions.index.is_unique or len(predictions) != len(expected) or len(expected.difference(predictions.index)):
            raise ValueError("Backend returned an incorrect item/timestamp forecast grid.")
        columns = ["mean"] + [str(q) for q in self.quantile_levels]
        predictions = predictions.reindex(expected)[columns]
        if not np.isfinite(predictions.to_numpy()).all():
            raise ValueError("Backend returned non-finite forecasts.")
        return TimeSeriesDataFrame(predictions)

    def save(self, path=None, verbose=True):
        destination = Path(path or self.path)
        if hasattr(self, "_artifact_path"):
            source = Path(self._artifact_path)
            target = destination / self._artifact_name
            if source.absolute() != target.absolute():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target)
        return super().save(path=path, verbose=verbose)

    def set_contexts(self, path_context):
        super().set_contexts(path_context)
        if hasattr(self, "_artifact_name"):
            self._artifact_path = str(Path(path_context) / self._artifact_name)

    def _more_tags(self):
        return super()._more_tags() | {"can_refit_full": True, "can_use_val_data": False}
