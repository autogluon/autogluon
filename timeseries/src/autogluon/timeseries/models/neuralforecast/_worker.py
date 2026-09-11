"""Isolated NeuralForecast process. This file deliberately does not import AutoGluon.

Inputs are locally generated CSV/JSON files. Checkpoints must be trusted: the
upstream NeuralForecast/PyTorch loader deserializes executable Python objects.
"""

import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def verify_backend(expected_revision, expected_models):
    """Accept the pinned VCS install or a clean editable checkout at that commit."""
    import neuralforecast
    from neuralforecast import models

    distribution = importlib.metadata.distribution("neuralforecast")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    revision = provenance.get("vcs_info", {}).get("commit_id")
    if revision != expected_revision:
        root = Path(neuralforecast.__file__).absolute().parent.parent
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=10
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True, timeout=10
            ).strip()
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("Install the pinned NeuralForecast fork in python_executable's environment.") from error
        if dirty:
            raise RuntimeError("The NeuralForecast checkout has modified tracked files; use a clean pinned checkout.")
    if revision != expected_revision:
        raise RuntimeError(f"Expected NeuralForecast {expected_revision}, found {revision}.")
    if set(models.__all__) != set(expected_models):
        raise RuntimeError("NeuralForecast's public model catalog differs from the pinned 66-model catalog.")
    return neuralforecast, models, {"revision": revision, "version": distribution.version, "python": sys.executable}


def read_frame(path, has_time=True):
    frame = pd.read_csv(path, keep_default_na=False)
    if has_time:
        frame["ds"] = pd.to_datetime(frame["ds"], unit="ns")
    return frame


def select_features(model_class, parameters, metadata):
    """Use numeric covariates only; explicit unsupported inputs are errors."""
    selected, ignored = {}, []
    for argument, capability, role in (
        ("hist_exog_list", "EXOGENOUS_HIST", "past"),
        ("futr_exog_list", "EXOGENOUS_FUTR", "known"),
        ("stat_exog_list", "EXOGENOUS_STAT", "static"),
    ):
        available = metadata[role]
        supported = bool(getattr(model_class, capability, False))
        values = parameters.get(argument, available if supported else [])
        if values is None:
            values = []
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{argument} must be a list of numeric column names.")
        if len(values) != len(set(values)) or set(values) - set(available):
            raise ValueError(f"{argument} contains duplicate, unavailable, or non-numeric columns: {values}.")
        if values and not supported:
            raise ValueError(f"{model_class.__name__} does not support {argument}.")
        # Older univariate constructors forward unknown kwargs to Lightning.
        if supported or argument in parameters:
            parameters[argument] = values
        selected[argument] = values
        ignored.extend(sorted(set(available) - set(values)))
    return selected, ignored


def make_loss(specification, quantiles):
    from neuralforecast.losses import pytorch

    if isinstance(specification, str):
        specification = {"name": specification}
    if not isinstance(specification, dict) or set(specification) - {"name", "kwargs"}:
        raise ValueError("loss must be a name or {'name': ..., 'kwargs': {...}}.")
    allowed = {"MAE", "MSE", "RMSE", "HuberLoss", "MQLoss", "HuberMQLoss", "DistributionLoss", "PMM", "GMM", "NBMM"}
    name = specification.get("name")
    if name not in allowed:
        raise ValueError(f"Unsupported loss specification {name!r}; supported names: {sorted(allowed)}.")
    kwargs = dict(specification.get("kwargs", {}))
    if name in {"MQLoss", "HuberMQLoss", "DistributionLoss", "PMM", "GMM", "NBMM"}:
        kwargs.setdefault("quantiles", quantiles)
    return getattr(pytorch, name)(**kwargs)


def check_panel(frame, expected_items):
    groups = list(frame.groupby("unique_id", sort=True))
    if not groups:
        raise ValueError("The item panel must not be empty.")
    if [key for key, _ in groups] != expected_items:
        raise ValueError("This model requires the same complete item panel as at fit time.")
    first = groups[0][1]["ds"].to_numpy()
    if any(not np.array_equal(group["ds"].to_numpy(), first) for _, group in groups[1:]):
        raise ValueError("Multivariate and hierarchical models require equal, synchronized timestamp grids.")


def configure_model(config, model_module, frame):
    name = config["model_name"]
    if name not in config["expected_models"]:
        raise ValueError(f"Unknown NeuralForecast model {name!r}.")
    parameters = dict(config["model_parameters"])
    hint = None
    if name == "HINT":
        base_name = parameters.pop("base_model", "NHITS")
        if base_name not in config["expected_models"] or base_name == "HINT":
            raise ValueError("HINT base_model must name a non-HINT NeuralForecast model.")
        matrix = np.asarray(parameters.pop("S", []), dtype=float)
        n_items = frame["unique_id"].nunique()
        if matrix.ndim != 2 or matrix.shape[0] != n_items or not 0 < matrix.shape[1] <= n_items:
            raise ValueError("HINT requires S with shape (number of items, number of bottom-level items).")
        if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != matrix.shape[1]:
            raise ValueError("HINT S must be finite and have full column rank.")
        if not np.allclose(matrix[-matrix.shape[1]:], np.eye(matrix.shape[1])):
            raise ValueError("HINT requires the bottom-level identity block at the end of S and hierarchy_item_ids.")
        hint = {"S": matrix.tolist(), "reconciliation": parameters.pop("reconciliation", "BottomUp")}
        if hint["reconciliation"] not in {"BottomUp", "MinTraceOLS", "MinTraceWLS", "Identity"}:
            raise ValueError("Unsupported HINT reconciliation method.")
        parameters.setdefault("loss", {"name": "DistributionLoss", "kwargs": {"distribution": "Normal"}})
        model_class = getattr(model_module, base_name)
    else:
        model_class = getattr(model_module, name)
    if "h" in parameters or "alias" in parameters:
        raise ValueError("prediction_length and the adapter control h and alias; do not override them.")
    if any(parameters.get(key) for key in ("cat_exog_list", "hist_categorical_list", "futr_categorical_list", "stat_categorical_list")):
        raise ValueError("This bridge accepts numeric covariates; encode categorical inputs before fitting.")
    selected, ignored = select_features(model_class, parameters, config["features"])
    multivariate = bool(getattr(model_class, "MULTIVARIATE", False))
    if multivariate:
        n_items = frame["unique_id"].nunique()
        if "n_series" in parameters and parameters["n_series"] != n_items:
            raise ValueError("n_series must match the complete training panel.")
        parameters["n_series"] = n_items
    parameters.setdefault("input_size", 32 * math.ceil(max(32, 2 * config["horizon"]) / 32))
    parameters.setdefault("accelerator", "cpu")
    parameters.setdefault("devices", 1)
    if parameters["devices"] != 1:
        raise ValueError("The isolated bridge currently supports a single device per model.")
    parameters.setdefault("random_seed", config.get("random_seed", 1))
    parameters.setdefault("logger", False)
    parameters.setdefault("enable_progress_bar", False)
    parameters.setdefault("enable_model_summary", False)
    for loss_key in ("loss", "valid_loss"):
        if loss_key in parameters and parameters[loss_key] is not None:
            parameters[loss_key] = make_loss(parameters[loss_key], config["quantiles"])
    model = model_class(h=config["horizon"], alias="forecast", **parameters)
    loss = model.loss
    distribution = bool(getattr(loss, "is_distribution_output", False))
    native = distribution or hasattr(loss, "quantiles") or getattr(loss, "outputsize_multiplier", 1) > 1
    if distribution and callable(getattr(loss, "update_quantile", None)):
        loss.update_quantile(config["quantiles"])
    if native and hasattr(loss, "quantiles"):
        grid = np.asarray(loss.quantiles.detach().cpu(), dtype=float)
        if any(not np.isclose(grid, q, atol=1e-6, rtol=0).any() for q in config["quantiles"]):
            raise ValueError("The native loss quantile grid must include every predictor quantile; configure loss accordingly.")
    if hint is not None and not distribution:
        raise ValueError("HINT requires a distribution-output loss on its base model.")
    return model, selected, ignored, multivariate, native, hint


def save_checkpoint(nf, path, loss_hparams):
    """Keep forward-pass caches out of constructor metadata, preserving state_dict.

    The pinned NF DistributionLoss caches a non-leaf distr_mean tensor during
    training. NF.save deep-copies hparams. Use its pre-fit constructor loss
    values for that metadata only; learned model/loss state is saved unchanged.
    """
    model = nf.models[0]
    originals = {key: model.hparams[key] for key in loss_hparams}
    model.hparams.update(copy.deepcopy(loss_hparams))
    try:
        nf.save(path=str(path), overwrite=False, save_dataset=False)
    finally:
        model.hparams.update(originals)


def fit(config, root, neuralforecast, model_module, provenance):
    frame = read_frame(root / "history.csv")
    model, selected, ignored, multivariate, native, hint = configure_model(config, model_module, frame)
    temporal = selected["hist_exog_list"] + selected["futr_exog_list"]
    frame = frame[["unique_id", "ds", "y"] + temporal]
    statics = selected["stat_exog_list"]
    static = read_frame(root / "static.csv", has_time=False)[["unique_id"] + statics] if statics else None
    items = sorted(frame["unique_id"].unique().tolist())
    if multivariate or hint is not None:
        check_panel(frame, items)
    validation_size = config["validation_size"]
    if validation_size and validation_size < config["horizon"]:
        raise ValueError("validation_size must be zero or at least prediction_length.")
    if getattr(model, "early_stop_patience_steps", -1) > 0 and validation_size == 0:
        raise ValueError("Set validation_size >= prediction_length to use NeuralForecast early stopping.")
    intervals = None
    if not native:
        from neuralforecast.utils import PredictionIntervals

        intervals = PredictionIntervals(
            n_windows=config["calibration_windows"],
            method="conformal_distribution",
            step_size=config["calibration_step_size"] or config["horizon"],
        )
        # Ensure that even the earliest calibration window has actual context.
        calibration_span = config["horizon"] + intervals.step_size * (intervals.n_windows - 1)
        required = (1 if getattr(model, "start_padding_enabled", False) else model.input_size)
        required += calibration_span + validation_size + config["horizon"]
        if frame.groupby("unique_id").size().min() < required:
            raise ValueError(f"Conformal calibration requires at least {required} observations per item for this configuration.")
    loss_hparams = copy.deepcopy({key: model.hparams[key] for key in ("loss", "valid_loss") if key in model.hparams})
    nf = neuralforecast.NeuralForecast(models=[model], freq=config["freq"])
    nf.fit(df=frame, static_df=static, val_size=validation_size, prediction_intervals=intervals)
    save_checkpoint(nf, root / "checkpoint", loss_hparams)
    info = provenance | {
        "model_name": config["model_name"], "features": selected, "ignored_features": ignored,
        "multivariate": multivariate, "hint": hint, "items": items,
        "quantile_method": "native" if native else "neuralforecast_conformal_distribution",
        "loss": type(model.loss).__name__,
        "training_end": {str(key): str(group["ds"].max()) for key, group in frame.groupby("unique_id")},
    }
    (root / "backend.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def normalize_forecasts(raw, model, quantiles):
    """Use explicit NF labels; never interpolate or fabricate missing quantiles."""
    name = repr(model)
    result = raw[["unique_id", "ds"]].copy()
    native_names = {}
    if hasattr(model.loss, "quantiles"):
        grid = np.asarray(model.loss.quantiles.detach().cpu(), dtype=float)
        suffixes = list(model.loss.output_names)
        if getattr(model.loss, "is_distribution_output", False):
            suffixes = suffixes[1:1 + len(grid)]
        if len(suffixes) == len(grid):
            native_names = dict(zip(grid, [name + suffix for suffix in suffixes]))
    for quantile in quantiles:
        candidates = [f"{name}_ql{quantile}", f"{name}-ql{quantile}"]
        candidates.extend(column for q, column in native_names.items() if np.isclose(q, quantile, atol=1e-6, rtol=0))
        column = next((column for column in candidates if column in raw), None)
        if column is None:
            raise ValueError(f"Backend did not return quantile {quantile}; returned columns: {list(raw)}.")
        result[str(quantile)] = raw[column].to_numpy()
    # AutoGluon's `mean` is its point-output slot; quantile-only backends supply P50.
    result["mean"] = raw[name].to_numpy() if name in raw else result["0.5"].to_numpy()
    if not np.isfinite(result[["mean"] + [str(q) for q in quantiles]].to_numpy()).all():
        raise ValueError("NeuralForecast returned non-finite predictions.")
    result["ds"] = pd.to_datetime(result["ds"]).astype("datetime64[ns]").astype("int64")
    return result


def predict(config, root, neuralforecast, model_module):
    artifacts = Path(config["artifacts"])
    info = json.loads((artifacts / "backend.json").read_text(encoding="utf-8"))
    frame = read_frame(root / "history.csv")
    selected = info["features"]
    frame = frame[["unique_id", "ds", "y"] + selected["hist_exog_list"] + selected["futr_exog_list"]]
    if info["multivariate"] or info["hint"] is not None:
        check_panel(frame, info["items"])
    if info["quantile_method"] != "native" and sorted(frame["unique_id"].unique()) != info["items"]:
        raise ValueError("Conformal intervals require the same item set as calibration; refit for a different panel.")
    statics = selected["stat_exog_list"]
    static = read_frame(root / "static.csv", has_time=False)[["unique_id"] + statics] if statics else None
    future = read_frame(root / "future.csv")[["unique_id", "ds"] + selected["futr_exog_list"]] if selected["futr_exog_list"] else None
    nf = neuralforecast.NeuralForecast.load(path=str(artifacts / "checkpoint"))
    model = nf.models[0]
    quantiles = config["quantiles"]
    if info["hint"] is not None:
        # NF's HINT.save stores only the base model. Reconstruct reconciliation explicitly.
        model.loss.update_quantile(quantiles)
        model = model_module.HINT(
            h=config["horizon"], S=np.asarray(info["hint"]["S"]), model=model,
            reconciliation=info["hint"]["reconciliation"], alias="forecast",
        )
        for attribute in ("hist_exog_list", "futr_exog_list", "stat_exog_list"):
            setattr(model, attribute, getattr(model.model, attribute))
        nf.models = [model]
        # Passing quantiles through HINT would replace its internal bootstrap sample grid.
        raw = nf.predict(df=frame, static_df=static, futr_df=future)
    else:
        # A fixed MQLoss grid (including a single median) is already in the checkpoint.
        predict_kwargs = {"quantiles": quantiles} if info["quantile_method"] != "native" or getattr(model.loss, "is_distribution_output", False) else {}
        raw = nf.predict(df=frame, static_df=static, futr_df=future, **predict_kwargs)
    normalize_forecasts(raw, model, quantiles).to_csv(root / "predictions.csv", index=False, float_format="%.17g")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit", "predict"))
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    config = json.loads(args.request.read_text(encoding="utf-8"))
    root = args.request.parent
    nf, models, provenance = verify_backend(config["revision"], config["expected_models"])
    import torch

    torch.set_num_threads(max(1, config["num_cpus"]))
    np.random.seed(config["random_seed"])
    if args.action == "fit":
        fit(config, root, nf, models, provenance)
    else:
        predict(config, root, nf, models)


if __name__ == "__main__":
    main()
