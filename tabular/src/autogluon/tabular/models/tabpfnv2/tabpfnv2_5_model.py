from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_HAS_LOGGED_TABPFN_LICENSE: bool = False
_HAS_LOGGED_TABPFN_NONCOMMERICAL: bool = False
_HAS_LOGGED_TABPFN_CPU_WARNING: bool = False

_INFERENCE_DTYPE_BYTES = 4
_NARROWED_INFERENCE_DTYPES = {
    np.dtype(np.float64): np.float32,
    np.dtype(np.int64): np.int32,
}
_NARROWED_RAW_TARGET_DTYPES = {np.dtype(np.int64): np.int32}
"""Narrowing allowed for a target that is still preprocessed after being stored."""


def _narrow_array(obj: object, name: str, narrowed_dtypes: dict) -> None:
    """Replace `obj.name` with a narrower view of itself, if one is allowed."""
    array = getattr(obj, name, None)
    narrower = narrowed_dtypes.get(getattr(array, "dtype", None))
    if narrower is not None:
        setattr(obj, name, array.astype(narrower, copy=False))


class TabPFNModel(AbstractTorchModel):
    """TabPFN-2.5 is a tabular foundation model that is developed and maintained by PriorLabs: https://priorlabs.ai/.

    This class is an abstract template for various TabPFN versions as subclasses.

    Paper: Accurate predictions on small data with a tabular foundation model
    Authors: Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister & Frank Hutter
    Codebase: https://github.com/PriorLabs/TabPFN
    License: https://github.com/PriorLabs/TabPFN/blob/main/LICENSE

    .. versionadded:: 1.5.0
    """

    gpu_strongly_recommended: bool = True  # in-context inference is 12-63x slower on CPU
    ag_key = "NOTSET"
    ag_name = "NOTSET"
    ag_priority = 40
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression", "quantile"]
    fixed_random_state: int | None = None
    """If not None, this fixes the random state to a static value to avoid that the
    validation score is misleading for the refit model."""

    custom_model_dir: str | None = None
    """Directory containing the model checkpoints. Overridable per fit via the
    ``custom_model_dir`` hyperparameter."""
    license_noncommercial: ClassVar[bool] = False
    """Whether this version's default checkpoints are released under Prior Labs'
    noncommercial license; controls the license notice logged at fit time."""
    max_gpus: int = 8
    """Maximum number of GPUs requested by default; TabPFN spreads inference over a
    device list when more than one GPU is assigned."""
    default_classification_model: str | None = "NOTSET"
    default_regression_model: str | None = "NOTSET"
    default_model_map: dict | None = None
    max_batch_size_min: int = 1_000
    """Lower bound of the ``"auto"`` ``ag.max_batch_size`` (prediction chunking)
    resolution; also the prediction-batch floor assumed by memory estimates.

    TabPFN-2.5/2.6 re-process the joint train + prediction-batch sequence per
    chunk, so peak VRAM scales with the batch size and chunks sized near the
    training set already amortize the context cost. A low floor keeps small
    datasets from paying 100k-row prediction-batch memory (and from being
    skipped by memory estimates assuming it). Versions whose training context
    is reused across chunks (TabPFN-3) override this with a high floor, since
    for them small chunks multiply predict time while saving little memory."""

    _default_auxiliary_params_extra = {
        "max_rows": 100_000,
        "max_features": 2000,
        "max_classes": 10,
        # "auto" resolves at fit time to min(1M, max(100k, n_train));
        # None disables prediction chunking.
        "max_batch_size": "auto",
        "model_telemetry": False,
    }
    minimum_num_gpus = 1
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,  # Better to refit the model for faster inference and similar quality as the bag.
    }
    """Set fold_fitting_strategy to sequential_local, as parallel folding crashes if model weights aren't pre-downloaded."""
    default_resources_physical_cores_only = True
    default_num_gpus = max_gpus

    tabpfn_fit_file_name = "fitted_estimator.tabpfn_fit"
    """Sidecar holding the fitted state, written next to `model_file_name`."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cat_indices = None
        # `ag.max_batch_size="auto"` resolved against the training size during `_fit`.
        self._max_batch_size_resolved: int | None = None

    def _default_model_map(self) -> dict[str, str | None]:
        fallback = {
            "binary": self.default_classification_model,
            "multiclass": self.default_classification_model,
            "regression": self.default_regression_model,
            "quantile": self.default_regression_model,
        }
        default_model_map = dict(self.default_model_map) if self.default_model_map is not None else {}
        return {k: default_model_map.get(k, v) for k, v in fallback.items()}

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> pd.DataFrame:
        """Record which columns are categorical; leave their values as `category` dtype.

        TabPFN casts every column named in `categorical_features_indices` back to `category`
        itself (`tabpfn.preprocessing.clean.fix_dtypes`) and ordinal-encodes from there, so it
        needs the indices but not encoded values. Label-encoding first is worse than redundant:
        `.cat.codes` maps missing values to -1, and the cast back to `category` then makes -1 an
        ordinary level, so TabPFN sees no missing values in those columns and its missing-value
        handling never runs. Passing the dtype through keeps the missingness and is cheaper
        (`category` stores one code byte per row plus a level table, against float64's eight).

        The indices are still needed. TabPFN infers a column's modality from its dtype, and a
        `category` column whose levels are integers reads as numeric (`_is_numeric_pandas_series`
        coerces it), so without them anything above `min_unique_for_numerical` levels is treated
        as NUMERICAL rather than CATEGORICAL. AutoGluon's default `CategoryFeatureGenerator`
        minimizes memory by re-coding levels to integers, which makes that the common case.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            categorical_features = X.select_dtypes(include=["category"]).columns.tolist()
            self._cat_indices = [X.columns.get_loc(column) for column in categorical_features]

        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        time_limit: float | None = None,
        verbosity: int = 2,
        **kwargs,
    ):
        if not self.params_aux.get("model_telemetry", False):
            self.disable_tabpfn_telemetry()

        # "auto" prediction chunking resolves against the training size: chunks
        # re-attend the full training context, so chunks smaller than the training set
        # multiply predict time at large n_train while saving little memory. Bounded
        # to [max_batch_size_min, 1M]. None disables chunking entirely. The resolved
        # value is fit state (read via `_get_max_batch_size`), not a params_aux mutation.
        if self.aux_params.max_batch_size == "auto":
            self._max_batch_size_resolved = min(1_000_000, max(self.max_batch_size_min, len(X)))

        from tabpfn import TabPFNClassifier, TabPFNRegressor

        is_classification = self.problem_type in ["binary", "multiclass"]

        model_base = TabPFNClassifier if is_classification else TabPFNRegressor

        self._resolve_fit_device(num_gpus=num_gpus)  # CPU-fallback warning + CUDA availability check
        device = self._get_tabpfn_device(num_gpus=num_gpus)

        if verbosity >= 2:
            # logs "Built with PriorLabs-TabPFN"
            self._log_license(device=device)
            self._log_cpu_warning(device=device)

        X = self.preprocess(X, y=y, is_train=True)

        hps = self._get_model_params()
        custom_model_dir = hps.pop("custom_model_dir", self.custom_model_dir)
        hps["device"] = device
        hps["n_jobs"] = num_cpus  # FIXME: remove this, it doesn't do anything, use n_preprocessing_jobs??
        hps["categorical_features_indices"] = self._cat_indices

        # Resolve preprocessing
        if "preprocessing/scaling" in hps:
            hps["inference_config/PREPROCESS_TRANSFORMS"] = [
                {
                    "name": scaler,
                    "global_transformer_name": hps.pop("preprocessing/global", None),
                    "categorical_name": hps.pop("preprocessing/categoricals", "numeric"),
                    "append_original": hps.pop("preprocessing/append_original", True),
                }
                for scaler in hps["preprocessing/scaling"]
            ]
        for k in [
            "preprocessing/scaling",
            "preprocessing/categoricals",
            "preprocessing/append_original",
            "preprocessing/global",
        ]:
            hps.pop(k, None)

        # Remove task specific HPs
        if is_classification:
            hps.pop("inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS", None)
        else:
            hps.pop("balance_probabilities", None)

        if self.fixed_random_state is not None:
            hps[self.seed_name] = self.fixed_random_state

        model_path = self._resolve_model_path(
            hps=hps, is_classification=is_classification, custom_model_dir=custom_model_dir
        )
        if model_path is not None:
            # str, not Path: `save_fitted_tabpfn_model` writes the estimator's params
            # to JSON, which has no encoder for Path.
            hps["model_path"] = str(model_path)

        # Resolve inference_config
        inference_config = {
            _k: v for k, v in hps.items() if k.startswith("inference_config/") and (_k := k.split("/")[-1])
        }
        if inference_config:
            hps["inference_config"] = inference_config
        for k in list(hps.keys()):
            if k.startswith("inference_config/"):
                del hps[k]

        # Model and fit
        self.model = model_base(**hps)
        self.model = self.model.fit(
            X=X,
            y=y,
        )
        self._narrow_inference_context()

    def _narrow_inference_context(self):
        """Store the in-context training set at the precision inference uses.

        TabPFN keeps the training data it attends over on the inference engine and
        converts it with ``torch.as_tensor(..., dtype=torch.float32)`` at predict time,
        so the float64 arrays its preprocessing produces are never read at full width.
        Narrowing them halves what the fitted model holds and what its pickle writes.

        Which arrays can narrow depends on the fit mode. With preprocessing cached, the
        stored arrays are already preprocessed and feed that conversion directly, so
        both narrow losslessly -- once per ensemble member, which is where the size
        comes from. ``fit_mode="low_memory"`` instead keeps the raw training set and
        re-runs the preprocessing on every predict: the features still narrow
        losslessly, but a float target does not, because its transforms would then be
        computed at the narrower precision.

        Skipped when ``inference_precision`` forces a wider dtype, the one case where
        the extra precision reaches the model.
        """
        forced_dtype = getattr(self.model, "forced_inference_dtype_", None)
        if forced_dtype is not None and forced_dtype.itemsize > _INFERENCE_DTYPE_BYTES:
            return
        executor = getattr(self.model, "executor_", None)
        members = getattr(executor, "ensemble_members", None)
        if members is not None:
            for member in members:
                _narrow_array(member, "X_train", _NARROWED_INFERENCE_DTYPES)
                _narrow_array(member, "y_train", _NARROWED_INFERENCE_DTYPES)
        else:
            _narrow_array(executor, "X_train", _NARROWED_INFERENCE_DTYPES)
            _narrow_array(executor, "y_train", _NARROWED_RAW_TARGET_DTYPES)

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        """Save the fitted estimator beside the pickle, without the foundation weights.

        The weights are the same for every model of a given TabPFN version, so
        pickling them into each fitted model writes a copy of the checkpoint per
        model. ``save_fitted_tabpfn_model`` persists only the fitted state;
        :meth:`load` reloads the weights from ``model_path``.

        This makes a saved model depend on its checkpoint still being resolvable
        when it is loaded, rather than being self-contained.
        """
        if not self.is_fit():
            return super().save(path=path, verbose=verbose)

        from tabpfn import save_fitted_tabpfn_model

        path = path if path is not None else self.path
        os.makedirs(path, exist_ok=True)
        save_fitted_tabpfn_model(self.model, os.path.join(path, self.tabpfn_fit_file_name))

        # Detaching before `super().save()` is what keeps the weights out of the
        # pickle, and skips the torch device round-trip it would otherwise do.
        estimator = self.model
        self.model = None
        try:
            return super().save(path=path, verbose=verbose)
        finally:
            self.model = estimator

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """Load the pickle, then reattach the fitted estimator and its weights."""
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        fit_path = os.path.join(path, cls.tabpfn_fit_file_name)
        if model.model is None and os.path.exists(fit_path):
            from tabpfn import load_fitted_tabpfn_model

            model.model = load_fitted_tabpfn_model(fit_path, device=model.suggest_device_infer(verbose=verbose))
        return model

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        if not self.params_aux.get("model_telemetry", False):
            self.disable_tabpfn_telemetry()

        if self.problem_type == "quantile":
            y_pred = self.model.predict(
                X,
                output_type="quantiles",
                quantiles=self.quantile_levels,
            )
            return np.column_stack(y_pred)

        return super()._predict_proba(X=X, kwargs=kwargs)

    @staticmethod
    def _get_tabpfn_device(num_gpus: int) -> str | list[str]:
        """TabPFN device argument: a device list when fitting with multiple GPUs.

        ``num_gpus`` can exceed the CUDA-visible device count (resource grants may be
        counted via NVML, which ignores ``CUDA_VISIBLE_DEVICES``), so the device list is
        clamped to the devices torch can actually address.
        """
        if num_gpus <= 0:
            return "cpu"
        import torch

        num_devices = min(int(num_gpus), max(1, torch.cuda.device_count()))
        if num_devices == 1:
            return "cuda"
        return [f"cuda:{i}" for i in range(num_devices)]

    def _set_default_params(self):
        default_params = {
            "ignore_pretraining_limits": True,  # to ignore warnings and size limits
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _ag_params(self) -> set[str]:
        # model_telemetry: whether the tabpfn library's telemetry is left enabled
        # during fit/predict (disabled by default).
        return {"model_telemetry"}

    def _get_max_batch_size(self) -> int | None:
        max_batch_size = self.aux_params.max_batch_size
        if max_batch_size == "auto":
            return self._max_batch_size_resolved
        return max_batch_size

    def get_device(self) -> str:
        return self.model.devices_[0].type

    def _set_device(self, device: str):
        self.model.to(device)
        self._sync_inner_checkpoints_to_engine_devices(device=device)

    def _sync_inner_checkpoints_to_engine_devices(self, device: str) -> None:
        """Point `models_` back at the checkpoints the inference engine just moved.

        `tabpfn.base.estimator_to_device` (which backs `estimator.to()`) updates the estimator's
        device bookkeeping and moves the inference engine's per-device model caches, but leaves
        `models_` -- the loaded checkpoints -- referencing whatever it referenced before.

        With a single device that is harmless, because `models_[i]` *is* the engine's only cached
        copy and so gets moved with it. With several devices the engine keeps one copy per device,
        `models_[i]` is no longer the copy that survives a move, and two problems follow. Both are
        visible in the pickled artifact, since `models_` is pickled along with the engine:

        * A GPU fit leaves `models_` on CUDA, so the artifact holds CUDA-tagged storages even
          though `save` moved the model to CPU to keep it portable, and loading it on a CPU-only
          machine raises "Attempting to deserialize object on a CUDA device".
        * The weights are stored twice -- once via `models_` and once via the engine cache --
          doubling both the artifact and the memory a loaded model occupies.

        Re-pointing `models_` at the engine's copies fixes both: the weights follow the device the
        engine moved them to, and they exist exactly once. tabpfn documents that references
        obtained from a cache are invalidated by `.to()`, so re-reading them afterwards is the
        supported order. Falls back to moving `models_` directly if the engine does not expose the
        caches in the shape we expect.
        """
        models = getattr(self.model, "models_", None) or []
        caches = getattr(getattr(self.model, "executor_", None), "model_caches", None) or []
        if len(caches) == len(models) and all(cache.get_devices() for cache in caches):
            self.model.models_ = [cache.get(cache.get_devices()[0]) for cache in caches]
        else:
            for inner_model in models:
                inner_model.to(device)

    @classmethod
    def _n_test_for_memory_estimate(cls, *, n_train: int, hyperparameters: dict | None) -> int:
        """Proxy for the prediction batch size in memory estimates.

        These estimates bound *fit* memory, and the predictions made during a fit are
        on held-out folds of the training data, so the batch is bounded by the
        training size as well as by ``ag.max_batch_size`` chunking — hence the
        minimum of the two. (Predicting on a test set far larger than the training
        data can exceed this; that is inference-time memory, which AutoGluon's
        fit-time memory checks do not cover.)
        """
        max_batch_size = (hyperparameters or {}).get("ag.max_batch_size", "auto")
        if max_batch_size is None or max_batch_size == "auto":
            # "auto" resolves to at least max_batch_size_min at fit time; explicit
            # None (chunking disabled) has no bound, so use the same proxy.
            max_batch_size = max(cls.max_batch_size_min, n_train)
        return min(int(max_batch_size), n_train)

    @classmethod
    def disable_tabpfn_telemetry(cls):
        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

    def _resolve_model_path(
        self, hps: dict, is_classification: bool, custom_model_dir: str | None = None
    ) -> Path | None:
        from tabpfn.model_loading import resolve_model_path

        if custom_model_dir is None:
            custom_model_dir = self.custom_model_dir
        if custom_model_dir is not None:
            model_dir = Path(custom_model_dir)
        else:
            _, model_dir, _, _ = resolve_model_path(
                model_path=None,
                which="classifier" if is_classification else "regressor",
            )
            model_dir = model_dir[0]

        default_model_map = self._default_model_map()

        zip_model_path = hps.pop(
            "zip_model_path",
            default_model_map,
        )

        if isinstance(zip_model_path, (list, tuple)):
            if len(zip_model_path) != 2:
                raise ValueError(
                    "zip_model_path as a list/tuple must have length 2: [classification_model, regression_model]"
                )
            zip_model_path = {
                "binary": zip_model_path[0],
                "multiclass": zip_model_path[0],
                "regression": zip_model_path[1],
                "quantile": zip_model_path[1],
            }

        if not isinstance(zip_model_path, dict):
            raise ValueError(
                "zip_model_path must be either "
                "[classification_model, regression_model] or "
                "{'binary': ..., 'multiclass': ..., 'regression': ...}"
            )

        zip_model_path = {**default_model_map, **zip_model_path}
        model_path = zip_model_path.get(self.problem_type)

        if model_path is None:
            return None

        return model_dir / model_path

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Heuristic memory estimate based on TabPFN's memory estimate logic in:
        https://github.com/PriorLabs/TabPFN/blob/57a2efd3ebdb3886245e4d097cefa73a5261a969/src/tabpfn/model/memory.py#L147.

        This is based on GPU memory usage, but hopefully with overheads it also approximates CPU memory usage.
        """
        # TODO: update, this is not correct anymore, consider using internal TabPFN functions directly.
        features_per_group = 3  # Based on TabPFNv2 default (unused)
        n_layers = 12  # Based on TabPFNv2 default
        embedding_size = 192  # Based on TabPFNv2 default
        dtype_byte_size = 2  # Based on TabPFNv2 default

        model_mem = 14489108  # Based on TabPFNv2 default

        n_samples, n_features = X.shape[0], min(X.shape[1], 500)
        n_feature_groups = (n_features) / features_per_group + 1  # TODO: Unsure how to calculate this

        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = n_samples * n_feature_groups * embedding_size * n_layers * dtype_byte_size

        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        # Add some buffer to each term + 1 GB overhead to be safe
        return int(model_mem + 4 * X_mem + 2 * activation_mem + baseline_overhead_mem_est)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        raise NotImplementedError("This method must be implemented in the subclass.")

    def _log_license(self, device: str):
        if self.license_noncommercial:
            global _HAS_LOGGED_TABPFN_NONCOMMERICAL
            if not _HAS_LOGGED_TABPFN_NONCOMMERICAL:
                logger.log(
                    30,
                    f"\tWarning: {self.ag_name} is a NONCOMMERCIAL model. "
                    "Usage of this artifact (including through AutoGluon) is not permitted "
                    "for commercial tasks unless granted explicit permission "
                    "by the model authors (PriorLabs).",
                )
                _HAS_LOGGED_TABPFN_NONCOMMERICAL = True  # Avoid repeated logging
        else:
            global _HAS_LOGGED_TABPFN_LICENSE
            if not _HAS_LOGGED_TABPFN_LICENSE:
                logger.log(20, "\tBuilt with PriorLabs-TabPFN")  # Aligning with TabPFNv2 license requirements
                _HAS_LOGGED_TABPFN_LICENSE = True  # Avoid repeated logging

    def _log_cpu_warning(self, device: str):
        global _HAS_LOGGED_TABPFN_CPU_WARNING
        if not _HAS_LOGGED_TABPFN_CPU_WARNING:
            if device == "cpu":
                logger.log(
                    20, "\tRunning TabPFN on CPU. This can be very slow. It is recommended to run TabPFN on a GPU."
                )
                _HAS_LOGGED_TABPFN_CPU_WARNING = True


class RealTabPFNv25Model(TabPFNModel):
    """RealTabPFN-v2.5 version: https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report.

    We name this model RealTabPFN-v2.5 as its default checkpoints were trained on
    real-world datasets, following the naming conventions of Prior Labs.
    The extra checkpoints include models trained on only synthetic datasets as well.

    .. versionadded:: 1.5.0
    """

    ag_key = "REALTABPFN-V2.5"
    ag_name = "RealTabPFN-v2.5"
    license_noncommercial: ClassVar[bool] = True

    default_classification_model: str | None = "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    default_regression_model: str | None = "tabpfn-v2.5-regressor-v2.5_default.ckpt"

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        """The list of checkpoints to use for hyperparameter tuning."""
        if problem_type == "classification":
            return [
                "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
                "tabpfn-v2.5-classifier-v2.5_large-features-L.ckpt",
                "tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt",
                "tabpfn-v2.5-classifier-v2.5_large-samples.ckpt",
                "tabpfn-v2.5-classifier-v2.5_real-large-features.ckpt",
                "tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features.ckpt",
                "tabpfn-v2.5-classifier-v2.5_real.ckpt",
                "tabpfn-v2.5-classifier-v2.5_variant.ckpt",
            ]

        return [
            "tabpfn-v2.5-regressor-v2.5_low-skew.ckpt",
            "tabpfn-v2.5-regressor-v2.5_quantiles.ckpt",
            "tabpfn-v2.5-regressor-v2.5_real-variant.ckpt",
            "tabpfn-v2.5-regressor-v2.5_real.ckpt",
            "tabpfn-v2.5-regressor-v2.5_small-samples.ckpt",
            "tabpfn-v2.5-regressor-v2.5_variant.ckpt",
        ]


class RealTabPFNv2Model(TabPFNModel):
    """RealTabPFN-v2 version

    We name this model RealTabPFN-v2 as its default checkpoints were trained on
    real-world datasets, following the naming conventions of Prior Labs.
    The extra checkpoints include models trained on only synthetic datasets as well.

    .. versionadded:: 1.5.0
    """

    ag_key = "REALTABPFN-V2"
    ag_name = "RealTabPFN-v2"

    # TODO: Verify if this is the same as the "default" ckpt
    default_classification_model: str | None = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
    default_regression_model: str | None = "tabpfn-v2-regressor-v2_default.ckpt"

    _default_auxiliary_params_extra = {
        "max_rows": 10_000,
        "max_features": 500,
        "max_classes": 10,
        "max_batch_size": 10000,  # TabPFN seems to cryptically error if predicting on 100,000 samples.
    }

    # FIXME: Avoid code dupe. This one has 500 features max, 2.5 has 2000.
    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Heuristic memory estimate based on TabPFN's memory estimate logic in:
        https://github.com/PriorLabs/TabPFN/blob/57a2efd3ebdb3886245e4d097cefa73a5261a969/src/tabpfn/model/memory.py#L147.

        This is based on GPU memory usage, but hopefully with overheads it also approximates CPU memory usage.
        """
        # TODO: update, this is not correct anymore, consider using internal TabPFN functions directly.
        features_per_group = 3  # Based on TabPFNv2 default (unused)
        n_layers = 12  # Based on TabPFNv2 default
        embedding_size = 192  # Based on TabPFNv2 default
        dtype_byte_size = 2  # Based on TabPFNv2 default

        model_mem = 14489108  # Based on TabPFNv2 default

        n_samples, n_features = X.shape[0], min(X.shape[1], 500)
        n_feature_groups = (n_features) / features_per_group + 1  # TODO: Unsure how to calculate this

        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = n_samples * n_feature_groups * embedding_size * n_layers * dtype_byte_size

        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        # Add some buffer to each term + 1 GB overhead to be safe
        return int(model_mem + 4 * X_mem + 2 * activation_mem + baseline_overhead_mem_est)
