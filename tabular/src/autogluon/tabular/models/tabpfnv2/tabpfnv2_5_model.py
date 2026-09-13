from __future__ import annotations

import copy
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from ._weight_fetch import weight_fetch_policy

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


def _tensor_bytes(modules) -> int:
    """Bytes held by the parameters and buffers of `modules`, each tensor counted once."""
    seen = set()
    total = 0
    for module in modules:
        for tensor in (*module.parameters(), *module.buffers()):
            if id(tensor) not in seen:
                seen.add(id(tensor))
                total += tensor.numel() * tensor.element_size()
    return total


def _narrow_array(obj: object, name: str, narrowed_dtypes: dict) -> None:
    """Replace `obj.name` with a narrower view of itself, if one is allowed."""
    array = getattr(obj, name, None)
    narrower = narrowed_dtypes.get(getattr(array, "dtype", None))
    if narrower is not None:
        setattr(obj, name, array.astype(narrower, copy=False))


#: Built networks the process keeps for reuse, one per (checkpoint, estimator type, device),
#: most recently used last. Sized by `TabPFNModel.shared_network_capacity`.
_MODEL_SPECS: OrderedDict[tuple, object] = OrderedDict()
_MODEL_SPECS_LOCK = threading.RLock()


def _shared_model_specs(checkpoint_path: str, estimator_type: str, device: str, capacity: int):
    """The registered network for the key, built on first use.

    The registry keeps the `capacity` most recently used networks. Evicting one drops the
    registry's reference only: estimators that hold it keep it, and it is freed once the
    last of them is gone. `capacity <= 0` builds a network that is not registered.
    """
    key = (checkpoint_path, estimator_type, device)
    with _MODEL_SPECS_LOCK:
        if capacity <= 0:
            return _build_model_specs(checkpoint_path, estimator_type, device)
        if key in _MODEL_SPECS:
            _MODEL_SPECS.move_to_end(key)
        else:
            _MODEL_SPECS[key] = _build_model_specs(checkpoint_path, estimator_type, device)
            while len(_MODEL_SPECS) > capacity:
                _MODEL_SPECS.popitem(last=False)
        return _MODEL_SPECS[key]


def release_shared_networks() -> None:
    """Drop the registry's references to every shared network; live estimators keep theirs."""
    with _MODEL_SPECS_LOCK:
        _MODEL_SPECS.clear()


def _build_model_specs(checkpoint_path: str, estimator_type: str, device: str):
    """The model specs a tabpfn estimator accepts as `model_path`, with the network on `device`."""
    import dataclasses
    import inspect

    from tabpfn.base import ClassifierModelSpecs, RegressorModelSpecs
    from tabpfn.model_loading import load_model_criterion_config, resolve_model_version

    version = resolve_model_version(checkpoint_path)
    type_kw = (
        "estimator_type" if "estimator_type" in inspect.signature(load_model_criterion_config).parameters else "which"
    )
    models, criterion, configs, inference_config = load_model_criterion_config(
        model_path=checkpoint_path,
        check_bar_distribution_criterion=estimator_type == "regressor",
        cache_trainset_representation=False,
        version=version.value,
        download_if_not_exists=True,
        **{type_kw: estimator_type},
    )
    try:
        from tabpfn.inference_config import cpu_sample_limit
    except ImportError:
        pass
    else:
        inference_config = dataclasses.replace(inference_config, MAX_CPU_SAMPLES=cpu_sample_limit(version))
    model = models[0]
    model.to(device)
    if estimator_type == "regressor":
        criterion.to(device)
        return RegressorModelSpecs(model, configs[0], inference_config, criterion)
    return ClassifierModelSpecs(model, configs[0], inference_config)


def _shallow_copy(obj):
    new = object.__new__(type(obj))
    new.__dict__.update(obj.__dict__)
    return new


def _detach_network(estimator):
    """A shallow copy of the fitted estimator without its network; the live estimator keeps it."""
    est = _shallow_copy(estimator)
    est.models_ = None
    if hasattr(est, "executor_"):
        executor = _shallow_copy(est.executor_)
        executor.model_caches = None
        est.executor_ = executor
    for name in ("znorm_space_bardist_", "raw_space_bardist_"):
        if hasattr(est, name):
            setattr(est, name, copy.deepcopy(getattr(est, name)).to("cpu"))
    return est


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
    skipped by memory estimates assuming it). Versions for which every chunk
    re-runs the forward pass over the whole training context (TabPFN-3) override
    this with a high floor, since for them small chunks multiply predict time
    while saving little memory."""

    max_batch_size_slack: int = 0
    """Rows a prediction set may exceed the training set by before the ``"auto"``
    ``ag.max_batch_size`` resolution chunks it: ``"auto"`` resolves to
    ``n_train + max_batch_size_slack`` (within ``[max_batch_size_min, 1M]``)."""

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

    shared_network_capacity: ClassVar[int] = 1
    """Built networks the process keeps for reuse across this class's fits and loads, one per
    (checkpoint, estimator type, device), most recently used first. A fit or load whose network
    is not registered builds it and evicts the least recently used entry beyond the capacity; an
    estimator holding an evicted network keeps it until it is released. 0 shares nothing.
    Process-wide, so not a per-config hyperparameter: a value set for one config changes what
    every other config of the class sees."""

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

        # "auto" prediction chunking resolves against the training size (see
        # `_resolve_auto_max_batch_size`). None disables chunking entirely. The resolved
        # value is fit state (read via `_get_max_batch_size`), not a params_aux mutation.
        if self.aux_params.max_batch_size == "auto":
            self._max_batch_size_resolved = self._resolve_auto_max_batch_size(n_train=len(X))

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
            checkpoint_path = str(Path(model_path).resolve())
            self._checkpoint_path = checkpoint_path
            self._estimator_type = "classifier" if is_classification else "regressor"
            if self._shares_module(hps, device) and self.shared_network_capacity > 0:
                hps["model_path"] = _shared_model_specs(
                    checkpoint_path, self._estimator_type, device, self.shared_network_capacity
                )
            else:
                hps["model_path"] = checkpoint_path

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
        with weight_fetch_policy(self.aux_params.fetch_pretrained_weights, stage="fit", model_name=self.name):
            self.model = self.model.fit(
                X=X,
                y=y,
            )
        self._narrow_inference_context()
        if model_path is not None and self._shares_module(hps, device) and self.shared_network_capacity > 0:
            # The string keeps the specs out of `get_params()` and the pickle.
            self.model.model_path = checkpoint_path
            if self._estimator_type == "regressor":
                # `fit` assigns the shared criterion; every `.to(device)` would move it in place.
                self.model.znorm_space_bardist_ = copy.deepcopy(self.model.znorm_space_bardist_)

    @staticmethod
    def _shares_module(hps: dict, device) -> bool:
        """Whether the fit can run on a network object shared with other estimators of the process."""
        import torch

        return (
            isinstance(device, str)
            and hps.get("fit_mode", "fit_preprocessors") == "fit_preprocessors"
            and not isinstance(hps.get("inference_precision"), torch.dtype)
        )

    def _network_detached(self) -> bool:
        return self.model is not None and not getattr(self.model, "models_", None)

    def _ensure_network(self, device: str | None = None) -> None:
        """Attach a network to an estimator that was pickled without one."""
        if not self._network_detached():
            return
        import torch

        device = torch.device(device or self.device or self.get_device()).type
        # A weightless pickle depends on the checkpoint being reachable at load time; the
        # fetch policy decides whether a missing one may be downloaded now.
        with weight_fetch_policy(self.aux_params.fetch_pretrained_weights, stage="load", model_name=self.name):
            spec = _shared_model_specs(
                self._checkpoint_path, self._estimator_type, device, self.shared_network_capacity
            )
        est = self.model
        est.models_ = [spec.model]
        if hasattr(est, "executor_"):
            est.executor_._set_models(est.models_)
        est.to(device)
        self._sync_inner_checkpoints_to_engine_devices(device=device)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        est = state.get("model")
        if est is not None and getattr(est, "models_", None) and not self.aux_params.save_pretrained_weights:
            state["model"] = _detach_network(est)
        return state

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        """Pickle on CPU only when the weights are kept; a weightless pickle holds no device tensors."""
        if not (self.is_fit() and self.aux_params.save_pretrained_weights):
            return super().save(path=path, verbose=verbose)
        original_device = self.device
        self.set_device("cpu")
        try:
            return super().save(path=path, verbose=verbose)
        finally:
            self.set_device(original_device)

    def predict_proba(self, X, **kwargs):
        self._ensure_network()
        return super().predict_proba(X, **kwargs)

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

    def _get_memory_size(self) -> int:
        """Pickle size of the model, with the foundation model weights measured from their tensors.

        Pickling the whole model serialises the weights, hundreds of MB, just to measure them. They
        are counted from the parameter and buffer sizes of the loaded checkpoints instead, and only
        the rest of the fitted state is pickled, from shallow copies of the estimator and its inference
        engine with the checkpoints detached (the split `save_fitted_tabpfn_model` makes; its own
        weight-free engine copy is a deep copy that would copy the weights first). With several devices
        the engine holds a copy of the checkpoints per device, which a pickle would include and this
        count does not.

        The base implementation collects garbage first to make room for a pickle that holds the weights;
        the weightless pickle here is small, so that pass is skipped.
        """
        estimator = self.model
        if estimator is None:
            return super()._get_memory_size()
        weightless = copy.copy(estimator)
        weightless.models_ = []
        weightless.executor_ = copy.copy(estimator.executor_)
        weightless.executor_._set_models([])
        self.model = weightless
        try:
            memory_size = self._get_pickled_size()
        finally:
            self.model = estimator
        return memory_size + _tensor_bytes(estimator.models_)

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
        if self._network_detached():
            self._ensure_network(device)
            return
        self.model.to(device)
        self._sync_inner_checkpoints_to_engine_devices(device=device)

    @classmethod
    def _class_tags(cls):
        # `save` does the CPU round trip itself, and only when the weights are kept.
        return {"can_set_device": True, "set_device_on_save_to": None, "set_device_on_load": True}

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
    def _resolve_auto_max_batch_size(cls, *, n_train: int) -> int:
        """The prediction chunk size ``ag.max_batch_size="auto"`` stands for at this training size.

        Chunks re-attend the full training context, so chunks smaller than the training
        set multiply predict time at large ``n_train`` while saving little memory; the
        slack keeps a prediction set slightly larger than the training set (a held-out
        fold of a two-fold bag) in a single chunk. Bounded to ``[max_batch_size_min, 1M]``.
        """
        return min(1_000_000, max(cls.max_batch_size_min, n_train + cls.max_batch_size_slack))

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
            # explicit None (chunking disabled) has no bound, so use the "auto" proxy.
            max_batch_size = cls._resolve_auto_max_batch_size(n_train=n_train)
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
