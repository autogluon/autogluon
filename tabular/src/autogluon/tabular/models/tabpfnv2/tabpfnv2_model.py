"""
Code Adapted from TabArena: https://github.com/autogluon/tabrepo/blob/main/tabrepo/benchmark/models/ag/tabpfnv2/tabpfnv2_model.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from sklearn.preprocessing import PowerTransformer
from typing_extensions import Self

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular import __version__

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_HAS_LOGGED_TABPFN_LICENSE: bool = False


# TODO: merge into TabPFnv2 codebase
class FixedSafePowerTransformer(PowerTransformer):
    """Fixed version of safe power."""

    def __init__(
        self,
        variance_threshold: float = 1e-3,
        large_value_threshold: float = 100,
        method="yeo-johnson",
        standardize=True,
        copy=True,
    ):
        super().__init__(method=method, standardize=standardize, copy=copy)
        self.variance_threshold = variance_threshold
        self.large_value_threshold = large_value_threshold

        self.revert_indices_ = None

    def _find_features_to_revert_because_of_failure(
        self,
        transformed_X: np.ndarray,
    ) -> None:
        # Calculate the variance for each feature in the transformed data
        variances = np.nanvar(transformed_X, axis=0)

        # Identify features where the variance is not close to 1
        mask = np.abs(variances - 1) > self.variance_threshold
        non_unit_variance_indices = np.where(mask)[0]

        # Identify features with values greater than the large_value_threshold
        large_value_indices = np.any(transformed_X > self.large_value_threshold, axis=0)
        large_value_indices = np.nonzero(large_value_indices)[0]

        # Identify features to revert based on either condition
        self.revert_indices_ = np.unique(
            np.concatenate([non_unit_variance_indices, large_value_indices]),
        )

    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"overflow encountered",
                    category=RuntimeWarning,
                )
                return super()._yeo_johnson_optimize(x)  # type: ignore
        except scipy.optimize._optimize.BracketError:
            return np.nan

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        if np.isnan(lmbda):
            return x

        return super()._yeo_johnson_transform(x, lmbda)  # type: ignore

    def _revert_failed_features(
        self,
        transformed_X: np.ndarray,
        original_X: np.ndarray,
    ) -> np.ndarray:
        # Replace these features with the original features
        if self.revert_indices_ and (self.revert_indices_) > 0:
            transformed_X[:, self.revert_indices_] = original_X[:, self.revert_indices_]

        return transformed_X

    def fit(self, X: np.ndarray, y: Any | None = None) -> FixedSafePowerTransformer:
        super().fit(X, y)

        # Check and revert features as necessary
        self._find_features_to_revert_because_of_failure(super().transform(X))  # type: ignore
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        transformed_X = super().transform(X)
        return self._revert_failed_features(transformed_X, X)  # type: ignore


# FIXME: Need to take this logic into v6 for loading on CPU
class TabPFNV2Model(AbstractTorchModel):
    """
    TabPFNv2 is a tabular foundation model pre-trained purely on synthetic data that achieves
    state-of-the-art results with in-context learning on small datasets with <=10000 samples and <=500 features.
    TabPFNv2 is developed and maintained by PriorLabs: https://priorlabs.ai/

    TabPFNv2 is the top performing method for small datasets on TabArena-v0.1: https://tabarena.ai

    Paper: Accurate predictions on small data with a tabular foundation model
    Authors: Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister & Frank Hutter
    Codebase: https://github.com/PriorLabs/TabPFN
    License: https://github.com/PriorLabs/TabPFN/blob/main/LICENSE

    .. versionadded:: 1.4.0
    """
    ag_key = "TABPFNV2"
    ag_name = "TabPFNv2"
    ag_priority = 105
    seed_name = "random_state"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached_model = False
        self._feature_generator = None
        self._cat_features = None
        self._cat_indices = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._cat_indices = []

            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        # This converts categorical features to numeric via stateful label encoding.
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X
            )

            if is_train:
                # Detect/set cat features and indices
                if self._cat_features is None:
                    self._cat_features = self._feature_generator.features_in[:]
                self._cat_indices = [X.columns.get_loc(col) for col in self._cat_features]

        return X

    def _get_model_cls(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        is_classification = self.problem_type in ["binary", "multiclass"]
        model_base = TabPFNClassifier if is_classification else TabPFNRegressor
        return model_base

    # FIXME: Crashes during model download if bagging with parallel fit.
    #  Consider adopting same download logic as TabPFNMix which doesn't crash during model download.
    # FIXME: Maybe support child_oof somehow with using only one model and being smart about inference time?
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        try:
            from tabpfn.model import preprocessing
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import tabpfn! To use the TabPFNv2 model, "
                f"do: `pip install autogluon.tabular[tabpfn]=={__version__}`.",
            )
            raise err

        preprocessing.SafePowerTransformer = FixedSafePowerTransformer

        is_classification = self.problem_type in ["binary", "multiclass"]

        model_base = self._get_model_cls()

        from tabpfn.model.loading import resolve_model_path
        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        if verbosity >= 2:
            # logs "Built with PriorLabs-TabPFN"
            self._log_license(device=device)

        X = self.preprocess(X, is_train=True)

        hps = self._get_model_params()
        hps["device"] = device
        hps["n_jobs"] = num_cpus
        hps["categorical_features_indices"] = self._cat_indices

        _, model_dir, _, _ = resolve_model_path(
            model_path=None,
            which="classifier" if is_classification else "regressor",
        )
        if is_classification:
            if "classification_model_path" in hps:
                hps["model_path"] = model_dir / hps.pop("classification_model_path")
            if "regression_model_path" in hps:
                del hps["regression_model_path"]
        else:
            if "regression_model_path" in hps:
                hps["model_path"] = model_dir / hps.pop("regression_model_path")
            if "classification_model_path" in hps:
                del hps["classification_model_path"]

        # Resolve inference_config
        inference_config = {
            _k: v
            for k, v in hps.items()
            if k.startswith("inference_config/") and (_k := k.split("/")[-1])
        }
        if inference_config:
            hps["inference_config"] = inference_config
        for k in list(hps.keys()):
            if k.startswith("inference_config/"):
                del hps[k]

        # TODO: remove power from search space and TabPFNv2 codebase
        # Power transform can fail. To avoid this, make all power be safepower instead.
        if "PREPROCESS_TRANSFORMS" in inference_config:
            safe_config = []
            for preprocessing_dict in inference_config["PREPROCESS_TRANSFORMS"]:
                if preprocessing_dict["name"] == "power":
                    preprocessing_dict["name"] = "safepower"
                safe_config.append(preprocessing_dict)
            inference_config["PREPROCESS_TRANSFORMS"] = safe_config
        if "REGRESSION_Y_PREPROCESS_TRANSFORMS" in inference_config:
            safe_config = []
            for preprocessing_name in inference_config[
                "REGRESSION_Y_PREPROCESS_TRANSFORMS"
            ]:
                if preprocessing_name == "power":
                    preprocessing_name = "safepower"
                safe_config.append(preprocessing_name)
            inference_config["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = safe_config

        # Resolve model_type
        n_ensemble_repeats = hps.pop("n_ensemble_repeats", None)
        model_is_rf_pfn = hps.pop("model_type", "no") == "dt_pfn"
        if model_is_rf_pfn:
            from .rfpfn import RandomForestTabPFNClassifier, RandomForestTabPFNRegressor

            hps["n_estimators"] = 1
            rf_model_base = (
                RandomForestTabPFNClassifier
                if is_classification
                else RandomForestTabPFNRegressor
            )
            self.model = rf_model_base(
                tabpfn=model_base(**hps),
                categorical_features=self._cat_indices,
                n_estimators=n_ensemble_repeats,
            )
        else:
            if n_ensemble_repeats is not None:
                hps["n_estimators"] = n_ensemble_repeats
            self.model = model_base(**hps)

        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def get_device(self) -> str:
        return self.model.devices_.type

    def _set_device(self, device: str):
        pass  # TODO: Unknown how to properly set device for TabPFN after loading. Refer to `_set_device_tabpfn`.

    # FIXME: This is not comprehensive. Need model authors to add an official API set_device
    def _set_device_tabpfn(self, device: str):
        import torch
        # Move all torch components to the target device
        device = self.to_torch_device(device)
        self.model.devices_ = device
        if hasattr(self.model.executor_, "model") and self.model.executor_.model is not None:
            self.model.executor_.model.to(self.model.devices_)
        if hasattr(self.model.executor_, "models"):
            self.model.executor_.models = [m.to(self.model.devices_) for m in self.model.executor_.models]

        # Restore other potential torch objects from fitted_attrs
        for key, value in vars(self.model).items():
            if key.endswith("_") and hasattr(value, "to"):
                setattr(self.model, key, value.to(self.model.devices_))

    def model_weights_path(self, path: str | None = None) -> Path:
        if path is None:
            path = self.path
        return Path(path) / "config.tabpfn_fit"

    def save(self, path: str = None, verbose=True) -> str:
        _model = self.model
        is_fit = self.is_fit()
        if is_fit:
            self._save_model_artifact(path=path)
            self._cached_model = True
            self.model = None
        path = super().save(path=path, verbose=verbose)
        if is_fit:
            self.model = _model
        return path

    # TODO: It is required to do this because it is unknown how to otherwise save TabPFN in CPU-only mode.
    #  Even though we would generally prefer to save it in the pkl for better insurance
    #  that the model will work in future (self-contained)
    def _save_model_artifact(self, path: str | None = None):
        # save with CPU device so it can be loaded on a CPU only machine
        device_og = self.device
        self._set_device_tabpfn(device="cpu")
        self.model.save_fit_state(path=self.model_weights_path(path=path))
        self._set_device_tabpfn(device=device_og)

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True) -> Self:
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._cached_model:
            model._load_model_artifact()
            model._cached_model = False
        return model

    def _load_model_artifact(self):
        model_cls = self._get_model_cls()
        device = self.suggest_device_infer()
        self.model = model_cls.load_from_fit_state(path=self.model_weights_path(), device=device)
        self.device = device

    def _log_license(self, device: str):
        global _HAS_LOGGED_TABPFN_LICENSE
        if not _HAS_LOGGED_TABPFN_LICENSE:
            logger.log(20, "\tBuilt with PriorLabs-TabPFN")  # Aligning with TabPFNv2 license requirements
            if device == "cpu":
                logger.log(
                    20,
                    "\tRunning TabPFNv2 on CPU. This can be very slow. "
                    "It is recommended to run TabPFNv2 on a GPU."
                )
            _HAS_LOGGED_TABPFN_LICENSE = True  # Avoid repeated logging

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def _set_default_params(self):
        default_params = {
            "ignore_pretraining_limits": True,  # to ignore warnings and size limits
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 10000,
                "max_features": 500,
                "max_classes": 10,
                "max_batch_size": 10000,  # TabPFN seems to cryptically error if predicting on 100,000 samples.
            }
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            # FIXME: Find a work-around to avoid crash if parallel and weights are not downloaded
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": True,  # Better to refit the model for faster inference and similar quality as the bag.
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=hyperparameters,
            **kwargs,
        )

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
        # features_per_group = 2  # Based on TabPFNv2 default (unused)
        n_layers = 12  # Based on TabPFNv2 default
        embedding_size = 192  # Based on TabPFNv2 default
        dtype_byte_size = 2  # Based on TabPFNv2 default

        model_mem = 14489108  # Based on TabPFNv2 default

        n_samples, n_features = X.shape[0], X.shape[1]
        n_feature_groups = n_features + 1  # TODO: Unsure how to calculate this

        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = (
            n_samples * n_feature_groups * embedding_size * n_layers * dtype_byte_size
        )

        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        # Add some buffer to each term + 1 GB overhead to be safe
        return int(
            model_mem + 4 * X_mem + 2 * activation_mem + baseline_overhead_mem_est
        )

    @classmethod
    def _class_tags(cls):
        return {
            "can_estimate_memory_usage_static": True,
            "can_set_device": True,
            "set_device_on_save_to": None,
            "set_device_on_load": False,
        }

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
