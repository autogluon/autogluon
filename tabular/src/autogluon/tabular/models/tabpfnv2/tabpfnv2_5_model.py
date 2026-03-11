from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_HAS_LOGGED_TABPFN_LICENSE: bool = False
_HAS_LOGGED_TABPFN_NONCOMMERICAL: bool = False
_HAS_LOGGED_TABPFN_CPU_WARNING: bool = False


class TabPFNModel(AbstractTorchModel):
    """TabPFN-2.5 is a tabular foundation model that is developed and maintained by PriorLabs: https://priorlabs.ai/.

    This class is an abstract template for various TabPFN versions as subclasses.

    Paper: Accurate predictions on small data with a tabular foundation model
    Authors: Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister & Frank Hutter
    Codebase: https://github.com/PriorLabs/TabPFN
    License: https://github.com/PriorLabs/TabPFN/blob/main/LICENSE

    .. versionadded:: 1.5.0
    """

    ag_key = "NOTSET"
    ag_name = "NOTSET"
    ag_priority = 40
    seed_name = "random_state"

    custom_model_dir: str | None = None
    default_classification_model: str | None = "NOTSET"
    default_regression_model: str | None = "NOTSET"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
            X = X.copy(deep=False)
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

            if is_train:
                # Detect/set cat features and indices
                if self._cat_features is None:
                    self._cat_features = self._feature_generator.features_in[:]
                self._cat_indices = [X.columns.get_loc(col) for col in self._cat_features]

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

        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.model.loading import resolve_model_path
        from torch.cuda import is_available

        is_classification = self.problem_type in ["binary", "multiclass"]

        model_base = TabPFNClassifier if is_classification else TabPFNRegressor

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        if verbosity >= 2:
            # logs "Built with PriorLabs-TabPFN"
            self._log_license(device=device)
            self._log_cpu_warning(device=device)

        X = self.preprocess(X, y=y, is_train=True)

        hps = self._get_model_params()
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

        # Resolve model_path
        if self.custom_model_dir is not None:
            model_dir = Path(self.custom_model_dir)
        else:
            _, model_dir, _, _ = resolve_model_path(
                model_path=None,
                which="classifier" if is_classification else "regressor",
            )
            model_dir = model_dir[0]
        clf_path, reg_path = hps.pop(
            "zip_model_path",
            [self.default_classification_model, self.default_regression_model],
        )
        model_path = clf_path if is_classification else reg_path
        if model_path is not None:
            hps["model_path"] = model_dir / model_path

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

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    def _set_default_params(self):
        default_params = {
            "ignore_pretraining_limits": True,  # to ignore warnings and size limits
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def get_device(self) -> str:
        return self.model.devices_[0].type

    def _set_device(self, device: str):
        self.model.to(device)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression", "quantile"]

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 100_000,
                "max_features": 2000,
                "max_classes": 10,
                "model_telemetry": False,
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
    def disable_tabpfn_telemetry(cls):
        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

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

        n_samples, n_features = X.shape[0], min(X.shape[1], 2000)
        n_feature_groups = (n_features) / features_per_group + 1  # TODO: Unsure how to calculate this

        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = n_samples * n_feature_groups * embedding_size * n_layers * dtype_byte_size

        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        # Add some buffer to each term + 1 GB overhead to be safe
        return int(model_mem + 4 * X_mem + 2 * activation_mem + baseline_overhead_mem_est)

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        raise NotImplementedError("This method must be implemented in the subclass.")

    def _log_license(self, device: str):
        pass

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

    def _log_license(self, device: str):
        global _HAS_LOGGED_TABPFN_NONCOMMERICAL
        if not _HAS_LOGGED_TABPFN_NONCOMMERICAL:
            logger.log(
                30,
                "\tWarning: TabPFN-2.5 is a NONCOMMERCIAL model. "
                "Usage of this artifact (including through AutoGluon) is not permitted "
                "for commercial tasks unless granted explicit permission "
                "by the model authors (PriorLabs).",
            )  # Aligning with TabPFNv25 license
            _HAS_LOGGED_TABPFN_NONCOMMERICAL = True  # Avoid repeated logging


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

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 10_000,
                "max_features": 500,
                "max_classes": 10,
                "max_batch_size": 10000,  # TabPFN seems to cryptically error if predicting on 100,000 samples.
            }
        )
        return default_auxiliary_params

    def _log_license(self, device: str):
        global _HAS_LOGGED_TABPFN_LICENSE
        if not _HAS_LOGGED_TABPFN_LICENSE:
            logger.log(20, "\tBuilt with PriorLabs-TabPFN")  # Aligning with TabPFNv2 license requirements
            _HAS_LOGGED_TABPFN_LICENSE = True  # Avoid repeated logging

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
