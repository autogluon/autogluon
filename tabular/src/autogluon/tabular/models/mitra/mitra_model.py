from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from typing_extensions import Self

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular import __version__
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

logger = logging.getLogger(__name__)


class MitraModel(AbstractTorchModel):
    """
    Mitra is a tabular foundation model pre-trained purely on synthetic data with the goal
    of optimizing fine-tuning performance over in-context learning performance.
    Mitra was developed by the AutoGluon team @ AWS AI.

    Mitra's default hyperparameters outperforms all methods for small datasets on TabArena-v0.1 (excluding ensembling): https://tabarena.ai

    Authors: Xiyuan Zhang, Danielle C. Maddix, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W. Mahoney, Cuixiong Hu, Huzefa Rangwala, George Karypis, Bernie Wang
    Blog Post: https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models
    License: Apache-2.0

    .. versionadded:: 1.4.0
    """

    ag_key = "MITRA"
    ag_name = "Mitra"
    weights_file_name = "model.pt"
    ag_priority = 55
    seed_name = "seed"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._weights_saved = False
        self._feature_generator = None

    @staticmethod
    def _get_default_device():
        """Get the best available device for the current system."""
        if ResourceManager.get_gpu_count_torch(cuda_only=True) > 0:
            logger.log(15, "Using CUDA GPU")
            return "cuda"
        else:
            return "cpu"

    def get_model_cls(self):
        if self.problem_type in ["binary", "multiclass"]:
            from .sklearn_interface import MitraClassifier

            model_cls = MitraClassifier
        elif self.problem_type == "regression":
            from .sklearn_interface import MitraRegressor

            model_cls = MitraRegressor
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")
        return model_cls

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)

        if is_train:
            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        # This converts categorical features to numeric via stateful label encoding.
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        # TODO: Reset the number of threads based on the specified num_cpus
        need_to_reset_torch_threads = False
        torch_threads_og = None

        try:
            model_cls = self.get_model_cls()
            import torch
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import Mitra! To use the Mitra model, "
                f"do: `pip install autogluon.tabular[mitra]=={__version__}`.",
            )
            raise err

        if num_cpus is not None and isinstance(num_cpus, (int, float)):
            torch_threads_og = torch.get_num_threads()
            if torch_threads_og != num_cpus:
                # reset torch threads back to original value after fit
                torch.set_num_threads(num_cpus)
                need_to_reset_torch_threads = True

        hyp = self._get_model_params()

        hf_cls_model = hyp.pop("hf_cls_model", None)
        hf_reg_model = hyp.pop("hf_reg_model", None)
        if self.problem_type in ["binary", "multiclass"]:
            hf_model = hf_cls_model
        elif self.problem_type == "regression":
            hf_model = hf_reg_model
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")
        if hf_model is None:
            hf_model = hyp.pop("hf_general_model", None)
        if hf_model is None:
            hf_model = hyp.pop("hf_model", None)
        if hf_model is not None:
            logger.log(30, f"\tCustom hf_model specified: {hf_model}")
            hyp["hf_model"] = hf_model

        if hyp.get("device", None) is None:
            if num_gpus == 0:
                hyp["device"] = "cpu"
            else:
                hyp["device"] = self._get_default_device()

        if hyp["device"] == "cpu" and hyp.get("fine_tune", True):
            logger.log(
                30,
                f"\tWarning: Attempting to fine-tune Mitra on CPU. This will be very slow. "
                f"We strongly recommend using a GPU instance to fine-tune Mitra.",
            )

        if "state_dict_classification" in hyp:
            state_dict_classification = hyp.pop("state_dict_classification")
            if self.problem_type in ["binary", "multiclass"]:
                hyp["state_dict"] = state_dict_classification
        if "state_dict_regression" in hyp:
            state_dict_regression = hyp.pop("state_dict_regression")
            if self.problem_type in ["regression"]:
                hyp["state_dict"] = state_dict_regression

        if "verbose" not in hyp:
            hyp["verbose"] = verbosity >= 3

        self.model = model_cls(**hyp)

        X = self.preprocess(X, is_train=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        model = self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            time_limit=time_limit,
        )

        for i in range(len(model.trainers)):
            model.trainers[i].post_fit_optimize()

        self.model = model

        if need_to_reset_torch_threads:
            torch.set_num_threads(torch_threads_og)

    def _set_default_params(self):
        default_params = {
            "n_estimators": 1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 10000,
                "max_features": 500,
                "max_classes": 10,
            }
        )
        return default_auxiliary_params

    def weights_path(self, path: str | None = None) -> str:
        if path is None:
            path = self.path
        return str(Path(path) / self.weights_file_name)

    def save(self, path: str = None, verbose=True) -> str:
        _model_weights_list = None
        if self.model is not None:
            self._save_model_artifact(path=path)
            _model_weights_list = []
            for i in range(len(self.model.trainers)):
                _model_weights_list.append(self.model.trainers[i].model)
                self.model.trainers[i].model = None

        path = super().save(path=path, verbose=verbose)
        if _model_weights_list is not None:
            for i in range(len(self.model.trainers)):
                self.model.trainers[i].model = _model_weights_list[i]
        return path

    def _save_model_artifact(self, path: str | None):
        if path is None:
            path = self.path
        import torch

        device_og = self.device
        self.set_device("cpu")

        _model_weights_list = []
        for i in range(len(self.model.trainers)):
            _model_weights_list.append(self.model.trainers[i].model)

        os.makedirs(path, exist_ok=True)
        torch.save(_model_weights_list, self.weights_path(path=path))
        self.set_device(device_og)
        self._weights_saved = True

    def _load_model_artifact(self):
        import torch

        device = self.suggest_device_infer()
        model_weights_list = torch.load(self.weights_path(), weights_only=False)  # nosec B614
        for i in range(len(self.model.trainers)):
            self.model.trainers[i].model = model_weights_list[i]
        self.set_device(device)

    def _set_device(self, device: str):
        for i in range(len(self.model.trainers)):
            self.model.trainers[i].set_device(device)

    def get_device(self) -> str:
        return self.model.trainers[0].device

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True) -> Self:
        model: MitraModel = super().load(path=path, reset_paths=reset_paths, verbose=verbose)

        if model._weights_saved:
            model._load_model_artifact()
            model._weights_saved = False
        return model

    @classmethod
    def download_weights(cls, repo_id: str):
        """
        Download weights for Mitra from HuggingFace from `repo_id`.
        Requires an internet connection.
        """
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=repo_id, filename="config.json")
        hf_hub_download(repo_id=repo_id, filename="model.safetensors")

    @classmethod
    def download_default_weights(cls):
        """
        Download default weights for Mitra from HuggingFace.
        Includes both classifier and regressor weights.

        This is useful to call when building a docker image to avoid having to download Mitra weights for each instance.
        This is also useful for benchmarking as a first sanity check
        to avoid HuggingFace potentially blocking the download.

        Requires an internet connection.
        """
        cls.download_weights(repo_id="autogluon/mitra-classifier")
        cls.download_weights(repo_id="autogluon/mitra-regressor")

    @classmethod
    def supported_problem_types(cls) -> Optional[List[str]]:
        return ["binary", "multiclass", "regression"]

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        # FIXME: Test if it works with parallel, need to enable n_cpus support
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",  # FIXME: Comment out after debugging for large speedup
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return self.estimate_memory_usage_static(
            X=X, problem_type=self.problem_type, num_classes=self.num_classes, **kwargs
        )

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        # Multiply by 0.9 as currently this is overly safe
        return int(
            0.9
            * max(
                cls._estimate_memory_usage_static_cpu_icl(X=X, **kwargs),
                cls._estimate_memory_usage_static_cpu_ft_icl(X=X, **kwargs),
                cls._estimate_memory_usage_static_gpu_cpu(X=X, **kwargs),
                cls._estimate_memory_usage_static_gpu_gpu(X=X, **kwargs),
            )
        )

    @classmethod
    def _estimate_memory_usage_static_cpu_icl(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        rows, features = X.shape[0], X.shape[1]

        # For very small datasets, use a more conservative estimate
        if rows * features < 100:  # Small dataset threshold
            # Use a simpler linear formula for small datasets
            cpu_memory_kb = 1.3 * (100 * rows * features + 1000000)  # 1GB base + linear scaling
        else:
            # Original formula for larger datasets
            cpu_memory_kb = 1.3 * (
                0.001748 * (rows**2) * features + 0.001206 * rows * (features**2) + 10.3482 * rows * features + 6409698
            )
        return int(cpu_memory_kb * 1e3)

    @classmethod
    def _estimate_memory_usage_static_cpu_ft_icl(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        rows, features = X.shape[0], X.shape[1]

        # For very small datasets, use a more conservative estimate
        if rows * features < 100:  # Small dataset threshold
            # Use a simpler linear formula for small datasets
            cpu_memory_kb = 1.3 * (200 * rows * features + 2000000)  # 2GB base + linear scaling
        else:
            # Original formula for larger datasets
            cpu_memory_kb = 1.3 * (
                0.001 * (rows**2) * features + 0.004541 * rows * (features**2) + 46.2974 * rows * features + 5605681
            )
        return int(cpu_memory_kb * 1e3)

    @classmethod
    def _estimate_memory_usage_static_gpu_cpu(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        rows, features = X.shape[0], X.shape[1]

        # For very small datasets, use a more conservative estimate
        if rows * features < 100:  # Small dataset threshold
            return int(2.5 * 1e9)  # 2.5GB for small datasets
        else:
            return int(5 * 1e9)  # 5GB for larger datasets

    @classmethod
    def _estimate_memory_usage_static_gpu_gpu(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        rows, features = X.shape[0], X.shape[1]

        # For very small datasets, use a more conservative estimate
        if rows * features < 100:  # Small dataset threshold
            # Use a simpler linear formula for small datasets
            gpu_memory_mb = 1.3 * (10 * rows * features + 2000)  # 2GB base + linear scaling
        else:
            # Original formula for larger datasets
            gpu_memory_mb = 1.3 * (0.05676 * rows * features + 3901)
        return int(gpu_memory_mb * 1e6)

    @classmethod
    def _class_tags(cls):
        return {
            "can_estimate_memory_usage_static": True,
            "can_set_device": True,
            "set_device_on_save_to": None,
            "set_device_on_load": False,
        }

    def _more_tags(self) -> dict:
        tags = {"can_refit_full": True}
        return tags
