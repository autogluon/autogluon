# TODO: To ensure deterministic operations we need to set torch.use_deterministic_algorithms(True)
# and os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'. The CUBLAS environment variable configures
# the workspace size for certain CUBLAS operations to ensure reproducibility when using CUDA >= 10.2.
# Both settings are required to ensure deterministic behavior in operations such as matrix multiplications.
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import os
from typing import List, Optional

import pandas as pd
import torch
import logging

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

logger = logging.getLogger(__name__)


# TODO: Needs memory usage estimate method
class MitraModel(AbstractModel):
    ag_key = "MITRA"
    ag_name = "Mitra"
    weights_file_name = "model.pt"
    ag_priority = 55

    def __init__(self, problem_type=None, **kwargs):
        super().__init__(**kwargs)
        self.problem_type = problem_type
        self._weights_saved = False

    @staticmethod
    def _get_default_device():
        """Get the best available device for the current system."""
        if ResourceManager.get_gpu_count_torch(cuda_only=True) > 0:
            logger.info("Using CUDA GPU")
            return "cuda"
        else:
            return "cpu"

    def get_model_cls(self):
        from .sklearn_interface import MitraClassifier

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = MitraClassifier
        elif self.problem_type == "regression":
            from .sklearn_interface import MitraRegressor

            model_cls = MitraRegressor
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")
        return model_cls

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float = None,
        num_cpus: int = 1,
        **kwargs,
    ):
        # TODO: Reset the number of threads based on the specified num_cpus
        need_to_reset_torch_threads = False
        torch_threads_og = None
        if num_cpus is not None and isinstance(num_cpus, (int, float)):
            torch_threads_og = torch.get_num_threads()
            if torch_threads_og != num_cpus:
                # reset torch threads back to original value after fit
                torch.set_num_threads(num_cpus)
                need_to_reset_torch_threads = True

        model_cls = self.get_model_cls()

        hyp = self._get_model_params()
        if "state_dict_classification" in hyp:
            state_dict_classification = hyp.pop("state_dict_classification")
            if self.problem_type in ["binary", "multiclass"]:
                hyp["state_dict"] = state_dict_classification
        if "state_dict_regression" in hyp:
            state_dict_regression = hyp.pop("state_dict_regression")
            if self.problem_type in ["regression"]:
                hyp["state_dict"] = state_dict_regression

        self.model = model_cls(
            **hyp,
        )

        X = self.preprocess(X)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.model = self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            time_limit=time_limit,
        )

        if need_to_reset_torch_threads:
            torch.set_num_threads(torch_threads_og)

    def _set_default_params(self):
        default_params = {
            "device": self._get_default_device(),
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

    @property
    def weights_path(self) -> str:
        return os.path.join(self.path, self.weights_file_name)

    def save(self, path: str = None, verbose=True) -> str:
        _model_weights_list = None
        if self.model is not None:
            _model_weights_list = []
            for i in range(len(self.model.trainers)):
                _model_weights_list.append(self.model.trainers[i].model)
                self.model.trainers[i].checkpoint = None
                self.model.trainers[i].model = None
                self.model.trainers[i].optimizer = None
                self.model.trainers[i].scheduler_warmup = None
                self.model.trainers[i].scheduler_reduce_on_plateau = None
            self._weights_saved = True
        path = super().save(path=path, verbose=verbose)
        if _model_weights_list is not None:
            import torch

            os.makedirs(self.path, exist_ok=True)
            torch.save(_model_weights_list, self.weights_path)
            for i in range(len(self.model.trainers)):
                self.model.trainers[i].model = _model_weights_list[i]
        return path

    @classmethod
    def load(cls, path: str, reset_paths=False, verbose=True):
        model: MitraModel = super().load(path=path, reset_paths=reset_paths, verbose=verbose)

        if model._weights_saved:
            import torch

            model_weights_list = torch.load(model.weights_path, weights_only=False)  # nosec B614
            for i in range(len(model.model.trainers)):
                model.model.trainers[i].model = model_weights_list[i]
            model._weights_saved = False
        return model

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
        return max(
            cls._estimate_memory_usage_static_cpu_icl(X=X, **kwargs),
            cls._estimate_memory_usage_static_cpu_ft_icl(X=X, **kwargs),
            cls._estimate_memory_usage_static_gpu_cpu(X=X, **kwargs),
            cls._estimate_memory_usage_static_gpu_gpu(X=X, **kwargs),
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
    def _class_tags(cls) -> dict:
        return {
            "can_estimate_memory_usage_static": True,
        }

    def _more_tags(self) -> dict:
        tags = {"can_refit_full": True}
        return tags
