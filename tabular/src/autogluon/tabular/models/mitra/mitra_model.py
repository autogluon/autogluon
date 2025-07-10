import pandas as pd
from typing import Optional, List
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel


# TODO: Needs memory usage estimate method
class MitraModel(AbstractModel):
    ag_key = "MITRA"
    ag_name = "Mitra"

    def __init__(self, problem_type=None, **kwargs):
        super().__init__(**kwargs)
        self.problem_type = problem_type

    def get_model_cls(self):
        from .sklearn_interface import MitraClassifier
        if self.problem_type in ['binary', 'multiclass']:
            model_cls = MitraClassifier
        elif self.problem_type == 'regression':
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

    def _set_default_params(self):
        default_params = {
            "device": "cuda", # "cpu"
            "n_estimators": 1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

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
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 1
        return num_cpus, num_gpus

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return self.estimate_memory_usage_static(X=X, problem_type=self.problem_type, num_classes=self.num_classes, **kwargs)
    
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
        cpu_memory_kb = 1.3 * (0.001748 * (X.shape[0]**2) * X.shape[1] + \
                        0.001206 * X.shape[0] * (X.shape[1]**2) + \
                        10.3482 * X.shape[0] * X.shape[1] + \
                        6409698)
        return int(cpu_memory_kb * 1e3)

    @classmethod
    def _estimate_memory_usage_static_cpu_ft_icl(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        cpu_memory_kb = 1.3 * (0.001 * (X.shape[0]**2) * X.shape[1] + \
                        0.004541 * X.shape[0] * (X.shape[1]**2) + \
                        46.2974 * X.shape[0] * X.shape[1] + \
                        5605681)
        return int(cpu_memory_kb * 1e3)
    
    @classmethod
    def _estimate_memory_usage_static_gpu_cpu(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        return int(5 * 1e9)

    @classmethod
    def _estimate_memory_usage_static_gpu_gpu(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        gpu_memory_mb = 1.3 * (0.05676 * X.shape[0] * X.shape[1] + 3901)
        return int(gpu_memory_mb * 1e6)

    @classmethod
    def _class_tags(cls) -> dict:
        return {
            "can_estimate_memory_usage_static": True,
        }
    
    def _more_tags(self) -> dict:
        tags = {"can_refit_full": True}
        return tags
