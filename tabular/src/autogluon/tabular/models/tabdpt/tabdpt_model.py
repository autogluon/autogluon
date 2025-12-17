from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


# FIXME: Nick:
#  TODO: batch_size is linear to memory usage
#   512 default
#   should be less for very large datasets
#   128 batch_size on Bioresponse -> 12 GB VRAM
#       Train Data Rows:    2500
#       Train Data Columns: 1776
#       Problem Type:       binary
#  FIXME: Just set context_size = infinity, everything is way faster, memory usage is way lower, etc.
#   Train Data Rows:    100000
#   Train Data Columns: 10
#   binary
#   only takes 6.7 GB during inference with batch_size = 512
# FIXME: Make it work when loading on CPU?
# FIXME: Can we run 8 in parallel to speed up?
# TODO: clip_sigma == 1 is terrible, clip_sigma == 16 maybe very good? What about higher values?
#  clip_sigma >= 16 is roughly all equivalent
# FIXME: TabDPT stores self.X_test for no reason
# FIXME: TabDPT creates faiss_knn even if it is never used. Better if `context_size=None` means it is never created.
# TODO: unit test
# TODO: memory estimate
class TabDPTModel(AbstractTorchModel):
    ag_key = "TABDPT"
    ag_name = "TabDPT"
    seed_name = "seed"
    ag_priority = 50
    default_random_seed = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._predict_hps = None
        self._use_flash_og = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )
        from tabdpt import TabDPTClassifier, TabDPTRegressor

        model_cls = (
            TabDPTClassifier
            if self.problem_type in [BINARY, MULTICLASS]
            else TabDPTRegressor
        )
        fit_params, self._predict_hps = self._get_tabdpt_params(num_gpus=num_gpus)

        X = self.preprocess(X)
        y = y.to_numpy()
        self.model = model_cls(
            device=device,
            **fit_params,
        )
        self.model.fit(X=X, y=y)

    def _get_tabdpt_params(self, num_gpus: float) -> tuple[dict, dict]:
        model_params = self._get_model_params()

        valid_predict_params = (self.seed_name, "context_size", "permute_classes", "temperature", "n_ensembles")

        predict_params = {}
        for hp in valid_predict_params:
            if hp in model_params:
                predict_params[hp] = model_params.pop(hp)
        predict_params.setdefault(self.seed_name, self.default_random_seed)
        predict_params.setdefault("context_size", None)

        supported_predict_params = (
            (self.seed_name, "context_size", "n_ensembles", "permute_classes", "temperature")
            if self.problem_type in [BINARY, MULTICLASS]
            else (self.seed_name, "context_size", "n_ensembles")
        )
        predict_params = {key: val for key, val in predict_params.items() if key in supported_predict_params}

        fit_params = model_params

        fit_params.setdefault("verbose", False)
        fit_params.setdefault("compile", False)
        if fit_params.get("use_flash", True):
            fit_params["use_flash"] = self._use_flash(num_gpus=num_gpus)
        return fit_params, predict_params

    @staticmethod
    def _use_flash(num_gpus: float) -> bool:
        """Detect if torch's native flash attention is available on the current machine."""
        if num_gpus == 0:
            return False

        import torch

        if not torch.cuda.is_available():
            return False

        device = torch.device("cuda:0")
        capability = torch.cuda.get_device_capability(device)

        return capability != (7, 5)

    def _post_fit(self, **kwargs):
        super()._post_fit(**kwargs)
        self._use_flash_og = self.model.use_flash
        return self

    def get_device(self) -> str:
        return self.model.device

    def _set_device(self, device: str):
        self.model.to(device)
        if device == "cpu":
            self.model.use_flash = False
            self.model.model.use_flash = False
        else:
            self.model.use_flash = self._use_flash_og
            self.model.model.use_flash = self._use_flash_og

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def get_minimum_resources(
        self, is_gpu_available: bool = False
    ) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 0.5 if is_gpu_available else 0,
        }

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type in [REGRESSION]:
            y_pred = self.model.predict(X, **self._predict_hps)
            return y_pred

        y_pred_proba = self.model.ensemble_predict_proba(X, **self._predict_hps)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """TabDPT requires numpy array as input."""
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X
            )
        return X.to_numpy()

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 100000,  # TODO: Test >100k rows
                "max_features": 2500,  # TODO: Test >2500 features
                "max_classes": 10,  # TODO: Test >10 classes
            }
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "refit_folds": True,
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble
