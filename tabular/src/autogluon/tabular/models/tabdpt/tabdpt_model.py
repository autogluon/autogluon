from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@contextmanager
def suppress_tqdm_output():
    saved_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = saved_stderr


# TODO: unit test
# TODO: memory estimate
class TabDPTModel(AbstractModel):
    ag_key = "TABDPT"
    ag_name = "TabDPT"
    seed_name = "seed"
    ag_priority = 50
    default_random_seed = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._predict_hps = None

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
        supported_predict_hps = (
            ("context_size", "permute_classes", "temperature", "n_ensembles")
            if model_cls is TabDPTClassifier
            else ("context_size", "n_ensembles")
        )

        hps = self._get_model_params()
        random_seed = hps.pop(self.seed_name, self.default_random_seed)
        context_size = hps.pop("context_size", None)
        if context_size is None:
            context_size = 1_000_000_000  # set to infinity
        self._predict_hps = {k: v for k, v in hps.items() if k in supported_predict_hps}
        self._predict_hps["seed"] = random_seed
        self._predict_hps["context_size"] = context_size

        X = self.preprocess(X)
        y = y.to_numpy()
        # FIXME: Move defaults elsewhere
        self.model = model_cls(
            device=device,
            inf_batch_size=hps.get("inf_batch_size", 512),
            use_flash=self._use_flash(num_gpus=num_gpus),
            normalizer=hps.get("normalizer", "standard"),
            missing_indicators=hps.get("missing_indicators", False),
            clip_sigma=hps.get("clip_sigma", 4),
            feature_reduction=hps.get("feature_reduction", "pca"),
            faiss_metric=hps.get("faiss_metric", "l2"),
            compile=hps.get("compile", False),
        )
        self.model.fit(X=X, y=y)

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
            with suppress_tqdm_output():
                y_pred = self.model.predict(X, **self._predict_hps)
            return y_pred

        with suppress_tqdm_output():
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

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "refit_folds": True,
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble
