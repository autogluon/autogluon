from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular import __version__
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


class NoriModel(AbstractTorchModel):
    """
    Nori is a small (~5.5M-parameter) tabular foundation model for regression via
    in-context learning. Given a few labeled context rows it predicts on query rows
    in a single forward pass, with no task-specific training. It is pretrained purely
    on synthetic data.

    Nori is regression-only; it does not support classification.

    Codebase: https://github.com/synthefy/synthefy-nori
    Model: https://huggingface.co/Synthefy/Nori
    License: Apache-2.0

    .. versionadded:: 1.5.0
    """

    ag_key = "NORI"
    ag_name = "Nori"
    ag_priority = 40

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        try:
            from synthefy_nori import NoriRegressor
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import synthefy_nori! To use the Nori model, "
                f"do: `pip install autogluon.tabular[nori]=={__version__}`.",
            )
            raise err

        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        hyp = self._get_model_params()
        # AutoGluon fits many models (per-fold bagging, HPO), so torch.compile's
        # one-time per-process cold compile is not worth it here. Off by default;
        # users can re-enable it via the `compile_model` hyperparameter.
        hyp.setdefault("compile_model", False)

        X = self.preprocess(X, y=y)
        y = y.to_numpy()
        self.model = NoriRegressor(device=device, **hyp)
        self.model.fit(X=X, y=y)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        # Nori is regression-only: `predict` returns point estimates directly.
        return self.model.predict(X)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Nori requires a fully numeric numpy array as input."""
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.to_numpy()

    def get_device(self) -> str:
        return self.model.device

    def _set_device(self, device: str):
        # Nori builds its inner predictor lazily on first predict, reading the device
        # off the regressor. Update the device and drop any cached predictor so the
        # next predict rebuilds on the new device (e.g. GPU -> CPU on load/save).
        self.model.device = device
        self.model._predictor = None

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 0.5 if is_gpu_available else 0,
        }

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                # TODO: Conservative caps for a small in-context-learning model; revisit with benchmarks.
                "max_rows": 50000,
                "max_features": 2000,
            }
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """
        Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": True,  # Better to refit the model for faster inference and similar quality as the bag.
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        # Keep AbstractTorchModel's device-management tags (save on CPU, restore on
        # load) and add static memory estimation.
        tags = super()._class_tags()
        tags["can_estimate_memory_usage_static"] = True
        return tags

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

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
        """Heuristic memory estimate.

        Nori is a small (~5.5M-parameter) in-context-learning model: the whole
        context is held in memory and attended over at predict time, so memory
        scales with (context rows x features). This is a primitive heuristic in the
        spirit of the other tabular-foundation-model wrappers and can be improved.
        """
        dataset_size_mem_est = 3 * get_approximate_df_mem_usage(X).sum()  # roughly 3x DataFrame memory size
        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        model_mem = 25_000_000  # ~5.5M params plus buffers
        embedding_size = 192
        dtype_byte_size = 4

        n_samples, n_features = X.shape[0], min(X.shape[1], 500)
        activation_mem = n_samples * n_features * embedding_size * dtype_byte_size

        model_mem_estimate = (model_mem + 2 * activation_mem) * 1.3  # 30% buffer

        return int(model_mem_estimate + dataset_size_mem_est + baseline_overhead_mem_est)
