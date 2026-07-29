from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class NoriModel(AbstractTorchModel):
    """
    Nori is a tabular foundation model for regression via in-context learning.
    Given a few labeled context rows it predicts on query rows in a single forward
    pass, with no task-specific training. It is pretrained purely on synthetic data.

    Nori is regression-only; it does not support classification.

    Model size variants are selected with the ``model`` hyperparameter, which is
    forwarded to ``NoriRegressor``:

    - ``"nori"`` (default): the base ~6M-parameter checkpoint (``Synthefy/Nori``).
    - ``"nori-30m"``: the larger ~30M-parameter checkpoint (``Synthefy/Nori-30M``).

    For example::

        predictor.fit(..., hyperparameters={NoriModel: {"model": "nori-30m"}})

    Codebase: https://github.com/synthefy/synthefy-nori
    Model: https://huggingface.co/Synthefy/Nori (base), https://huggingface.co/Synthefy/Nori-30M (30M)
    License: Apache-2.0

    .. versionadded:: 1.6.0
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
                "\tFailed to import synthefy_nori! To use the Nori model, do: `pip install synthefy-nori`.",
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
        hyp.pop("device", None)  # device is set explicitly from the allocated resources

        X = self.preprocess(X, y=y)
        y = y.to_numpy()
        self.model = NoriRegressor(device=device, **hyp)
        self.model.fit(X=X, y=y)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        # Nori is regression-only: `predict` returns point estimates directly.
        return self.model.predict(X)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Nori requires a fully numeric array as input; cast to float32 (the dtype
        Nori coerces to internally) with NaN preserved (handled natively)."""
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return np.asarray(X.to_numpy(), dtype=np.float32)

    def get_device(self) -> str:
        # NoriRegressor's device may be None (auto) or a torch.device; normalize to str.
        device = self.model.device
        if device is None:
            return "cpu"
        return device if isinstance(device, str) else device.type

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
                # Nori attends queries over the full context with no internal chunking:
                # measured predict-phase VRAM is ~5 GB at 10k rows x 10 features,
                # ~21 GB at 10k x 100, and ~58 GB at the 50k-row cap (100 features),
                # so the cap only fits on high-memory GPUs.
                "max_rows": 50000,
                "max_features": 2000,
                # Chunk prediction: an unchunked 50k-query predict against a 50k-row
                # context fails with a CUDA kernel-configuration error (after ~76 GB);
                # 10k-query chunks on the same context run fine.
                "max_batch_size": 10000,
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
        """CPU memory estimate: a ~3 GB process baseline (torch + model + inference
        buffers) plus the dataset footprint.

        Calibrated on measured fit+predict RSS (2.3-3.1 GB across 10k-50k rows,
        10-1000 features): the baseline dominates and the per-cell term is small,
        as Nori's activations live on the GPU.
        """
        baseline_mem_est = 3e9
        dataset_mem_est = 5 * get_approximate_df_mem_usage(X).sum()
        return int(baseline_mem_est + dataset_mem_est)
