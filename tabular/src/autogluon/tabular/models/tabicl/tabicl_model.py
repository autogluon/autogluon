"""
Code Adapted from TabArena: https://github.com/autogluon/tabarena/blob/main/tabarena/tabarena/benchmark/models/ag/tabicl/tabicl_model.py
"""

from __future__ import annotations

import logging

import pandas as pd

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular import __version__
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

logger = logging.getLogger(__name__)


# TODO: Verify if crashes when weights are not yet downloaded and fit in parallel
class TabICLModel(AbstractTorchModel):
    """
    TabICL is a foundation model for tabular data using in-context learning
    that is scalable to larger datasets than TabPFNv2. It is pretrained purely on synthetic data.
    TabICL currently only supports classification tasks.

    TabICL is one of the top performing methods overall on TabArena-v0.1: https://tabarena.ai

    Paper: TabICL: A Tabular Foundation Model for In-Context Learning on Large Data
    Authors: Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan
    Codebase: https://github.com/soda-inria/tabicl
    License: BSD-3-Clause

    .. versionadded:: 1.4.0
    """

    ag_key = "TABICL"
    ag_name = "TabICL"
    ag_priority = 65
    seed_name = "random_state"

    def get_model_cls(self):
        from tabicl import TabICLClassifier

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = TabICLClassifier
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")
        return model_cls

    @staticmethod
    def _get_batch_size(n_cells: int):
        if n_cells <= 4_000_000:
            return 8
        elif n_cells <= 6_000_000:
            return 4
        else:
            return 2

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        try:
            import tabicl
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import tabicl! To use the TabICL model, "
                f"do: `pip install autogluon.tabular[tabicl]=={__version__}`.",
            )
            raise err

        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        model_cls = self.get_model_cls()
        hyp = self._get_model_params()
        hyp["batch_size"] = hyp.get("batch_size", self._get_batch_size(X.shape[0] * X.shape[1]))
        self.model = model_cls(
            **hyp,
            device=device,
            n_jobs=num_cpus,
        )
        X = self.preprocess(X)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def get_device(self) -> str:
        return self.model.device_.type

    # TODO: Better to have an official TabICL method for this
    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model.device_ = device
        self.model.device = self.model.device_.type
        self.model.model_ = self.model.model_.to(self.model.device_)
        self.model.inference_config_.COL_CONFIG.device = self.model.device_
        self.model.inference_config_.ROW_CONFIG.device = self.model.device_
        self.model.inference_config_.ICL_CONFIG.device = self.model.device_

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 30000,
                "max_features": 2000,
            }
        )
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

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
        hyperparameters: dict = None,
        **kwargs,
    ) -> int:
        """
        Heuristic memory estimate that is very primitive.
        Can be vastly improved.
        """
        if hyperparameters is None:
            hyperparameters = {}

        dataset_size_mem_est = 3 * get_approximate_df_mem_usage(X).sum()  # roughly 3x DataFrame memory size
        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        n_rows = X.shape[0]
        n_features = X.shape[1]
        batch_size = hyperparameters.get("batch_size", cls._get_batch_size(X.shape[0] * X.shape[1]))
        embedding_dim = 128
        bytes_per_float = 4
        model_mem_estimate = 2 * batch_size * embedding_dim * bytes_per_float * (4 + n_rows) * n_features

        model_mem_estimate *= 1.3  # add 30% buffer

        # TODO: Observed memory spikes above expected values on large datasets, increasing mem estimate to compensate
        model_mem_estimate *= 2.0  # Note: 1.5 is not large enough, still gets OOM

        mem_estimate = model_mem_estimate + dataset_size_mem_est + baseline_overhead_mem_est

        return mem_estimate

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """
        Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            # FIXME: If parallel, uses way more memory, seems to behave incorrectly, so we force sequential.
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": True,  # Better to refit the model for faster inference and similar quality as the bag.
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
