from __future__ import annotations

import pandas as pd

from .tabpfnv2_5_model import RealTabPFNv25Model


class TabPFNv26Model(RealTabPFNv25Model):
    """TabPFN-2.6 version."""

    ag_key = "TABPFN-2.6"
    ag_name = "TabPFN-2.6"

    fixed_random_state: int = 0
    """We found that the validation score is misleading for TabPFN, when one uses a
    different random state for the refit model than for models fit during CV.
    This is because TabPFN's random state determines the preprocessing of TabPFN
    """

    default_classification_model: str | None = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
    default_regression_model: str | None = "tabpfn-v2.6-regressor-v2.6_default.ckpt"

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 100_000,
                "max_features": 2500,
                "max_classes": 10,
                "model_telemetry": False,
            }
        )
        return default_auxiliary_params

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        """The list of checkpoints to use for hyperparameter tuning."""
        raise NotImplementedError(
            "We did not benchmark more checkpoints or tuning."
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
        # TODO: update, this is not correct anymore, consider using internal TabPFN functions directly.
        features_per_group = 3  # Based on TabPFNv2 default (unused)
        n_layers = 12  # Based on TabPFNv2 default
        embedding_size = 192  # Based on TabPFNv2 default
        dtype_byte_size = 2  # Based on TabPFNv2 default

        model_mem = 14489108  # Based on TabPFNv2 default

        n_samples, n_features = X.shape[0], min(X.shape[1], 500)
        n_feature_groups = n_features / features_per_group + 1  # TODO: Unsure how to calculate this

        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = n_samples * n_feature_groups * embedding_size * n_layers * dtype_byte_size

        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        # Add some buffer to each term + 1 GB overhead to be safe
        return int(model_mem + 4 * X_mem + activation_mem + baseline_overhead_mem_est)
