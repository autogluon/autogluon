from __future__ import annotations

import pandas as pd

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage

from .tabpfnv2_5_model import RealTabPFNv25Model


class TabPFN3Model(RealTabPFNv25Model):
    """TabPFN-3 version: https://priorlabs.ai/.

    Requires ``tabpfn>=8.0`` for the v3 checkpoints.

    .. versionadded:: 1.6.0
    """

    ag_key = "TABPFN-3"
    ag_name = "TabPFN-3"

    fixed_random_state: int = 0
    """We found that the validation score is misleading for TabPFN, when one uses a
    different random state for the refit model than for models fit during CV.
    This is because TabPFN's random state determines the preprocessing of TabPFN.
    """

    default_classification_model: str | None = "tabpfn-v3-classifier-v3_default.ckpt"
    default_regression_model: str | None = "tabpfn-v3-regressor-v3_default.ckpt"

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 500_000,
                "max_features": 2500,
                "max_classes": 160,
                # TabPFN-3 batches inference internally once past this many samples.
                "max_batch_size": 150_000,
                "model_telemetry": False,
            }
        )
        return default_auxiliary_params

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        """The list of checkpoints to use for hyperparameter tuning."""
        raise NotImplementedError("We did not benchmark more checkpoints or tuning.")

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Assume a 10 GB baseline (model + activations) plus the dataset memory footprint.

        The v2 layer/embedding heuristic used by the older TabPFN models does not
        transfer to the v3 architecture.
        """
        baseline_mem_est = 10 * 1e9  # 10 GB minimum for TabPFN-3 model + activations
        dataset_mem_est = 5 * get_approximate_df_mem_usage(X).sum()
        return int(baseline_mem_est + dataset_mem_est)
