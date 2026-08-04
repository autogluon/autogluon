from __future__ import annotations

from typing import ClassVar

import pandas as pd

from .tabpfnv2_5_model import TabPFNModel


class TabPFNv26Model(TabPFNModel):
    """TabPFN-2.6 version."""

    ag_key = "TABPFN-2.6"
    ag_name = "TabPFN-2.6"
    license_noncommercial: ClassVar[bool] = True

    fixed_random_state: int = 0
    """We found that the validation score is misleading for TabPFN, when one uses a
    different random state for the refit model than for models fit during CV.
    This is because TabPFN's random state determines the preprocessing of TabPFN
    """

    default_classification_model: str | None = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
    default_regression_model: str | None = "tabpfn-v2.6-regressor-v2.6_default.ckpt"

    _default_auxiliary_params_extra = {
        "max_rows": 100_000,
        # No feature cap: TabPFN-2.6 and -3 handle very wide data well, and on BeyondArena's
        # widest tasks (up to 22k columns) they are the two strongest methods of 28. Set to None
        # rather than removed, because the base TabPFNModel (2.5) caps at 2000 and subclass
        # `_default_auxiliary_params_extra` entries are merged base-most first, so an absent key
        # would inherit that tighter cap instead of lifting it. Memory remains bounded by
        # `_estimate_memory_usage_static`, which is what skips a fit that genuinely will not fit.
        "max_features": None,
        "max_classes": 10,
        "model_telemetry": False,
    }

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Peak CPU RSS: a ~2.25 GB process baseline plus ~8.5 float64 copies of the
        train + prediction-batch data made by TabPFN-2.6's preprocessing.

        Calibrated on measured fit+predict RSS across all 51 TabArena tasks
        (1.0-1.33x of measured, no underestimates; categorical-heavy real data
        needs a higher copy count than numeric-only synthetic data) plus synthetic
        sweeps (1k-100k rows, 10-2000 features).
        """
        n_train, n_features = X.shape
        n_test = cls._n_test_for_memory_estimate(n_train=n_train, hyperparameters=hyperparameters)
        baseline_mem_est = 2.25e9
        preprocessing_mem_est = 8.5 * 8 * (n_train + n_test) * n_features
        return int(baseline_mem_est + preprocessing_mem_est)

    @classmethod
    def _estimate_gpu_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        problem_type: str | None = None,
        **kwargs,
    ) -> int:
        """Peak VRAM (reserved + CUDA context) across fit and prediction.

        TabPFN-2.6 materializes the joint train + prediction-batch sequence, so peak
        VRAM is symmetric in total rows. The per-cell (row x feature) cost is
        piecewise: ~3.8 KB/cell while tabpfn runs full activations, dropping ~5x
        once its memory-saving mode kicks in above a total-cell threshold (see
        ``tabpfn.architectures.base.memory.should_save_peak_mem``: ~6M cells on an
        80 GB device, scaled by free VRAM; 5M is used below as a conservative knee).
        Features count to ~300, with a shallow slope to the hard cap at ~1000. Rows
        also carry a feature-independent cost (~25 KB, ~40 KB for regression's
        distributional output), and regression costs ~2.5x overall. Peak assumes
        default-sized ensembles (``n_estimators=1`` peaks ~1.8x lower). Calibrated
        on all 51 TabArena tasks (1.02-2.8x of measured at actual prediction sizes,
        no underestimates) plus synthetic sweeps (1k-100k train rows, up to 100k
        prediction rows, 10-2000 features).
        """
        n_train, n_features = X.shape
        n_test = cls._n_test_for_memory_estimate(n_train=n_train, hyperparameters=hyperparameters)
        total_rows = n_train + n_test
        is_regression = problem_type == "regression"
        regression_multiplier = 2.5 if is_regression else 1.0
        cells = total_rows * min(n_features, 300)
        memory_saving_knee = 5e6
        return int(
            0.85e9  # CUDA context + model weights floor
            + regression_multiplier
            * (
                (40e3 if is_regression else 25e3) * total_rows
                + 3.8e3 * min(cells, memory_saving_knee)
                + 0.8e3 * max(cells - memory_saving_knee, 0)
                + 0.4e3 * total_rows * max(0, min(n_features, 1000) - 300)
            )
        )

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        """The list of checkpoints to use for hyperparameter tuning."""
        raise NotImplementedError("We did not benchmark more checkpoints or tuning.")
