"""
Code Adapted from TabArena: https://github.com/autogluon/tabarena/blob/main/tabarena/tabarena/benchmark/models/ag/tabicl/tabicl_model.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.tabular import __version__
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

logger = logging.getLogger(__name__)


# TODO: Verify if crashes when weights are not yet downloaded and fit in parallel
class TabICLModel(AbstractTorchModel):
    """
    TabICL is a foundation model for tabular data using in-context learning
    that is scalable to larger datasets than TabPFNv2. It is pretrained purely on synthetic data.

    The default TabICL version used is TabICLv2.

    TabICL is one of the top performing methods overall on TabArena-v0.1: https://tabarena.ai
    TabICLv2 significantly improves upon TabICLv1, and achieves very strong performance on TabArena.

    Paper: TabICL: A Tabular Foundation Model for In-Context Learning on Large Data
    Authors: Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan

    Paper: TabICLv2: A better, faster, scalable, and open tabular foundation model
    Authors: Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan

    Codebase: https://github.com/soda-inria/tabicl
    License: BSD-3-Clause

    .. versionadded:: 1.4.0
    """

    gpu_strongly_recommended: bool = True  # in-context inference is 12-63x slower on CPU
    ag_key = "TABICL"
    ag_name = "TabICL"

    default_classification_model: str | None = "tabicl-classifier-v2-20260212.ckpt"
    default_regression_model: str | None = "tabicl-regressor-v2-20260212.ckpt"

    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression", "quantile"]

    _default_auxiliary_params_extra = {
        # TODO: Instead of caps, should we subsample for large datasets?
        # Measured peak VRAM is already ~83 GB at 100k rows x 100 features, so a 1M-row fit is far
        # outside what a single GPU serves; 500k keeps the cap within reach of a large card.
        "max_rows": 500000,
        "max_features": 2000,  # TODO: What should be the cap? 10k features works, but unsure if it is good
        # No prediction chunking: tabicl batches internally (VRAM-adaptive with
        # OOM-halving), and each external chunk would re-encode the full training
        # context (kv_cache is off by default) — a 1024-row cap made a 100k-row
        # predict ~100x slower while saving no VRAM.
        "max_batch_size": None,
    }
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,  # Better to refit the model for faster inference and similar quality as the bag.
    }
    """Set fold_fitting_strategy to sequential_local, as parallel folding crashes if model weights aren't pre-downloaded."""
    default_resources_physical_cores_only = True
    default_num_gpus = 1

    def get_model_cls(self):
        if self.problem_type in ["binary", "multiclass"]:
            from tabicl import TabICLClassifier

            model_cls = TabICLClassifier
        else:
            from tabicl import TabICLRegressor

            model_cls = TabICLRegressor
        return model_cls

    @staticmethod
    def _get_batch_size(n_cells: int):
        """Datasets-per-batch for tabicl's internal batching, by table size in cells.

        tabicl halves the batch on OOM, so these are starting points rather than hard limits; the
        tiers exist to avoid paying for a failed attempt first. The smallest tier matters because
        the column-embedding output, shape ``(batch_size, n_rows, n_columns, embed_dim)``, is
        tabicl's dominant allocation, so on the largest tables even 2 is too many.
        """
        if n_cells <= 4_000_000:
            return 8
        if n_cells <= 6_000_000:
            return 4
        if n_cells <= 500_000_000:
            return 2
        return 1

    def get_checkpoint_version(self, hyperparameter: dict) -> str:
        clf_checkpoint = self.default_classification_model
        reg_checkpoint = self.default_regression_model

        # Resolve HPO
        if "checkpoint_version" in hyperparameter:
            if isinstance(hyperparameter["checkpoint_version"], str):
                clf_checkpoint = hyperparameter["checkpoint_version"]
                reg_checkpoint = hyperparameter["checkpoint_version"]
            elif isinstance(hyperparameter["checkpoint_version"], tuple):
                clf_checkpoint = hyperparameter["checkpoint_version"][0]
                reg_checkpoint = hyperparameter["checkpoint_version"][1]
            else:
                raise ValueError(
                    "checkpoint_version hyperparameter must be either a string or a tuple of two strings (clf, reg)."
                )

        if self.problem_type in ["binary", "multiclass"]:
            return clf_checkpoint

        return reg_checkpoint

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        """Cast `category` columns that are entirely missing in a prediction batch to object dtype.

        Temporary workaround for https://github.com/soda-inria/tabicl/issues/143 (tabicl<=2.1.1):
        tabicl masks features that are all-NaN in the batch and fills them with the float 0.0, which
        poisons a `category` column, and its fitted `OrdinalEncoder` then crashes
        (`TypeError: ufunc 'isnan' not supported`). Happens whenever a categorical column is entirely
        missing within one prediction batch, e.g. an out-of-fold validation split during bagging. As
        an object column the same fill is dtype-legal and the 0.0 values go down the encoder's
        unknown-value path, which is the behavior tabicl's masking intends. The columns are all-NaN,
        so the cast loses nothing. Remove once the pinned tabicl carries the upstream fix.
        """
        X = super()._preprocess(X, **kwargs)
        if not is_train:
            all_nan_categoricals = [
                col
                for col, dtype in X.dtypes.items()
                if isinstance(dtype, pd.CategoricalDtype) and X[col].isna().all()
            ]
            if all_nan_categoricals:
                X = X.copy(deep=False)
                for col in all_nan_categoricals:
                    X[col] = pd.Series(np.nan, index=X.index, dtype=object)
        return X

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

        device = self._resolve_fit_device(num_gpus=num_gpus)

        model_cls = self.get_model_cls()
        hyp = self._get_model_params()
        hyp["batch_size"] = hyp.get("batch_size", self._get_batch_size(X.shape[0] * X.shape[1]))
        # Pin the checkpoint rather than inheriting whatever the installed tabicl defaults to, and
        # resolve the per-problem-type form of the `checkpoint_version` hyperparameter (a bare
        # string, or a `(classification, regression)` tuple) that the library itself does not accept.
        hyp["checkpoint_version"] = self.get_checkpoint_version(hyperparameter=hyp)
        self.model = model_cls(
            **hyp,
            device=device,
            n_jobs=num_cpus,
        )
        X = self.preprocess(X, y=y, is_train=True)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        if self.problem_type == "quantile":
            X = self.preprocess(X, **kwargs)
            return np.asarray(self.model.predict(X, output_type="quantiles", alphas=self.quantile_levels))
        return super()._predict_proba(X=X, **kwargs)

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

        # FIXME: Likely this is overly conservative now, figure out more accurate memory estimate for TabICLv2
        #  Early testing shows that cutting this in half is safe.
        # TODO: Observed memory spikes above expected values on large datasets, increasing mem estimate to compensate
        model_mem_estimate *= 2.0  # Note: 1.5 is not large enough, still gets OOM

        mem_estimate = model_mem_estimate + dataset_size_mem_est + baseline_overhead_mem_est

        return mem_estimate

    @classmethod
    def _estimate_gpu_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict = None,
        **kwargs,
    ) -> int:
        """Minimum VRAM required across fit and prediction — NOT expected usage.

        tabicl plans its internal batches from free device memory, so its usage is
        opportunistic: the same task that reserves ~30 GB on an idle large GPU
        completes in under 1.5 GB of free VRAM at ~2x the runtime. Estimating usage
        instead of requirement would needlessly skip the model on busy/small devices.
        Requirement floors measured on synthetic fit+predict tasks: ~1 GB at 2M total
        cells (train + prediction rows x features), ~5 GB at 20M. The prediction-row
        count is unknown at fit time; assume at least 100k.

        Caveat: the plan is made from free memory at fit/predict start — VRAM claimed
        by other processes afterwards can still cause a hard OOM.
        """
        n_train, n_features = X.shape
        n_test = max(100_000, n_train)
        total_cells = (n_train + n_test) * n_features
        return int(0.7e9 + 250 * total_cells)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
