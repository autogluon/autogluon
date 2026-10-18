"""
Code Adapted from TabArena: https://github.com/autogluon/tabarena/blob/main/tabarena/tabarena/benchmark/models/ag/tabm/tabm_model.py
Partially adapted from pytabkit's TabM implementation.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from autogluon.tabular import __version__
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

logger = logging.getLogger(__name__)


class TabMModel(AbstractTorchModel):
    """
    TabM is an efficient ensemble of MLPs that is trained simultaneously with mostly shared parameters.

    TabM is one of the top performing methods overall on TabArena-v0.1: https://tabarena.ai

    Paper: TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling
    Authors: Yury Gorishniy, Akim Kotelnikov, Artem Babenko
    Codebase: https://github.com/yandex-research/tabm
    License: Apache-2.0

    Partially adapted from pytabkit's TabM implementation.

    .. versionadded:: 1.4.0
    """

    ag_key = "TABM"
    ag_name = "TabM"
    ag_priority = 85
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]

    _MAX_FEATURES_FOR_MEMORY_ESTIMATE: int = 2000
    """Feature count past which the per-feature memory cost saturates (batches shrink
    and the embedding layers stop dominating), so memory estimates stop scaling."""

    _default_auxiliary_params_extra = {
        "max_batch_size": 16384,  # avoid excessive VRAM usage
    }
    default_resources_physical_cores_only = True
    default_num_gpus = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._features_bool = None
        self._bool_to_cat = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ):
        start_time = time.time()

        try:
            # imports various dependencies such as torch
            from ._tabm_internal import TabMImplementation
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import tabm! To use the TabM model, "
                f"do: `pip install autogluon.tabular[tabm]=={__version__}`.",
            )
            raise err

        device = self._resolve_fit_device(num_gpus=num_gpus)

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X, X_val, y, y_val = generate_train_test_split(
                X=X,
                y=y,
                problem_type=self.problem_type,
                test_size=0.2,
                random_state=0,
            )

        hyp = self._get_model_params()
        bool_to_cat = hyp.pop("bool_to_cat", True)

        X = self.preprocess(X, y=y, is_train=True, bool_to_cat=bool_to_cat)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.model = TabMImplementation(
            n_threads=num_cpus,
            device=device,
            problem_type=self.problem_type,
            early_stopping_metric=self.stopping_metric,
            **hyp,
        )

        self.model.fit(
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            cat_col_names=X.select_dtypes(include="category").columns.tolist(),
            time_to_fit_in_seconds=time_limit - (time.time() - start_time) if time_limit is not None else None,
        )

    # FIXME: bool_to_cat is a hack: Maybe move to abstract model?
    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        bool_to_cat: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(required_special_types=["bool"])
        if self._bool_to_cat and self._features_bool:
            # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
            X = X.copy(deep=True)
            X[self._features_bool] = X[self._features_bool].astype("category")

        return X

    def get_device(self) -> str:
        return self.model.device_.type

    def _set_device(self, device: str):
        device = self.to_torch_device(device)
        self.model.device_ = device
        self.model.model_ = self.model.model_.to(device)

    def _get_default_stopping_metric(self):
        return self.eval_metric

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict = None,
        num_classes: int | None = 1,
        **kwargs,
    ) -> int:
        """Peak CPU RSS: the model's parameter + forward/backward footprint (which
        lives in RAM for a CPU fit) on top of a torch-process baseline.

        The parameter and activation terms scale with the feature count, which is
        capped for estimation the same way as in the GPU estimate: past a few
        thousand features the per-feature cost saturates rather than growing
        linearly (uncapped, a 22k-feature frame estimates >300 GB for a model that
        fits in 30). Validated on all 51 TabArena plus 69 BeyondArena tasks
        (1.1-17x of measured GPU-fit RSS, no underestimates); the upper end is
        wide data, where a CPU fit would genuinely hold the activations this
        estimate covers while a GPU fit keeps them in VRAM.
        """
        if num_classes is None:
            num_classes = 1
        if hyperparameters is None:
            hyperparameters = {}

        cat_sizes = []
        for col in X.select_dtypes(include=["category", "object"]):
            if isinstance(X[col], pd.CategoricalDtype):
                # Use .cat.codes for category dtype
                unique_codes = X[col].cat.codes.unique()
            else:
                # For object dtype, treat unique strings as codes
                unique_codes = X[col].astype("category").cat.codes.unique()
            cat_sizes.append(len(unique_codes))

        n_numerical = len(X.select_dtypes(include=["number"]).columns)

        n_features_eff = n_numerical + sum(cat_sizes)
        if n_features_eff > cls._MAX_FEATURES_FOR_MEMORY_ESTIMATE:
            scale = cls._MAX_FEATURES_FOR_MEMORY_ESTIMATE / n_features_eff
            n_numerical = int(n_numerical * scale)
            cat_sizes = [int(size * scale) for size in cat_sizes]

        return cls._estimate_tabm_ram(
            hyperparameters=hyperparameters,
            n_numerical=n_numerical,
            cat_sizes=cat_sizes,
            n_classes=num_classes,
            n_samples=len(X),
        )

    @classmethod
    def _estimate_tabm_ram(
        cls,
        hyperparameters: dict,
        n_numerical: int,
        cat_sizes: list[int],
        n_classes: int,
        n_samples: int,
    ) -> int:
        num_emb_n_bins = hyperparameters.get("num_emb_n_bins", 48)
        d_embedding = hyperparameters.get("d_embedding", 16)
        d_block = hyperparameters.get("d_block", 512)
        # not completely sure if this is hidden blocks or all blocks, taking the safe option below
        n_blocks = hyperparameters.get("n_blocks", "auto")
        if isinstance(n_blocks, str) and n_blocks == "auto":
            n_blocks = 3
        batch_size = hyperparameters.get("batch_size", "auto")
        if isinstance(batch_size, str) and batch_size == "auto":
            batch_size = cls.get_tabm_auto_batch_size(n_samples=n_samples)
        tabm_k = hyperparameters.get("tabm_k", 32)
        predict_batch_size = hyperparameters.get("eval_batch_size", 1024)

        # not completely sure
        n_params_num_emb = n_numerical * (num_emb_n_bins + 1) * d_embedding
        n_params_mlp = (
            (n_numerical + sum(cat_sizes)) * d_embedding * (d_block + tabm_k)
            + (n_blocks - 1) * d_block**2
            + n_blocks * d_block
            + d_block * (1 + max(1, n_classes))
        )
        # 4 bytes per float, up to 5 copies of parameters (1 standard, 1 .grad, 2 adam, 1 best_epoch)
        mem_params = 4 * 5 * (n_params_num_emb + n_params_mlp)

        # compute number of floats in forward pass (per batch element)
        # todo: numerical embedding layer (not sure if this is entirely correct)
        n_floats_forward = n_numerical * (num_emb_n_bins + d_embedding)
        # before and after scale
        n_floats_forward += 2 * (sum(cat_sizes) + n_numerical * d_embedding)
        # 2 for pre-act, post-act
        n_floats_forward += n_blocks * 2 * d_block + 2 * max(1, n_classes)
        # 2 for forward and backward, 4 bytes per float
        mem_forward_backward = 4 * max(batch_size * 2, predict_batch_size) * n_floats_forward * tabm_k
        # * 8 is pessimistic for the long tensors in the forward pass, 4 would probably suffice

        mem_ds = n_samples * (4 * n_numerical + 8 * len(cat_sizes))

        # some safety constants and offsets (the 5 is probably excessive); the
        # baseline is the measured torch-process floor (~1.6 GB), which dominates
        # on small datasets.
        mem_total = 5 * mem_ds + 1.2 * mem_forward_backward + 1.2 * mem_params + 1.6e9

        return mem_total

    @classmethod
    def get_tabm_auto_batch_size(cls, n_samples: int) -> int:
        # by Yury Gorishniy, inferred from the choices in the TabM paper.
        if n_samples < 2_800:
            return 32
        if n_samples < 4_500:
            return 64
        if n_samples < 6_400:
            return 128
        if n_samples < 32_000:
            return 256
        if n_samples < 108_000:
            return 512
        return 1024

    @classmethod
    def _estimate_gpu_memory_usage_static(
        cls,
        *,
        X,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Peak VRAM (reserved + CUDA context) across fit and prediction.

        TabM's peak has three drivers: k-ensemble embedding layers (~10.5 MB per
        feature), each categorical level (~1 MB of embedding parameters + optimizer
        state), and a per-cell (row x feature) term that saturates — ~140 B/cell up
        to ~100M cells, then ~40 B/cell, since past that point batches cover a
        shrinking fraction of the data and activations stop scaling with it.
        Numerical columns with missing values count as an extra feature each (the
        wrapper adds an indicator column per such column), and the feature count is
        capped for estimation (the per-feature cost stops growing past a few
        thousand). Calibrated on synthetic fit+predict measurements (1k-1M rows,
        10-1000 features) and all 51 TabArena plus 85 BeyondArena tasks spanning 100
        to 1M rows (1.0-7.3x, no underestimates; high-cardinality categoricals,
        missing-heavy data, and gene-expression frames up to 22k features). Epoch
        count adds time, not peak memory (SGD steady state).
        """
        n_train, n_features = X.shape
        numeric = X.select_dtypes(include=["number"])
        n_features_eff = min(n_features + int(numeric.isna().any().sum()), cls._MAX_FEATURES_FOR_MEMORY_ESTIMATE)
        sum_cat_levels = int(sum(X[col].nunique() for col in X.select_dtypes(include=["category", "object"]).columns))
        n_cells = n_train * n_features_eff
        cell_saturation = 100e6
        return int(
            1.2e9
            + 10.5e6 * n_features_eff
            + 140 * min(n_cells, cell_saturation)
            + 40 * max(n_cells - cell_saturation, 0)
            + 1.0e6 * sum_cat_levels
        )

    @classmethod
    def _class_tags(cls):
        return {
            "reset_torch_threads": True,
        }

    def _more_tags(self) -> dict:
        # TODO: Need to add train params support, track best epoch
        #  How to force stopping at a specific epoch?
        return {"can_refit_full": False}
