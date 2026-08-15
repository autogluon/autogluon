"""
DenseLight tabular NN from LightAutoML (LAMA), integrated as an AutoGluon model.

Architecture adapted from LightAutoML (Apache-2.0):
https://github.com/sb-ai-lab/LightAutoML
``lightautoml/ml_algo/torch_based/nn_models.py`` (DenseLightModel).

Training loop, preprocessing, and bagging are AutoGluon-native (no lightautoml dependency).

Implements the :class:`~autogluon.core.models.AbstractModel` contract from the
`custom model tutorial
<https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html>`_
(``_fit`` / ``_preprocess`` / defaults / auxiliary dtypes), then registers in-tree
as ``DENSELIGHT`` (#4505).
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.tabular import __version__
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

logger = logging.getLogger(__name__)


class DenseLightModel(AbstractTorchModel):
    """
    DenseLight is LightAutoML's popular tabular MLP with input concatenation into
    each hidden block (``concat_input=True``).

    Used on Kaggle as "LAMA NN" for ensemble diversity alongside AutoGluon
    (see issue #4505). This integration vendors the architecture only; it does
    **not** depend on the ``lightautoml`` package.

    Codebase (upstream architecture): https://github.com/sb-ai-lab/LightAutoML
    License: Apache-2.0

    **Usage** (registered key or custom-model class key, same as the tutorial)::

        from autogluon.tabular import TabularPredictor
        from autogluon.tabular.models import DenseLightModel

        # Registered key (after install: pip install "autogluon.tabular[denselight]")
        predictor.fit(train_data, hyperparameters={"DENSELIGHT": {}})

        # Custom-model style (class as hyperparameters key)
        predictor.fit(train_data, hyperparameters={DenseLightModel: {"n_epochs": 50}})

    .. versionadded:: 1.6.0
    """

    ag_key = "DENSELIGHT"
    ag_name = "DenseLight"
    ag_priority = 55
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1

    # Tutorial: restrict to dtypes the model can consume (drop raw text/image paths).
    _default_auxiliary_params_extra = dict(
        valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bool_to_cat = None
        self._features_bool = None

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
            from ._denselight_internal import DenseLightImplementation
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import denselight/torch! To use the DenseLight model, "
                f"do: `pip install autogluon.tabular[denselight]=={__version__}`.",
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

        X = self.preprocess(X, is_train=True, bool_to_cat=bool_to_cat)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.model = DenseLightImplementation(
            n_threads=num_cpus,
            device=device,
            problem_type=self.problem_type,
            early_stopping_metric=self.stopping_metric,
            **hyp,
        )

        remaining = time_limit - (time.time() - start_time) if time_limit is not None else None
        self.model.fit(
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            cat_col_names=X.select_dtypes(include="category").columns.tolist(),
            time_to_fit_in_seconds=remaining,
        )

    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        bool_to_cat: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(required_special_types=["bool"])
        if self._bool_to_cat and self._features_bool:
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

    def _get_default_auxiliary_params(self) -> dict:
        """Mirror the custom-model tutorial: declare valid raw dtypes explicitly."""
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            dict(
                valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
            )
        )
        return default_auxiliary_params

    def _set_default_params(self):
        # Defaults lean toward LAMA DenseLight (hidden_size [512, 750], quantile nums).
        default_params = dict(
            hidden_size=[512, 750],
            drop_rate=0.1,
            concat_input=True,
            use_bn=True,
            dropout_first=True,
            n_epochs=200,
            patience=16,
            lr=1e-3,
            weight_decay=1e-5,
            batch_size="auto",
            eval_batch_size=1024,
            use_quantile=True,
            bool_to_cat=True,
            gradient_clipping_norm=1.0,
        )
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "lr": space.Real(1e-4, 3e-2, log=True),
            "drop_rate": space.Real(0.0, 0.3),
            "batch_size": space.Categorical(64, 128, 256, 512, 1024),
            "hidden_size": space.Categorical([256, 256], [512, 512], [512, 750], [256, 512, 256]),
        }

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Conservative CPU RSS: dataset copies + parameter workspace + torch baseline."""
        if hyperparameters is None:
            hyperparameters = {}
        hidden_size = hyperparameters.get("hidden_size", [512, 750])
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        width = max(hidden_size) if hidden_size else 750
        n_features = max(len(X.columns), 1)
        # Rough param float count: input→h1 + concat path h1→h2 + head
        n_params = n_features * width + (n_features + width) * width + width
        mem_params = 4 * 5 * n_params  # weights, grad, adam×2, best
        dataset_mem = 8 * get_approximate_df_mem_usage(X).sum()
        baseline = 1.6e9
        return int(dataset_mem + 1.2 * mem_params + baseline)

    @classmethod
    def _estimate_gpu_memory_usage_static(
        cls,
        *,
        X,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Peak VRAM estimate (context + activations + params)."""
        if hyperparameters is None:
            hyperparameters = {}
        hidden_size = hyperparameters.get("hidden_size", [512, 750])
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        width = max(hidden_size) if hidden_size else 750
        n_train, n_features = X.shape
        batch_size = hyperparameters.get("batch_size", "auto")
        if batch_size == "auto":
            if n_train < 2800:
                batch_size = 32
            elif n_train < 32000:
                batch_size = 256
            else:
                batch_size = 512
        # CUDA context + params + batch activations
        return int(1.2e9 + 4 * n_features * width * 8 + 4 * int(batch_size) * width * 8)

    @classmethod
    def _class_tags(cls):
        return {
            "reset_torch_threads": True,
        }

    def _more_tags(self) -> dict:
        return {"can_refit_full": False}
