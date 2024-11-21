from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

logger = logging.getLogger(__name__)

# TODO: Add huggingface weights download support
class TabPFNMixModel(AbstractModel):
    """
    [Experimental model] Can be changed/removed without warning in future releases.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _get_model_type(self):
        from ._internal.tabpfnmix_classifier import TabPFNMixClassifier
        if self.problem_type in ['binary', 'multiclass']:
            model_cls = TabPFNMixClassifier
        else:
            # FIXME: Add regression support
            raise AssertionError(f"TabPFN does not support problem_type='{self.problem_type}'")
        return model_cls

    def _set_default_params(self):
        """Specifies hyperparameter values to use by default"""
        default_params = {
            # most important hyperparameters. Only set `n_estimators>1` if `max_epochs>1`, else there will be no benefit.
            # path_weights,  # most important
            # path_weights_classifier,  # if specified, overrides path_weights for classification problems
            # path_weights_regressor,  # if specified, overrides path_weights for regression problems
            "n_ensembles": 1,  # FIXME: RENAME: n_estimators
            "max_epochs": 0,  # fine-tuning epochs. Will do pure in-context learning if 0.

            # next most important hyperparameters
            "lr": 1.0e-05,
            "max_samples_query": 1024,  # larger = slower but better quality on datasets with at least this many validation samples
            "max_samples_support": 8196,  # larger = slower but better quality on datasets with at least this many training samples

            # other hyperparameters
            "early_stopping_patience": 40,  # TODO: Figure out optimal value
            "linear_attention": True,
            "lr_scheduler": False,
            "lr_scheduler_patience": 30,
            "optimizer": "adamw",
            "use_feature_count_scaling": True,
            "use_quantile_transformer": True,
            "weight_decay": 0,
            "y_as_float_embedding": True,

            # architecture hyperparameters, recommended to keep as default unless using a custom pre-trained backbone
            "n_classes": 10,
            "n_features": 100,
            "n_heads": 4,
            "n_layers": 12,
            "attn_dropout": 0.0,
            "dim": 512,

            # utility parameters, recommended to keep as default
            "split_val": False,
            "use_best_epoch": True,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # FIXME: Make X_val, y_val = None work well, currently it uses X and y as validation, should instead skip validation entirely
    # FIXME: Use `params_trained` for refit optimal epochs
    # FIXME: Handle model weights download
    # FIXME: GPU support?
    # FIXME: Save model weights to file instead of pickling?
    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, num_cpus: int = 1, num_gpus: float = 0, **kwargs):
        try_import_torch()
        import torch
        from ._internal.config.config_run import ConfigRun

        ag_params = self._get_ag_params()
        max_classes = ag_params.get("max_classes")
        if max_classes is not None and self.num_classes > max_classes:
            # TODO: Move to earlier stage when problem_type is checked
            raise AssertionError(f"Max allowed classes for the model is {max_classes}, " f"but found {self.num_classes} classes.")

        params = self._get_model_params()

        from ._internal.core.enums import Task
        if self.problem_type in [REGRESSION, QUANTILE]:
            task = Task.REGRESSION
            n_classes = 0
        else:
            task = Task.CLASSIFICATION
            n_classes = self.num_classes

        if num_gpus > 0:
            device = "cuda:0"
        else:
            device = "cpu"

        path_weights = None
        if task == Task.CLASSIFICATION:
            if "path_weights_classifier" in params and params["path_weights_classifier"] is not None:
                path_weights = Path(params["path_weights_classifier"])
        elif task == Task.REGRESSION:
            if "path_weights_regressor" in params and params["path_weights_regressor"] is not None:
                path_weights = Path(params["path_weights_regressor"])
        if path_weights is None:
            if "path_weights" in params and params["path_weights"] is not None:
                path_weights = Path(params["path_weights"])

        if path_weights is None:
            logger.log(15, "\tNo path_weights specified, fitting model from random initialization...")
        else:
            logger.log(15, f'\tLoading pre-trained weights from file... (path_weights="{path_weights}")')

        # FIXME: Note: Disabling path_config logic for v1.2 release, as it would be the only model to support this and it might cause bugs
        # FIXME: Don't require loading from file, allow user to specify everything?
        # TODO: Not a big fan of the path config logic due to possible portability issues
        # if "path_config" in params:
        #     path_config = Path(params["path_config"])
        # else:
        #     path_config = None
        # if path_config is not None:
        #     cfg = ConfigRun.load(Path(path_config))
        #     cfg.task = task
        #     cfg.device = device
        #     # FIXME: Cant use cfg values atm, need to allow overwriting values
        #     if params.get("max_epochs", None) is not None:
        #         cfg.hyperparams['max_epochs'] = params["max_epochs"]
        #     if params.get("n_ensembles", None) is not None:
        #         cfg.hyperparams['n_ensembles'] = params["n_ensembles"]
        # else:
        cfg = ConfigRun(hyperparams=params, task=task, device=device)
        if path_weights is None and "path_weights" not in params:
            raise ValueError(
                "Missing required model hyperparameter 'path_weights'. "
                "Either specify `path_weights=None` to train from random initialization (not recommended), "
                "or specify a local path to a pre-trained weights file such as `path/to/file/tabpfnmix_base.pt`."
            )

        if cfg.hyperparams["max_epochs"] == 0 and cfg.hyperparams["n_ensembles"] != 1:
            logger.log(
                30,
                f"WARNING: max_epochs should be > 0 if n_ensembles > 1, otherwise there will be zero quality benefit with slower inference. "
                f"(max_epochs={cfg.hyperparams['max_epochs']}, n_ensembles={cfg.hyperparams['n_ensembles']})"
            )

        X = self.preprocess(X)
        y = y.values
        if X_val is not None:
            X_val = self.preprocess(X_val)
            y_val = y_val.values

        need_to_reset_torch_threads = False
        torch_threads_og = None
        if num_cpus is not None and isinstance(num_cpus, (int, float)):
            torch_threads_og = torch.get_num_threads()
            if torch_threads_og != num_cpus:
                # reset torch threads back to original value after fit
                torch.set_num_threads(num_cpus)
                need_to_reset_torch_threads = True

        model_cls = self._get_model_type()

        self.model = model_cls(
            cfg=cfg,
            n_classes=n_classes,
            split_val=params["split_val"],
            path_to_weights=path_weights,
            stopping_metric=self.stopping_metric,
            use_best_epoch=params["use_best_epoch"],
        )
        self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
        )

        if need_to_reset_torch_threads:
            torch.set_num_threads(torch_threads_og)

        return self

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Converts categorical to label encoded integers
        Keeps missing values, as TabPFN automatically handles missing values internally.
        """
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            # FIXME: Check if this is needed, never actually tried removing it, copy pasted from TabPFNModel implementation in AG
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        X = X.values.astype(np.float64)
        return X

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {
            "problem_types": [BINARY, MULTICLASS],
        }
        default_ag_args.update(extra_ag_args)
        return default_ag_args

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_classes": 10,
            }
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
           #  "fold_fitting_strategy": "sequential_local",  # FIXME: Comment out after debugging for large speedup
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _get_maximum_resources(self) -> dict[str, int | float]:
        # torch model trains slower when utilizing virtual cores and this issue scale up when the number of cpu cores increases
        return {"num_cpus": ResourceManager.get_cpu_count_psutil(logical=False)}

    def _get_default_resources(self) -> tuple[int, float]:
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(X=X, problem_type=self.problem_type, num_classes=self.num_classes, hyperparameters=hyperparameters, **kwargs)

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        # TODO: This is wildly inaccurate, find a better estimation
        return 10 * get_approximate_df_mem_usage(X).sum()

    @classmethod
    def _class_tags(cls) -> dict:
        return {
            "can_estimate_memory_usage_static": True,
        }

    def _ag_params(self) -> set:
        return {"max_classes"}

    def _more_tags(self) -> dict:
        tags = {"can_refit_full": True}
        return tags
