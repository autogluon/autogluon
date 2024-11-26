from __future__ import annotations

import logging
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE
from autogluon.core.models import AbstractModel
from autogluon.core.utils import generate_train_test_split
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.features.generators import LabelEncoderFeatureGenerator

logger = logging.getLogger(__name__)


class TabPFNMixModel(AbstractModel):
    """
    [Experimental model] Can be changed/removed without warning in future releases.

    TabPFNMix is based off of the TabPFN and TabForestPFN models.

    It is a tabular transformer model pre-trained on purely synthetic data.

    It currently has several limitations:
    1. Does not support regression
    2. Does not support >10 classes
    3. Does not support GPU

    For more information, refer to the `./_internals/README.md` file.
    """
    weights_file_name = "model.pt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._weights_saved = False

    def _get_model_type(self):
        from ._internal.tabpfnmix_classifier import TabPFNMixClassifier
        from ._internal.tabpfnmix_regressor import TabPFNMixRegressor
        if self.problem_type in ['binary', 'multiclass']:
            model_cls = TabPFNMixClassifier
        elif self.problem_type in ['regression']:
            model_cls = TabPFNMixRegressor
        else:
            raise AssertionError(f"TabPFN does not support problem_type='{self.problem_type}'")
        return model_cls

    def _set_default_params(self):
        """Specifies hyperparameter values to use by default"""
        default_params = {
            # most important hyperparameters. Only set `n_estimators>1` if `max_epochs>1`, else there will be no benefit.
            # model_path,  # most important, defines huggingface model path
            "model_path_classifier": "autogluon/tabpfn-mix-1.0-classifier",  # if specified, overrides model_path for classification problems, set to None to ignore.
            "model_path_regressor": "autogluon/tabpfn-mix-1.0-regressor",  # if specified, overrides model_path for regression problems, set to None to ignore.
            # weights_path,  # most important, defines weights location (overrides huggingface weights if specified)
            # weights_path_classifier,  # if specified, overrides weights_path for classification problems
            # weights_path_regressor,  # if specified, overrides weights_path for regression problems
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

            # architecture hyperparameters, recommended to keep as default unless using a custom pre-trained backbone
            "n_classes": 10,
            "n_features": 100,
            "n_heads": 4,
            "n_layers": 12,
            "attn_dropout": 0.0,
            "dim": 512,
            "y_as_float_embedding": True,

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
    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, time_limit: float = None, num_cpus: int = 1, num_gpus: float = 0, **kwargs):
        time_start = time.time()
        try_import_torch()
        import torch
        from ._internal.config.config_run import ConfigRun

        ag_params = self._get_ag_params()
        max_classes = ag_params.get("max_classes")
        if max_classes is not None and self.num_classes is not None and self.num_classes > max_classes:
            # TODO: Move to earlier stage when problem_type is checked
            raise AssertionError(f"Max allowed classes for the model is {max_classes}, " f"but found {self.num_classes} classes.")

        params = self._get_model_params()
        sample_rows = ag_params.get("sample_rows", None)
        sample_rows_val = ag_params.get("sample_rows_val", None)
        max_rows = ag_params.get("max_rows", None)

        # TODO: Make max_rows generic
        if max_rows is not None and isinstance(max_rows, (int, float)) and len(X) > max_rows:
            raise AssertionError(f"Skipping model due to X having more rows than `ag.max_rows={max_rows}` (len(X)={len(X)})")

        # TODO: Make sample_rows generic
        if sample_rows is not None and isinstance(sample_rows, int) and len(X) > sample_rows:
            X, y = self._subsample_data(X=X, y=y, num_rows=sample_rows)

        # TODO: Make sample_rows generic
        if X_val is not None and y_val is not None and sample_rows_val is not None and isinstance(sample_rows_val, int) and len(X_val) > sample_rows_val:
            X_val, y_val = self._subsample_data(X=X_val, y=y_val, num_rows=sample_rows_val)

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

        model_path = None
        if task == Task.CLASSIFICATION:
            if "model_path_classifier" in params and params["model_path_classifier"] is not None:
                model_path = params["model_path_classifier"]
        elif task == Task.REGRESSION:
            if "model_path_regressor" in params and params["model_path_regressor"] is not None:
                model_path = params["model_path_regressor"]
        if model_path is None:
            model_path = params.get("model_path", None)

        weights_path = None
        if task == Task.CLASSIFICATION:
            if "weights_path_classifier" in params and params["weights_path_classifier"] is not None:
                weights_path = Path(params["weights_path_classifier"])
        elif task == Task.REGRESSION:
            if "weights_path_regressor" in params and params["weights_path_regressor"] is not None:
                weights_path = Path(params["weights_path_regressor"])
        if weights_path is None:
            if "weights_path" in params and params["weights_path"] is not None:
                weights_path = Path(params["weights_path"])

        if weights_path is None and model_path is None:
            logger.log(15, "\tNo model_path or weights_path specified, fitting model from random initialization...")
        elif weights_path is not None:
            logger.log(15, f'\tLoading pre-trained weights from file... (weights_path="{weights_path}")')

        cfg = ConfigRun(hyperparams=params, task=task, device=device)

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

        # FIXME: What if an exception occurs or timeout occurs after updating threads? Can we add logic to ensure torch num_threads are reset?
        need_to_reset_torch_threads = False
        torch_threads_og = None
        if num_cpus is not None and isinstance(num_cpus, (int, float)):
            torch_threads_og = torch.get_num_threads()
            if torch_threads_og != num_cpus:
                # reset torch threads back to original value after fit
                torch.set_num_threads(num_cpus)
                need_to_reset_torch_threads = True

        model_cls = self._get_model_type()

        if time_limit is not None:
            time_cur = time.time()
            time_left = time_limit - (time_cur - time_start)
            if time_left <= 0:
                raise TimeLimitExceeded(f"No time remaining to fit model (time_limit={time_limit:.2f}s, time_left={time_left:.2f}s)")
            time_limit = time_left

        self.model = model_cls(
            cfg=cfg,
            n_classes=n_classes,
            split_val=params["split_val"],
            model_path=model_path,
            weights_path=weights_path,
            stopping_metric=self.stopping_metric,
            use_best_epoch=params["use_best_epoch"],
        )
        self.model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            time_limit=time_limit,
        )

        # Ensure refit_full uses the same number of max_epochs as the original's best
        self.params_trained["max_epochs"] = self.model.trainer.best_epoch
        self.params_trained["ag.max_rows"] = None  # This ensures we don't raise an exception during refit_full

        # reduce memory and disk usage by 3x
        self.model.trainer.minimize_for_inference()

        if need_to_reset_torch_threads:
            torch.set_num_threads(torch_threads_og)

        return self

    # TODO: Make this generic by creating a generic `preprocess_train` and putting this logic prior to `_preprocess`.
    def _subsample_data(self, X: pd.DataFrame, y: pd.Series, num_rows: int, random_state=0) -> (pd.DataFrame, pd.Series):
        num_rows_to_drop = len(X) - num_rows
        X, _, y, _ = generate_train_test_split(
            X=X,
            y=y,
            problem_type=self.problem_type,
            test_size=num_rows_to_drop,
            random_state=random_state,
            min_cls_count_train=1,
        )
        return X, y

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

    # FIXME: Switch to `torch.save(_model_weights.state_dict(), PATH)`, need to reinitialize the model though...
    def save(self, path: str = None, verbose=True) -> str:
        _model_weights = None
        if self.model is not None:
            _model_weights = self.model.trainer.model
            self.model.trainer.model = None
            self._weights_saved = True
        path = super().save(path=path, verbose=verbose)
        if _model_weights is not None:
            import torch
            os.makedirs(self.path, exist_ok=True)
            torch.save(_model_weights, self.weights_path)
            self.model.trainer.model = _model_weights
        return path

    # FIXME: Switch to `weights_only=True`, need to reinitialize the model though...
    @classmethod
    def load(cls, path: str, reset_paths=False, verbose=True):
        model: TabPFNMixModel = super().load(path=path, reset_paths=reset_paths, verbose=verbose)

        if model._weights_saved:
            import torch
            model.model.trainer.model = torch.load(model.weights_path, weights_only=False)  # nosec B614
            model._weights_saved = False
        return model

    @property
    def weights_path(self) -> str:
        return os.path.join(self.path, self.weights_file_name)

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {
            "problem_types": [BINARY, MULTICLASS, REGRESSION],
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

    def get_minimum_ideal_resources(self) -> dict[str, int | float]:
        return {"num_cpus": 4}

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        # TODO: This is wildly inaccurate, find a better estimation
        # TODO: Fitting 8 in parallel causes many OOM errors with 32 GB of memory on relatively small datasets, so each model is using over 4 GB of memory
        # TODO: Fitting 4 in parallel still causes many OOM errors with 32 GB of memory on relatively small datasets, so each model is using over 8 GB of memory
        #  The below logic returns a minimum of 8.8 GB, to avoid OOM errors
        data_mem_usage = 5 * get_approximate_df_mem_usage(X).sum()  # rough estimate
        model_size = 160*1e6  # model weights are ~160 MB  # TODO: Avoid hardcoding, we can derive from the model itself?
        model_mem_usage = model_size * 5  # Account for 1x copy being fit, 1x copy checkpointed, 2x for optimizer, and 1x for overhead
        model_fit_usage = model_size * 50  # TODO: This is a placeholder large value to try to avoid OOM errors
        mem_usage_estimate = data_mem_usage + model_mem_usage + model_fit_usage
        return mem_usage_estimate

    @classmethod
    def _class_tags(cls) -> dict:
        return {
            "can_estimate_memory_usage_static": True,
        }

    def _ag_params(self) -> set:
        return {"max_classes", "max_rows", "sample_rows", "sample_rows_val"}

    def _more_tags(self) -> dict:
        tags = {"can_refit_full": True}
        return tags