"""
Code Adapted from TabArena: https://github.com/autogluon/tabrepo/blob/main/tabrepo/benchmark/models/ag/realmlp/realmlp_model.py
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular import __version__
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

logger = logging.getLogger(__name__)


@contextmanager
def set_logger_level(logger_name: str, level: int):
    _logger = logging.getLogger(logger_name)
    old_level = _logger.level
    _logger.setLevel(level)
    try:
        yield
    finally:
        _logger.setLevel(old_level)


# pip install pytabkit
class RealMLPModel(AbstractTorchModel):
    """
    RealMLP is an improved multilayer perception (MLP) model
    through a bag of tricks and better default hyperparameters.

    RealMLP is the top performing method overall on TabArena-v0.1: https://tabarena.ai

    Paper: Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data
    Authors: David Holzmüller, Léo Grinsztajn, Ingo Steinwart
    Codebase: https://github.com/dholzmueller/pytabkit
    License: Apache-2.0

    .. versionadded:: 1.4.0
    """

    ag_key = "REALMLP"
    ag_name = "RealMLP"
    ag_priority = 75
    seed_name = "random_state"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._features_bool = None
        self._bool_to_cat = None

    def get_model_cls(self, default_hyperparameters: Literal["td", "td_s"] = "td"):
        from pytabkit import (
            RealMLP_TD_Classifier,
            RealMLP_TD_Regressor,
            RealMLP_TD_S_Classifier,
            RealMLP_TD_S_Regressor,
        )

        assert default_hyperparameters in ["td", "td_s"]
        if self.problem_type in ["binary", "multiclass"]:
            if default_hyperparameters == "td":
                model_cls = RealMLP_TD_Classifier
            else:
                model_cls = RealMLP_TD_S_Classifier
        else:
            if default_hyperparameters == "td":
                model_cls = RealMLP_TD_Regressor
            else:
                model_cls = RealMLP_TD_S_Regressor
        return model_cls

    def get_device(self) -> str:
        return self.model.device

    def _set_device(self, device: str):
        self.model.to(device)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        start_time = time.time()

        try:
            import pytabkit
            import torch
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import pytabkit/torch! To use the ReaLMLP model, "
                f"do: `pip install autogluon.tabular[realmlp]=={__version__}`.",
            )
            raise err

        if verbosity == 0:
            _lightning_log_level = logging.ERROR
        elif verbosity <= 2:
            _lightning_log_level = logging.WARNING
        else:
            _lightning_log_level = logging.INFO

        # FIXME: code assume we only see one GPU in the fit process.
        device = "cpu" if num_gpus == 0 else "cuda:0"
        if (device == "cuda:0") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        hyp = self._get_model_params()

        default_hyperparameters = hyp.pop("default_hyperparameters", "td")

        model_cls = self.get_model_cls(default_hyperparameters=default_hyperparameters)

        metric_map = {
            "roc_auc": "1-auc_ovr_alt",
            "accuracy": "class_error",
            "balanced_accuracy": "1-balanced_accuracy",
            "log_loss": "cross_entropy",
            "rmse": "rmse",
            "root_mean_squared_error": "rmse",
            "r2": "rmse",
            "mae": "mae",
            "mean_average_error": "mae",
        }

        val_metric_name = metric_map.get(self.stopping_metric.name, None)

        init_kwargs = dict()

        if val_metric_name is not None:
            init_kwargs["val_metric_name"] = val_metric_name

        # TODO: Make this smarter? Maybe use `eval_metric.needs_pred`
        if hyp["use_ls"] is not None and isinstance(hyp["use_ls"], str) and hyp["use_ls"] == "auto":
            if val_metric_name is None:
                hyp["use_ls"] = False
            elif val_metric_name in ["cross_entropy", "1-auc_ovr_alt"]:
                hyp["use_ls"] = False
            else:
                hyp["use_ls"] = None

        if X_val is None:
            hyp["use_early_stopping"] = False
            hyp["val_fraction"] = 0

        bool_to_cat = hyp.pop("bool_to_cat", True)
        impute_bool = hyp.pop("impute_bool", True)
        name_categories = hyp.pop("name_categories", True)

        n_features = len(X.columns)
        if (
            "predict_batch_size" in hyp
            and isinstance(hyp["predict_batch_size"], str)
            and hyp["predict_batch_size"] == "auto"
        ):
            # simple heuristic to avoid OOM during inference time
            # note: this isn't fool-proof, and ignores the actual memory availability of the machine.
            # note: this is based on an assumption of 32 GB of memory available on the instance
            # default is 1024
            hyp["predict_batch_size"] = max(min(int(8192 * 200 / n_features), 8192), 64)

        self.model = model_cls(
            n_threads=num_cpus,
            device=device,
            **init_kwargs,
            **hyp,
        )

        X = self.preprocess(X, is_train=True, bool_to_cat=bool_to_cat, impute_bool=impute_bool)

        # FIXME: In rare cases can cause exceptions if name_categories=False, unknown why
        extra_fit_kwargs = {}
        if name_categories:
            cat_col_names = X.select_dtypes(include="category").columns.tolist()
            extra_fit_kwargs["cat_col_names"] = cat_col_names

        if X_val is not None:
            X_val = self.preprocess(X_val)

        with set_logger_level("lightning.pytorch", _lightning_log_level):
            self.model = self.model.fit(
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                time_to_fit_in_seconds=time_limit - (time.time() - start_time) if time_limit is not None else None,
                **extra_fit_kwargs,
            )

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        with set_logger_level("lightning.pytorch", logging.WARNING):
            return super()._predict_proba(X=X, kwargs=kwargs)

    # TODO: Move missing indicator + mean fill to a generic preprocess flag available to all models
    # FIXME: bool_to_cat is a hack: Maybe move to abstract model?
    def _preprocess(
        self, X: pd.DataFrame, is_train: bool = False, bool_to_cat: bool = False, impute_bool: bool = True, **kwargs
    ) -> pd.DataFrame:
        """
        Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        X = super()._preprocess(X, **kwargs)

        # FIXME: is copy needed?
        X = X.copy(deep=True)
        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(required_special_types=["bool"])
            if impute_bool:  # Technically this should do nothing useful because bools will never have NaN
                self._features_to_impute = self._feature_metadata.get_features(valid_raw_types=["int", "float"])
                self._features_to_keep = self._feature_metadata.get_features(invalid_raw_types=["int", "float"])
            else:
                self._features_to_impute = self._feature_metadata.get_features(
                    valid_raw_types=["int", "float"], invalid_special_types=["bool"]
                )
                self._features_to_keep = [
                    f for f in self._feature_metadata.get_features() if f not in self._features_to_impute
                ]
            if self._features_to_impute:
                self._imputer = SimpleImputer(strategy="mean", add_indicator=True)
                self._imputer.fit(X=X[self._features_to_impute])
                self._indicator_columns = [
                    c for c in self._imputer.get_feature_names_out() if c not in self._features_to_impute
                ]
        if self._imputer is not None:
            X_impute = self._imputer.transform(X=X[self._features_to_impute])
            X_impute = pd.DataFrame(X_impute, index=X.index, columns=self._imputer.get_feature_names_out())
            if self._indicator_columns:
                # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
                # TODO: Add to features_bool?
                X_impute[self._indicator_columns] = X_impute[self._indicator_columns].astype("category")
            X = pd.concat([X[self._features_to_keep], X_impute], axis=1)
        if self._bool_to_cat and self._features_bool:
            # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
            X[self._features_bool] = X[self._features_bool].astype("category")
        return X

    def _set_default_params(self):
        default_params = dict(
            # Don't use early stopping by default, seems to work well without
            use_early_stopping=False,
            early_stopping_additive_patience=40,
            early_stopping_multiplicative_patience=3,
            # verdict: use_ls="auto" is much better than None.
            use_ls="auto",
            # verdict: no impact, but makes more sense to be False.
            impute_bool=False,
            # verdict: name_categories=True avoids random exceptions being raised in rare cases
            name_categories=True,
            # verdict: bool_to_cat=True is equivalent to False in terms of quality, but can be slightly faster in training time
            #  and slightly slower in inference time
            bool_to_cat=True,
            # verdict: "td" is better than "td_s"
            default_hyperparameters="td",  # options ["td", "td_s"]
            predict_batch_size="auto",  # if auto, uses AutoGluon's heuristic to set a value between 8192 and 64.
        )
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_stopping_metric(self):
        return self.eval_metric

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
        Heuristic memory estimate that correlates strongly with RealMLP's more sophisticated method

        More comprehensive memory estimate logic:

        ```python
        from typing import Any

        from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface
        from pytabkit.models.data.data import DictDataset, TensorInfo
        from pytabkit.models.sklearn.default_params import DefaultParams

        def estimate_realmlp_cpu_ram_gb(hparams: dict[str, Any], n_numerical: int, cat_sizes: list[int], n_classes: int,
                                        n_samples: int):
            params = copy.copy(DefaultParams.RealMLP_TD_CLASS if n_classes > 0 else DefaultParams.RealMLP_TD_REG)
            params.update(hparams)

            ds = DictDataset(tensors=None, tensor_infos=dict(x_cont=TensorInfo(feat_shape=[n_numerical]),
                                                             x_cat=TensorInfo(cat_sizes=cat_sizes),
                                                             y=TensorInfo(cat_sizes=[n_classes])), device='cpu',
                             n_samples=n_samples)

            alg_interface = NNAlgInterface(**params)
            res = alg_interface.get_required_resources(ds, n_cv=1, n_refit=0, n_splits=1, split_seeds=[0], n_train=n_samples)
            return res.cpu_ram_gb
        ```

        """
        if hyperparameters is None:
            hyperparameters = {}
        plr_hidden_1 = hyperparameters.get("plr_hidden_1", 16)
        plr_hidden_2 = hyperparameters.get("plr_hidden_2", 4)
        hidden_width = hyperparameters.get("hidden_width", 256)

        num_features = len(X.columns)
        columns_mem_est = num_features * 8e5

        hidden_1_weight = 0.13
        hidden_2_weight = 0.42
        width_factor = math.sqrt(hidden_width / 256 + 0.6)

        columns_mem_est_hidden_1 = columns_mem_est * hidden_1_weight * plr_hidden_1 / 16 * width_factor
        columns_mem_est_hidden_2 = columns_mem_est * hidden_2_weight * plr_hidden_2 / 16 * width_factor
        columns_mem_est = columns_mem_est_hidden_1 + columns_mem_est_hidden_2

        dataset_size_mem_est = 5 * get_approximate_df_mem_usage(X).sum()  # roughly 5x DataFrame memory size
        baseline_overhead_mem_est = 3e8  # 300 MB generic overhead

        mem_estimate = dataset_size_mem_est + columns_mem_est + baseline_overhead_mem_est

        return mem_estimate

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        # TODO: Need to add train params support, track best epoch
        #  How to mirror RealMLP learning rate scheduler while forcing stopping at a specific epoch?
        tags = {"can_refit_full": False}
        return tags
