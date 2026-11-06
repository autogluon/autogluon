from __future__ import annotations

import copy

import pandas as pd

from ._internal.core.fit_context import FitContextMemo
from ._internal.data.feature_reduction import DEFAULT_FEATURE_REDUCTION, DEFAULT_MAX_FEATURES
from .checkpoint import CHECKPOINTS, DEFAULT_MODEL_VERSION, validate_model_version
from .mitra_model import MitraModel


class MitraV2Model(MitraModel):
    """Mitra's native v2 fine-tuning API, registered as ``MITRA``.

    V2 checkpoints and fine-tuning settings are the default. ``model_version="1"``
    selects v1 weights with the same v2 trainer. Use :class:`MitraModel` explicitly
    for the original v1 API. An explicit ``hf_model`` overrides the default
    checkpoint selected by ``model_version``. The checkpoint's ``config.json``
    defines the architecture; release metadata is not required.

    A fit defaults to 3,600 seconds shared across bag folds. AutoGluon's global
    ``time_limit`` can impose a shorter limit. Override the budget with
    ``ag.max_time_limit`` and, for bagging, ``ag.ens.ag.max_time_limit``.

    The v2 defaults use 50 fine-tuning steps, 10 warmup steps, and weight decay 0.3.
    The learning rate is 3e-6 for binary datasets with at most 16,384 training rows
    and 1e-5 otherwise. Bagging uses the outer training size for this rule.
    Held-out fold rows join inference support only after OOF scoring.

    Fine-tuning selects checkpoints using ``log_loss`` for classification and
    ``mse`` for regression, matching the released HF recipe independently of
    AutoGluon's ``eval_metric``. Set the ``metric`` hyperparameter to override it.

    ``max_features=256`` controls feature reduction independently of the
    ``ag.max_features`` fit constraint. Set ``feature_reduction=None`` to disable
    reduction or choose ``"select"`` / ``"svd"``; ``"auto"`` is the default.
    Feature reduction is fitted once on the outer training table before
    bagging, matching the released HF classification and regression recipes.
    """

    ag_key = "MITRA"
    ag_name = "Mitra"
    _default_auxiliary_params_extra = {
        "max_rows": 1_000_000,
        "max_features": 50_000,
        "max_classes": None,
        "max_time_limit": 3600,
    }
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "auto",
        "ag.max_time_limit": 3600,
    }

    def _validate_fine_tune_recipe(self):
        recipe = self.params.get("fine_tune_recipe", self._user_params.get("fine_tune_recipe"))
        if recipe is not None and (not isinstance(recipe, str) or recipe != "2"):
            raise ValueError(
                "MitraV2Model uses the v2 fine_tune_recipe. "
                "Use hyperparameters={MitraModel: {...}} for the original v1 API."
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_reducer = None
        self._feature_reduction_fitted_upstream = False
        self._use_training_projection = False
        self._pending_fold_support = None
        self._fit_context_memo = None

    def validate_fit_args(self, X: pd.DataFrame, **kwargs):
        super().validate_fit_args(X=X, **kwargs)
        # BaggedEnsembleModel validates its child template on the full table
        # before copying it into folds. Preserve that size when the fold's
        # own validation is skipped by the existing upstream-validation flag.
        if not self._fit_constraints_validated_upstream:
            self._num_samples_total = len(X)
            self._fit_context_memo = FitContextMemo()
            # HF fits both supervised selection and the regression projection
            # before bagging. This hook receives the outer training data and
            # labels; fold children reuse the fitted reducer.
            reducer = self._make_feature_reducer(X)
            y = kwargs.get("y")
            if y is not None or (self.problem_type == "regression" and reducer.method in ("auto", "svd")):
                reducer.fit(X, y=y, cache_training=True)
                if reducer.projection is not None or reducer.columns is not None:
                    self._feature_reducer = reducer
                    self._feature_reduction_fitted_upstream = True

    def _make_feature_reducer(self, X):
        from ._internal.data.feature_reduction import FeatureReducer

        return FeatureReducer(
            problem_type=self.problem_type,
            max_features=self.params.get("max_features", self._user_params.get("max_features", DEFAULT_MAX_FEATURES)),
            method=self.params.get(
                "feature_reduction",
                self._user_params.get("feature_reduction", DEFAULT_FEATURE_REDUCTION),
            ),
            has_categoricals=self._has_non_numeric_reduction_dtype(X),
        )

    @staticmethod
    def _has_non_numeric_reduction_dtype(X: pd.DataFrame) -> bool:
        return any(
            isinstance(dtype, pd.CategoricalDtype)
            or not (
                pd.api.types.is_numeric_dtype(dtype)
                or pd.api.types.is_datetime64_any_dtype(dtype)
                or pd.api.types.is_timedelta64_dtype(dtype)
            )
            for dtype in X.dtypes
        )

    @classmethod
    def _estimate_reduced_num_features(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict,
        problem_type: str | None,
    ) -> int:
        n_rows, n_features = X.shape
        max_features = hyperparameters.get("max_features", DEFAULT_MAX_FEATURES)
        if "feature_reduction" in hyperparameters:
            method = hyperparameters["feature_reduction"]
        else:
            method = DEFAULT_FEATURE_REDUCTION
        if (
            max_features is None
            or method is None
            or n_features <= max_features
            or cls._has_non_numeric_reduction_dtype(X)
        ):
            return n_features
        if method == "auto":
            method = "svd" if problem_type == "regression" else "select"
        if method == "svd":
            return min(max_features, n_rows, n_features)
        if problem_type != "regression":
            # Classification can retain the original width when the
            # continuous-feature gate rejects selection. Memory estimation
            # precedes fitting the reducer, so keep the conservative bound.
            return n_features
        return min(max_features, n_features)

    def _get_model_version(self) -> str:
        version = self.params.get(
            "model_version",
            self._user_params.get("model_version", DEFAULT_MODEL_VERSION),
        )
        return validate_model_version(version)

    def get_model_cls(self):
        self._get_model_version()
        self._validate_fine_tune_recipe()
        from .finetune import MitraFinetuneClassifier, MitraFinetuneRegressor

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = MitraFinetuneClassifier
        elif self.problem_type == "regression":
            model_cls = MitraFinetuneRegressor
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")
        return model_cls

    def _preprocess_align_features(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if getattr(self, "_feature_reduction_fitted_upstream", False):
            # A column varying across the outer table can be constant in one
            # fold. It still belongs to the shared projection's input schema.
            return self._feature_reducer.align_input(X)
        return super()._preprocess_align_features(X, **kwargs)

    def _preprocess_features(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        if is_train and not getattr(self, "_feature_reduction_fitted_upstream", False):
            self._feature_reducer = self._make_feature_reducer(X)
            X = super()._preprocess_features(X, is_train=True, **kwargs)
            self._feature_reducer.fit(X, kwargs["y"], cache_training=True)
        else:
            X = super()._preprocess_features(X, is_train=is_train, **kwargs)
        if getattr(self, "_feature_reducer", None) is not None:
            if is_train or getattr(self, "_use_training_projection", False):
                X = self._feature_reducer.transform_training(X)
            else:
                X = self._feature_reducer.transform(X)
        return X

    def _prepare_fit(self):
        self._pending_fold_support = None
        self._use_training_projection = self._fit_constraints_validated_upstream and getattr(
            self, "_feature_reduction_fitted_upstream", False
        )

    def _get_estimator_params(self) -> dict:
        hyp = super()._get_estimator_params()
        model_version = self._get_model_version()
        hyp["model_version"] = model_version
        hyp.pop("fine_tune_recipe", None)
        hyp.pop("max_features", None)
        hyp.pop("feature_reduction", None)
        return hyp

    def _initialize_estimator(self, model_cls, hyp: dict, X: pd.DataFrame):
        super()._initialize_estimator(model_cls, hyp, X)
        self.model._fit_context_memo = self._fit_context_memo
        self.model._num_samples_total = getattr(self, "_num_samples_total", len(X))

    def _finalize_estimator_fit(self, model, X_val: pd.DataFrame, y_val: pd.Series):
        super()._finalize_estimator_fit(model, X_val=X_val, y_val=y_val)
        if not self._use_training_projection and self._feature_reducer is not None:
            self._feature_reducer.clear_training_cache()
        if (
            self._fit_constraints_validated_upstream
            and getattr(model, "heldout_in_support", False)
            and X_val is not None
        ):
            # Retain the already preprocessed fold data until AutoGluon's
            # existing post-scoring cleanup. It must not enter the context
            # used to compute the fold's validation predictions.
            self._pending_fold_support = (X_val.to_numpy(copy=True), y_val.to_numpy(copy=True))

    def _set_default_params(self):
        default_params = {
            "n_estimators": 1,
            "model_version": DEFAULT_MODEL_VERSION,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_model_template_for_save(self):
        model = copy.copy(self)
        if self._feature_reducer is not None:
            model._feature_reducer = copy.copy(self._feature_reducer)
            model._feature_reducer.clear_training_cache()
            model._use_training_projection = False
        return model

    def reduce_memory_size(
        self, remove_fit: bool = True, remove_info: bool = False, requires_save: bool = True, **kwargs
    ):
        super().reduce_memory_size(
            remove_fit=remove_fit, remove_info=remove_info, requires_save=requires_save, **kwargs
        )
        pending_support = getattr(self, "_pending_fold_support", None)
        if remove_fit and requires_save and self.val_score is not None and pending_support is not None:
            # Both existing fold strategies call this after setting val_score,
            # before saving the child. Merge the retained fold context once
            # and release the temporary validation arrays.
            self.model.extend_support(*pending_support)
            self._pending_fold_support = None
        if (
            remove_fit
            and requires_save
            and (self.val_score is not None or not self._fit_constraints_validated_upstream)
            and getattr(self, "_feature_reducer", None) is not None
        ):
            self._feature_reducer.clear_training_cache()
            self._use_training_projection = False

    @classmethod
    def download_default_weights(cls, model_version: str = DEFAULT_MODEL_VERSION):
        """
        Download default weights for Mitra from HuggingFace.
        Includes both classifier and regressor weights.

        This is useful to call when building a docker image to avoid having to download Mitra weights for each instance.
        This is also useful for benchmarking as a first sanity check
        to avoid HuggingFace potentially blocking the download.

        Requires an internet connection.
        """
        validate_model_version(model_version)
        for repo_id in CHECKPOINTS[model_version].values():
            cls.download_weights(repo_id=repo_id)

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        kwargs["hyperparameters"] = self._get_model_params(convert_search_spaces_to_default=True)
        return self.estimate_memory_usage_static(
            X=X, problem_type=self.problem_type, num_classes=self.num_classes, **kwargs
        )

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        hyperparameters = kwargs.get("hyperparameters") or {}
        problem_type = kwargs.get("problem_type")
        is_binary = problem_type == "binary"
        train_support = hyperparameters.get("max_samples_support") or (
            20480 if problem_type == "regression" else 16384
        )
        train_query = hyperparameters.get("max_samples_query") or 1024
        eval_query = hyperparameters.get("eval_max_samples_query") or 16384
        predict_support = hyperparameters.get("predict_max_samples_support") or (16384 if is_binary else 32768)
        predict_query = hyperparameters.get("predict_max_samples_query") or 16384
        row_cap = max(train_support + train_query, train_support + eval_query, predict_support + predict_query)
        feature_cap = cls._estimate_reduced_num_features(
            X=X,
            hyperparameters=hyperparameters,
            problem_type=problem_type,
        )
        # Estimate the tensor shape seen by the network after feature
        # reduction, rather than rejecting reducible wide tables first.
        X = X.iloc[:row_cap, :feature_cap]
        # Multiply by 0.9 as currently this is overly safe.
        estimate = int(
            0.9
            * max(
                cls._estimate_memory_usage_static_cpu_icl(X=X, **kwargs),
                cls._estimate_memory_usage_static_cpu_ft_icl(X=X, **kwargs),
                cls._estimate_memory_usage_static_gpu_cpu(X=X, **kwargs),
                cls._estimate_memory_usage_static_gpu_gpu(X=X, **kwargs),
            )
        )
        num_classes = kwargs.get("num_classes")
        if num_classes is not None and num_classes > 10:
            from ._internal.data.class_hierarchy import hierarchy_node_count

            additional_nodes = hierarchy_node_count(num_classes, max_classes=10) - 1
            n_estimators = hyperparameters.get("n_estimators", 1)
            # Each additional node retains its fitted network while later
            # nodes train. Current classifier checkpoints occupy about 303 MB;
            # keep headroom for tensor/object overhead without multiplying the
            # one-node training workspace for every node.
            estimate += additional_nodes * n_estimators * 350_000_000
        return estimate
