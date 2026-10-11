from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from typing_extensions import Self

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular import __version__
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from .checkpoint import CHECKPOINTS, DEFAULT_MODEL_VERSION, validate_checkpoint_version, validate_model_version
from .recipes import DEFAULT_FINE_TUNE_RECIPE, get_finetune_recipe

logger = logging.getLogger(__name__)


class MitraModel(AbstractTorchModel):
    """
    Mitra is a tabular foundation model pre-trained purely on synthetic data with the goal
    of optimizing fine-tuning performance over in-context learning performance.
    Mitra was developed by the AutoGluon team @ AWS AI.

    Mitra v2: https://huggingface.co/autogluon/mitra-finetune
    Original Mitra: https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models
    License: Apache-2.0

    .. versionadded:: 1.4.0

    Mitra v2 weights and fine-tuning settings are the default. ``model_version``
    selects checkpoint weights; both versions use the new fine-tuning recipe
    by default. Set ``model_version="1", fine_tune_recipe="1"`` to reproduce
    the legacy settings. Both checkpoint versions use the same fine-tuning
    implementation. Versions are never inferred from the shared architecture.

    A Mitra fit defaults to a 3,600-second budget, shared across bag folds.
    AutoGluon's global ``time_limit`` can impose a shorter limit. Override
    Mitra's budget with ``ag.max_time_limit`` and, when bagging,
    ``ag.ens.ag.max_time_limit``.

    The v2 defaults use 50 full fine-tuning steps, 10 warmup steps, and weight
    decay 0.3.
    The learning rate is 3e-6 for binary datasets with at most 16,384
    training rows and 1e-5 otherwise. Pass ``lr`` to override it. With bagging,
    the size rule uses the outer training table, and each child's held-out rows
    join its inference support only after its out-of-fold score is recorded.

    Fine-tuning hyperparameters include ``weight_decay``, ``max_samples_support``,
    ``max_samples_query``, ``predict_max_samples_support``,
    ``predict_max_samples_query``, ``eval_max_samples_query``,
    ``balance_binary_support``, ``heldout_in_support``, and
    ``fine_tune_time_limit`` (an optional per-child training budget).
    ``max_features=256`` controls feature reduction, independently of the
    ``ag.max_features`` fit constraint. Set ``feature_reduction=None`` to disable
    reduction or choose ``"select"`` / ``"svd"``; ``"auto"`` is the default.
    With bagging, the unsupervised regression projection is fitted once on the
    outer training table. Supervised classification selection is fitted within
    each training fold.
    """

    gpu_strongly_recommended: bool = True  # in-context inference is 12-63x slower on CPU
    ag_key = "MITRA"
    ag_name = "Mitra"
    weights_file_name = "model.pt"
    ag_priority = 55
    seed_name = "seed"
    _supported_problem_types = ["binary", "multiclass", "regression"]

    _default_auxiliary_params_extra = {
        "max_rows": 10000,
        "max_features": 500,
        "max_classes": 10,
        "max_time_limit": 3600,
    }
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",  # FIXME: Comment out after debugging for large speedup
        "ag.max_time_limit": 3600,  # The bag ensemble shares HF's one-hour default.
    }
    default_resources_physical_cores_only = True
    default_num_gpus = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._weights_saved = False
        self._feature_reducer = None
        self._feature_reduction_fitted_upstream = False
        self._use_training_projection = False
        self._pending_fold_support = None

    def validate_fit_args(self, X: pd.DataFrame, **kwargs):
        super().validate_fit_args(X=X, **kwargs)
        # BaggedEnsembleModel validates its child template on the full table
        # before copying it into folds. Preserve that size when the fold's
        # own validation is skipped by the existing upstream-validation flag.
        if not self._fit_constraints_validated_upstream:
            self._num_samples_total = len(X)
            # The released regression recipe fits its unsupervised projection
            # before bagging. The existing template-validation hook supplies
            # the outer training table without changing AutoGluon's fold code.
            reducer = self._make_feature_reducer(X)
            if self.problem_type == "regression" and reducer.method in ("auto", "svd"):
                reducer.fit(X, y=None, cache_training=True)
                if reducer.projection is not None:
                    self._feature_reducer = reducer
                    self._feature_reduction_fitted_upstream = True

    def _make_feature_reducer(self, X):
        from ._internal.data.feature_reduction import FeatureReducer

        return FeatureReducer(
            problem_type=self.problem_type,
            max_features=self.params.get("max_features", self._user_params.get("max_features", 256)),
            method=self.params.get(
                "feature_reduction",
                self._user_params.get(
                    "feature_reduction", get_finetune_recipe(self._get_fine_tune_recipe())["feature_reduction"]
                ),
            ),
            has_categoricals=any(
                isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype) for dtype in X.dtypes
            ),
        )

    def _get_model_version(self) -> str:
        version = self.params.get(
            "model_version",
            self.params_trained.get("model_version", self._user_params.get("model_version", DEFAULT_MODEL_VERSION)),
        )
        return validate_model_version(version)

    def _get_fine_tune_recipe(self) -> str:
        recipe = self.params.get(
            "fine_tune_recipe",
            self.params_trained.get("fine_tune_recipe", self._user_params.get("fine_tune_recipe")),
        )
        if recipe is None:
            recipe = self.params_trained.get("fine_tune_recipe", DEFAULT_FINE_TUNE_RECIPE)
        get_finetune_recipe(recipe)
        return recipe

    @staticmethod
    def _get_default_device():
        """Get the best available device for the current system."""
        if ResourceManager.get_gpu_count_torch(cuda_only=True) > 0:
            logger.log(15, "Using CUDA GPU")
            return "cuda"
        else:
            return "cpu"

    def get_model_cls(self):
        self._get_model_version()
        self._get_fine_tune_recipe()
        from .finetune import MitraFinetuneClassifier, MitraFinetuneRegressor

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = MitraFinetuneClassifier
        elif self.problem_type == "regression":
            model_cls = MitraFinetuneRegressor
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")
        return model_cls

    @staticmethod
    def _resolve_hf_model(problem_type: str, hyp: dict) -> str | None:
        """Pop the `hf_*` checkpoint aliases from `hyp` and return the one that applies.

        The most specific alias wins: problem-specific > general > generic. Every alias is
        popped regardless of which one wins, as leaving an unused one behind would leak into
        the model constructor as an unexpected keyword argument.

        Each alias accepts either a local checkpoint directory or a HuggingFace repo id.
        """
        hf_cls_model = hyp.pop("hf_cls_model", None)
        hf_reg_model = hyp.pop("hf_reg_model", None)
        hf_general_model = hyp.pop("hf_general_model", None)
        hf_model = hyp.pop("hf_model", None)

        if problem_type in ["binary", "multiclass"]:
            hf_problem_model = hf_cls_model
        elif problem_type == "regression":
            hf_problem_model = hf_reg_model
        else:
            raise AssertionError(f"Unsupported problem_type: {problem_type}")

        return next((m for m in (hf_problem_model, hf_general_model, hf_model) if m is not None), None)

    def _preprocess_align_features(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if getattr(self, "_feature_reduction_fitted_upstream", False):
            # A column varying across the outer table can be constant in one
            # fold. It still belongs to the shared projection's input schema.
            return X[self._feature_reducer.input_columns]
        return super()._preprocess_align_features(X, **kwargs)

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        if is_train and not getattr(self, "_feature_reduction_fitted_upstream", False):
            self._feature_reducer = self._make_feature_reducer(X)
            X = self._label_encode_categoricals(X, is_train=True)
            self._feature_reducer.fit(X, kwargs["y"], cache_training=True)
        else:
            X = self._label_encode_categoricals(X, is_train=is_train)
        if getattr(self, "_feature_reducer", None) is not None:
            if is_train or getattr(self, "_use_training_projection", False):
                X = self._feature_reducer.transform_training(X)
            else:
                X = self._feature_reducer.transform(X)
        return X

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
        need_to_reset_torch_threads = False
        torch_threads_og = None
        self._pending_fold_support = None
        self._use_training_projection = self._fit_constraints_validated_upstream and getattr(
            self, "_feature_reduction_fitted_upstream", False
        )

        try:
            model_cls = self.get_model_cls()
            import torch
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import Mitra! To use the Mitra model, "
                f"do: `pip install autogluon.tabular[mitra]=={__version__}`.",
            )
            raise err

        hyp = self._get_model_params()
        model_version = self._get_model_version()
        hyp["model_version"] = model_version
        hyp["fine_tune_recipe"] = self._get_fine_tune_recipe()
        hyp.pop("max_features", None)
        hyp.pop("feature_reduction", None)

        hf_model = self._resolve_hf_model(problem_type=self.problem_type, hyp=hyp)
        if hf_model is not None:
            validate_checkpoint_version(hf_model, model_version)
            logger.log(30, f"\tCustom hf_model specified: {hf_model}")
            hyp["hf_model"] = hf_model

        self._log_cpu_fallback_warning(num_gpus=num_gpus)
        if hyp.get("device", None) is None:
            if num_gpus == 0:
                hyp["device"] = "cpu"
            else:
                hyp["device"] = self._get_default_device()

        if hyp["device"] == "cpu" and hyp.get("fine_tune", True):
            logger.log(
                30,
                "\tWarning: Attempting to fine-tune Mitra on CPU. This will be very slow. "
                "We strongly recommend using a GPU instance to fine-tune Mitra.",
            )

        # `state_dict_classification` / `state_dict_regression` pointed at a raw `.pt` checkpoint.
        # That loading path was never reachable, so these have always been no-ops. Pop them so an
        # existing hyperparameter dict keeps working, and point users at the replacement.
        for deprecated_key in ["state_dict", "state_dict_classification", "state_dict_regression"]:
            if hyp.pop(deprecated_key, None) is not None:
                logger.log(
                    30,
                    f"\tWarning: `{deprecated_key}` is deprecated and ignored. "
                    f"To load custom weights, save them with `Tab2D.save_pretrained(<dir>)` and pass "
                    f"the directory as `hf_model`, which also accepts a HuggingFace repo id.",
                )

        if "verbose" not in hyp:
            hyp["verbose"] = verbosity >= 3

        self.model = model_cls(**hyp)
        self.model._num_samples_total = getattr(self, "_num_samples_total", len(X))

        X = self.preprocess(X, y=y, is_train=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        if num_cpus is not None:
            torch_threads_og = torch.get_num_threads()
            if torch_threads_og != num_cpus:
                torch.set_num_threads(num_cpus)
                need_to_reset_torch_threads = True
        try:
            model = self.model.fit(
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                time_limit=time_limit,
            )
        finally:
            if need_to_reset_torch_threads:
                torch.set_num_threads(torch_threads_og)

        for i in range(len(model.trainers)):
            model.trainers[i].post_fit_optimize()

        self.model = model
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
            "fine_tune_recipe": DEFAULT_FINE_TUNE_RECIPE,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self):
        params = super()._get_default_auxiliary_params()
        if self._get_fine_tune_recipe() == "2":
            params.update(max_rows=1_000_000, max_features=50_000, max_classes=None)
        return params

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

    def weights_path(self, path: str | None = None) -> str:
        if path is None:
            path = self.path
        return str(Path(path) / self.weights_file_name)

    def save(self, path: str = None, verbose=True) -> str:
        if self.model is None:
            return super().save(path=path, verbose=verbose)
        self._save_model_artifact(path=path)
        trainer_states = []
        try:
            for trainer in self.model.trainers:
                state = {
                    name: getattr(trainer, name) for name in ("model", "bins", "bin_width") if hasattr(trainer, name)
                }
                trainer_states.append((trainer, state))
                trainer.model = None
                # The network is stored separately, but the regression bin
                # tensors stay in the pickle. They must load without CUDA.
                for name in ("bins", "bin_width"):
                    if name in state:
                        setattr(trainer, name, state[name].cpu())
            return super().save(path=path, verbose=verbose)
        finally:
            for trainer, state in trainer_states:
                for name, value in state.items():
                    setattr(trainer, name, value)

    def _save_model_artifact(self, path: str | None):
        if path is None:
            path = self.path
        import torch

        device_og = self.device
        self.set_device("cpu")
        try:
            model_weights = [trainer.model for trainer in self.model.trainers]
            os.makedirs(path, exist_ok=True)
            torch.save(model_weights, self.weights_path(path=path))
        finally:
            self.set_device(device_og)
        self._weights_saved = True

    def _load_model_artifact(self):
        import torch

        device = self.suggest_device_infer()
        model_weights_list = torch.load(self.weights_path(), map_location="cpu", weights_only=False)  # nosec B614
        for i in range(len(self.model.trainers)):
            self.model.trainers[i].model = model_weights_list[i]
        self.set_device(device)

    def _set_device(self, device: str):
        for i in range(len(self.model.trainers)):
            self.model.trainers[i].set_device(device)

    def get_device(self) -> str:
        return self.model.trainers[0].device

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True) -> Self:
        model: MitraModel = super().load(path=path, reset_paths=reset_paths, verbose=verbose)

        if "model_version" not in model.params and model.model is not None:
            # Persist the fitted recipe for refit_full of predictors saved
            # before model_version existed, as well as for ordinary inference.
            model.params_trained.setdefault("model_version", getattr(model.model, "model_version", "1"))
        if model.params.get("fine_tune_recipe") is None and model.model is not None:
            # Keep the stored choice when an earlier predictor used an
            # implicit recipe, including a version-dependent None default.
            model.params_trained.setdefault("fine_tune_recipe", getattr(model.model, "fine_tune_recipe", "1"))
        if model._weights_saved:
            model._load_model_artifact()
            model._weights_saved = False
        return model

    @classmethod
    def download_weights(cls, repo_id: str):
        """
        Download weights for Mitra from HuggingFace from `repo_id`.
        Requires an internet connection.
        """
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=repo_id, filename="config.json")
        hf_hub_download(repo_id=repo_id, filename="model.safetensors")

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
        recipe = hyperparameters.get("fine_tune_recipe")
        if recipe is None:
            recipe = DEFAULT_FINE_TUNE_RECIPE
        if recipe == "2":
            # The network sees a capped support set, even when the fitted model
            # retains a much larger training table for sampling at inference.
            support_cap = hyperparameters.get("predict_max_samples_support") or 32768
            query_cap = hyperparameters.get("max_samples_query", 1024)
            X = X.iloc[: support_cap + query_cap]
        # Multiply by 0.9 as currently this is overly safe
        return int(
            0.9
            * max(
                cls._estimate_memory_usage_static_cpu_icl(X=X, **kwargs),
                cls._estimate_memory_usage_static_cpu_ft_icl(X=X, **kwargs),
                cls._estimate_memory_usage_static_gpu_cpu(X=X, **kwargs),
                cls._estimate_memory_usage_static_gpu_gpu(X=X, **kwargs),
            )
        )

    @classmethod
    def _estimate_memory_usage_static_cpu_icl(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        rows, features = X.shape[0], X.shape[1]

        # For very small datasets, use a more conservative estimate
        if rows * features < 100:  # Small dataset threshold
            # Use a simpler linear formula for small datasets
            cpu_memory_kb = 1.3 * (100 * rows * features + 1000000)  # 1GB base + linear scaling
        else:
            # Original formula for larger datasets
            cpu_memory_kb = 1.3 * (
                0.001748 * (rows**2) * features + 0.001206 * rows * (features**2) + 10.3482 * rows * features + 6409698
            )
        return int(cpu_memory_kb * 1e3)

    @classmethod
    def _estimate_memory_usage_static_cpu_ft_icl(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        rows, features = X.shape[0], X.shape[1]

        # For very small datasets, use a more conservative estimate
        if rows * features < 100:  # Small dataset threshold
            # Use a simpler linear formula for small datasets
            cpu_memory_kb = 1.3 * (200 * rows * features + 2000000)  # 2GB base + linear scaling
        else:
            # Original formula for larger datasets
            cpu_memory_kb = 1.3 * (
                0.001 * (rows**2) * features + 0.004541 * rows * (features**2) + 46.2974 * rows * features + 5605681
            )
        return int(cpu_memory_kb * 1e3)

    @classmethod
    def _estimate_memory_usage_static_gpu_cpu(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        rows, features = X.shape[0], X.shape[1]

        # For very small datasets, use a more conservative estimate
        if rows * features < 100:  # Small dataset threshold
            return int(2.5 * 1e9)  # 2.5GB for small datasets
        else:
            return int(5 * 1e9)  # 5GB for larger datasets

    @classmethod
    def _estimate_memory_usage_static_gpu_gpu(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        rows, features = X.shape[0], X.shape[1]

        # For very small datasets, use a more conservative estimate
        if rows * features < 100:  # Small dataset threshold
            # Use a simpler linear formula for small datasets
            gpu_memory_mb = 1.3 * (10 * rows * features + 2000)  # 2GB base + linear scaling
        else:
            # Original formula for larger datasets
            gpu_memory_mb = 1.3 * (0.05676 * rows * features + 3901)
        return int(gpu_memory_mb * 1e6)

    @classmethod
    def _class_tags(cls):
        return {
            "can_set_device": True,
            "set_device_on_save_to": None,
            "set_device_on_load": False,
        }

    def _more_tags(self) -> dict:
        tags = {"can_refit_full": True}
        return tags
