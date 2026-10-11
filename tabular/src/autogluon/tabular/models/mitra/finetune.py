"""Shared native fine-tuning estimators for Mitra v1 and v2 checkpoints.

Adapted from autogluon/mitra-finetune on Hugging Face (Apache-2.0).
Configuration belongs to each estimator; no process-wide patches are installed.
"""

from __future__ import annotations

import time

import numpy as np

from autogluon.core.utils.exceptions import TimeLimitExceeded

from ._internal.core.trainer_finetune_v2 import TrainerFinetuneV2
from .checkpoint import CHECKPOINTS, DEFAULT_MODEL_VERSION, validate_model_version
from .recipes import DEFAULT_FINE_TUNE_RECIPE, get_finetune_recipe
from .sklearn_interface import DEFAULT_DEVICE, MitraClassifier, MitraRegressor


# Bag children in the same process reuse the context that fit, as in the HF
# implementation. Device and checkpoint scope prevent unrelated fits from
# inheriting another model's memory limit.
_FITTED_CONTEXT_MEMO: dict[tuple, tuple[int, int]] = {}


class MitraFinetuneMixin:
    def __init__(
        self,
        model_type="Tab2D",
        n_estimators=1,
        device=DEFAULT_DEVICE,
        fine_tune=True,
        fine_tune_steps=50,
        metric=None,
        hf_model=None,
        patience=40,
        lr=None,
        warmup_steps=None,
        shuffle_classes=False,
        shuffle_features=False,
        use_random_transforms=False,
        random_mirror_regression=True,
        random_mirror_x=True,
        seed=0,
        verbose=True,
        weight_decay=None,
        max_samples_support=None,
        max_samples_query=None,
        predict_max_samples_support=None,
        predict_max_samples_query=None,
        predict_min_samples_support=8192,
        predict_min_samples_query=256,
        eval_max_samples_query=None,
        balance_binary_support=None,
        heldout_in_support=None,
        fine_tune_time_limit=None,
        model_version=DEFAULT_MODEL_VERSION,
        fine_tune_recipe=None,
    ):
        self.model_version = validate_model_version(model_version)
        if fine_tune_recipe is None:
            fine_tune_recipe = DEFAULT_FINE_TUNE_RECIPE
        recipe = get_finetune_recipe(fine_tune_recipe)
        self.fine_tune_recipe = fine_tune_recipe
        super().__init__(
            model_type=model_type,
            n_estimators=n_estimators,
            device=device,
            fine_tune=fine_tune,
            fine_tune_steps=fine_tune_steps,
            metric=metric if metric is not None else ("mse" if self._task == "regression" else "log_loss"),
            hf_model=hf_model if hf_model is not None else CHECKPOINTS[self.model_version][self._task],
            patience=patience,
            lr=lr if lr is not None else recipe["lr"],
            warmup_steps=warmup_steps if warmup_steps is not None else recipe["warmup_steps"],
            shuffle_classes=shuffle_classes,
            shuffle_features=shuffle_features,
            use_random_transforms=use_random_transforms,
            random_mirror_regression=random_mirror_regression,
            random_mirror_x=random_mirror_x,
            seed=seed,
            verbose=verbose,
        )
        self.weight_decay = weight_decay if weight_decay is not None else recipe["weight_decay"]
        self.max_samples_support = (
            max_samples_support if max_samples_support is not None else recipe["max_samples_support"]
        )
        self.max_samples_query = max_samples_query if max_samples_query is not None else recipe["max_samples_query"]
        self.predict_max_samples_support = (
            predict_max_samples_support
            if predict_max_samples_support is not None
            else recipe["predict_max_samples_support"]
        )
        self.predict_max_samples_query = (
            predict_max_samples_query if predict_max_samples_query is not None else recipe["predict_max_samples_query"]
        )
        self.predict_min_samples_support = predict_min_samples_support
        self.predict_min_samples_query = predict_min_samples_query
        self.eval_max_samples_query = (
            eval_max_samples_query if eval_max_samples_query is not None else recipe["eval_max_samples_query"]
        )
        self.balance_binary_support = (
            balance_binary_support if balance_binary_support is not None else recipe["balance_binary_support"]
        )
        self.heldout_in_support = (
            heldout_in_support if heldout_in_support is not None else recipe["heldout_in_support"]
        )
        self.fine_tune_time_limit = fine_tune_time_limit
        for name in (
            "n_estimators",
            "max_samples_support",
            "max_samples_query",
            "predict_max_samples_support",
            "predict_max_samples_query",
            "predict_min_samples_support",
            "predict_min_samples_query",
            "eval_max_samples_query",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer.")
        if fine_tune_time_limit is not None and fine_tune_time_limit <= 0:
            raise ValueError("fine_tune_time_limit must be positive or None.")

    def _create_config(self, task, dim_output, time_limit=None):
        cfg, model_cls = super()._create_config(task, dim_output, time_limit)
        is_binary = task == "classification" and np.unique(self.y).size == 2
        n_rows = getattr(self, "_num_samples_total", len(self.y))
        lr = self.lr if self.lr is not None else (3e-6 if is_binary and n_rows <= 16384 else 1e-5)
        budget = time_limit
        if self.fine_tune_time_limit is not None:
            budget = (
                min(time_limit, self.fine_tune_time_limit) if time_limit is not None else self.fine_tune_time_limit
            )
        cfg.hyperparams.update(
            lr=lr,
            weight_decay=self.weight_decay,
            max_samples_support=self.max_samples_support or (20480 if task == "regression" else 16384),
            max_samples_query=self.max_samples_query,
            predict_max_samples_support=self.predict_max_samples_support or (16384 if is_binary else 32768),
            predict_max_samples_query=self.predict_max_samples_query,
            predict_min_samples_support=self.predict_min_samples_support,
            predict_min_samples_query=self.predict_min_samples_query,
            eval_max_samples_query=self.eval_max_samples_query,
            balance_binary_support=self.balance_binary_support and is_binary,
            budget=budget,
            fast_loop=get_finetune_recipe(self.fine_tune_recipe)["fast_loop"],
            fast_predict=get_finetune_recipe(self.fine_tune_recipe)["fast_predict"],
        )
        pending = getattr(self, "_fitted_context_key", None)
        if pending is not None and len(pending) == 6:
            requested = (int(cfg.hyperparams["max_samples_support"]), int(cfg.hyperparams["max_samples_query"]))
            self._fitted_context_key = pending + requested
            fitted = _FITTED_CONTEXT_MEMO.get(self._fitted_context_key)
            if fitted is not None and fitted < requested:
                cfg.hyperparams["max_samples_support"], cfg.hyperparams["max_samples_query"] = fitted
        return cfg, model_cls

    def _train_ensemble(self, X_train, y_train, X_valid, y_valid, task, dim_output, n_classes=0, time_limit=None):
        if get_finetune_recipe(self.fine_tune_recipe)["fast_loop"]:
            shape = np.shape(X_train)
            self._fitted_context_key = (
                str(task),
                int(shape[1]) if len(shape) > 1 else 1,
                int(round(int(shape[0]) / 256)),
                str(self.device),
                str(self.hf_model),
                self.model_version,
            )
        try:
            result = super()._train_ensemble(
                X_train, y_train, X_valid, y_valid, task, dim_output, n_classes=n_classes, time_limit=time_limit
            )
            key = getattr(self, "_fitted_context_key", None)
            if key is not None and len(key) == 8 and self.trainers:
                hp = self.trainers[0].cfg.hyperparams
                _FITTED_CONTEXT_MEMO[key] = (int(hp["max_samples_support"]), int(hp["max_samples_query"]))
            return result
        finally:
            self.__dict__.pop("_fitted_context_key", None)

    def _get_trainer_cls(self):
        return TrainerFinetuneV2

    def _configure_pretrained_model(self, cfg, model):
        super()._configure_pretrained_model(cfg, model)
        cfg.hyperparams["dim_output"] = model.dim_output
        if self.task == "regression":
            cfg.hyperparams["regression_loss"] = "mse" if model.dim_output == 1 else "cross_entropy"
        elif model.dim_output < np.unique(self.y).size:
            raise ValueError("The classification checkpoint has fewer outputs than the training data has classes.")

    def extend_support(self, X, y):
        """Extend inference context after the bag child's OOF predictions have been recorded."""
        self.X = np.concatenate([self.X, np.asarray(X)], axis=0)
        self.y = np.concatenate([self.y, np.asarray(y)], axis=0)


class MitraFinetuneClassifier(MitraFinetuneMixin, MitraClassifier):
    _task = "classification"

    def fit(self, X, y, X_val=None, y_val=None, time_limit=None):
        self.hierarchy_ = None
        classes = np.unique(y)
        if classes.size <= 10:
            return super().fit(X, y, X_val=X_val, y_val=y_val, time_limit=time_limit)
        from ._internal.data.class_hierarchy import ClassHierarchy

        X, y = np.asarray(X), np.asarray(y)
        if not np.array_equal(classes, np.arange(classes.size)):
            raise ValueError("Mitra hierarchy labels must be contiguous integers starting at zero.")
        if X_val is None:
            X_train, X_valid, y_train, y_valid = self._split_data(X, y)
        else:
            X_train, y_train, X_valid, y_valid = X, y, np.asarray(X_val), np.asarray(y_val)
        self.X, self.y = X, y
        self.hierarchy_ = ClassHierarchy(classes.size, max_classes=10, seed=self.seed)
        started = time.monotonic()
        for i, node in enumerate(self.hierarchy_.nodes):
            remaining = None if time_limit is None else time_limit - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                raise TimeLimitExceeded
            train_mask = node.mask(y_train)
            val_mask = node.mask(y_valid)
            estimator = MitraFinetuneClassifier(**self.get_params())
            estimator._num_samples_total = getattr(self, "_num_samples_total", len(y))
            estimator.fit(
                X_train[train_mask],
                node.encode(y_train[train_mask]),
                X_val=X_valid[val_mask] if val_mask.any() else None,
                y_val=node.encode(y_valid[val_mask]) if val_mask.any() else None,
                time_limit=None if remaining is None else remaining / (len(self.hierarchy_.nodes) - i),
            )
            node.estimator = estimator
        self.trainers = [trainer for node in self.hierarchy_.nodes for trainer in node.estimator.trainers]
        self.train_time = time.monotonic() - started
        return self

    def predict_proba(self, X):
        if self.hierarchy_ is not None:
            return self.hierarchy_.predict_proba(np.asarray(X))
        # HF retains the base classifier's float32 normalization and ensemble
        # reduction. A different softmax formula can alter probability ties.
        return super().predict_proba(X)

    def extend_support(self, X, y):
        if self.hierarchy_ is not None:
            X, y = np.asarray(X), np.asarray(y)
            for node in self.hierarchy_.nodes:
                mask = node.mask(y)
                if mask.any():
                    node.estimator.extend_support(X[mask], node.encode(y[mask]))
        super().extend_support(X, y)


class MitraFinetuneRegressor(MitraFinetuneMixin, MitraRegressor):
    _task = "regression"
