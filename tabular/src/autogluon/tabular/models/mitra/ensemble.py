"""Mitra's node-wise bagging for classification above the native head width."""

from __future__ import annotations

import copy
import os
import random
import shutil
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from autogluon.common.utils.random import get_numpy_seed
from autogluon.core.models import AbstractModel, BaggedEnsembleModel, StackerEnsembleModel
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

from ._internal.data.class_hierarchy import ClassHierarchy, hierarchy_node_count
from .mitra_v2_model import MitraV2Model


@contextmanager
def _node_random_context(*, device, seed=None, generators=()):
    """Isolate a node's seeded fit, or the extra OOF pass absent from HF."""
    import torch

    device = torch.device(device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    states = [(generator, generator.get_state()) for generator in generators]
    with torch.random.fork_rng(devices=devices):
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(get_numpy_seed(seed))
                torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for generator, state in states:
                generator.set_state(state)


class _MitraHierarchyNodeModel(MitraV2Model):
    """Use official validation for node checkpointing, preserving honest OOF."""

    def validate_fit_args(self, X, **kwargs):
        if not self._fit_constraints_validated_upstream:
            self._node_validation = (kwargs["X_val"], kwargs["y_val"])
        return super().validate_fit_args(X=X, **kwargs)

    def _fit(self, X, y, X_val=None, y_val=None, **kwargs):
        validation = self._node_validation
        super()._fit(X=X, y=y, X_val=validation[0], y_val=validation[1], **kwargs)
        if self._fit_constraints_validated_upstream and self.model.heldout_in_support:
            # The fold must remain excluded while AutoGluon computes its OOF
            # predictions. Only that fold, never official validation, is added
            # by MitraV2Model's existing post-scoring cleanup.
            self._pending_fold_support = (
                self.preprocess(X_val).to_numpy(copy=True),
                y_val.to_numpy(copy=True),
            )
        self._node_validation = None

    def _predict_proba(self, X, **kwargs):
        if self.val_score is not None:
            return super()._predict_proba(X, **kwargs)
        # HF saves a fitted child without predicting its held-out fold.
        # AutoGluon needs that OOF prediction, but it must not advance either
        # the child's support sampler or the next fold's training RNG stream.
        with _node_random_context(
            device=self.model.device, generators=[trainer.rng for trainer in self.model.trainers]
        ):
            return super()._predict_proba(X, **kwargs)


class _MitraHierarchyModel(AbstractModel):
    """One full-class model whose classifiers are independently bagged nodes.

    Keeping the hierarchy as a single outer child preserves AutoGluon's
    full-class prediction and OOF interfaces. Individual node bags remain
    ordinary AutoGluon bags and are loaded sequentially during prediction.
    """

    default_resources_physical_cores_only = True
    default_num_gpus = 1
    _supported_problem_types = ["multiclass"]

    def _more_tags(self):
        tags = super()._more_tags()
        tags.update(valid_oof=True, can_refit_full=False)
        return tags

    def _estimate_memory_usage(self, X, **kwargs):
        return MitraV2Model.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=self._node_hyperparameters,
            **kwargs,
        )

    def _load_node(self, node):
        return BaggedEnsembleModel.load(os.path.join(self.path, "nodes", node.bag_name), verbose=False)

    @staticmethod
    def _as_multiclass(probabilities):
        probabilities = np.asarray(probabilities)
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1 - probabilities, probabilities])
        return probabilities

    def _node_oof(self, bag, node_mask, X, root_validation_indices):
        """Every node prediction excludes the queried row from fit and support.

        For rows in a node, reuse OOF recorded before support extension. Rows
        outside that node were never part of its training table; assign them
        to the corresponding root fold, rather than predicting them with a
        model that could have seen their target.
        """
        own = self._as_multiclass(bag.predict_proba_oof())
        result = np.zeros((len(X), own.shape[1]), dtype=np.float64)
        result[node_mask] = own
        if node_mask.all():
            return result
        counts = np.zeros(len(X), dtype=np.int32)
        for child_name, root_indices in zip(bag.models, root_validation_indices):
            query_indices = root_indices[~node_mask[root_indices]]
            if len(query_indices):
                child = bag.load_child(child_name)
                probabilities = self._as_multiclass(child.predict_proba(X.iloc[query_indices]))
                result[query_indices] += probabilities
                counts[query_indices] += 1
                del child
        if np.any(counts[~node_mask] == 0):
            raise RuntimeError("A hierarchy OOF row has no prediction.")
        result[~node_mask] /= counts[~node_mask, None]
        return result

    def _fit(self, X, y, time_limit=None, num_cpus=1, num_gpus=0, **kwargs):
        X_val, y_val = self._official_validation
        if X_val is None or y_val is None:
            raise ValueError("Node-wise Mitra bagging requires an external validation set.")
        classes = np.unique(y)
        if not np.array_equal(classes, np.arange(self.num_classes)):
            raise ValueError("Mitra hierarchy labels must be contiguous integers starting at zero.")
        unknown = np.setdiff1d(np.unique(y_val), classes)
        if len(unknown):
            raise ValueError(f"Validation contains unknown classes: {unknown.tolist()}")
        y_array, y_val_array = np.asarray(y), np.asarray(y_val)
        self.model = ClassHierarchy(self.num_classes, seed=self._hierarchy_seed)
        self._node_model_root = self.path
        node_oof = {}
        root_validation_indices = None
        started = time.monotonic()
        for index, node in enumerate(self.model.nodes):
            remaining = None if time_limit is None else time_limit - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                raise TimeLimitExceeded
            train_mask, val_mask = node.mask(y_array), node.mask(y_val_array)
            X_node, X_val_node = X.loc[train_mask], X_val.loc[val_mask]
            y_node = pd.Series(node.encode(y_array[train_mask]), index=X_node.index)
            y_val_node = pd.Series(node.encode(y_val_array[val_mask]), index=X_val_node.index)
            if not len(X_val_node):
                raise ValueError("A hierarchy node has no external validation rows.")
            # HF fits a predictor at every node. Its feature generator sees
            # both training and official validation features, as in the
            # ordinary AutoGluon learner. Fitting only on training rows would
            # change rare categories and columns that become locally binary.
            node.feature_generator = AutoMLPipelineFeatureGenerator(verbosity=0)
            features = node.feature_generator.fit_transform(
                pd.concat([X_node, X_val_node], ignore_index=True),
                y=pd.concat([y_node, y_val_node], ignore_index=True),
            )
            X_val_node = features.iloc[len(X_node) :].set_index(X_val_node.index)
            X_node = features.iloc[: len(X_node)].set_index(X_node.index)
            node.bag_name = f"node_{index:03d}"
            hyperparameters = copy.deepcopy(self._node_hyperparameters)
            if "seed" in hyperparameters:
                hyperparameters["seed"] = self._hierarchy_seed + index
            problem_type = "binary" if len(node.groups) == 2 else "multiclass"
            child = _MitraHierarchyNodeModel(
                path=os.path.join(self.path, "nodes"),
                name=node.bag_name,
                problem_type=problem_type,
                eval_metric="log_loss",
                hyperparameters=hyperparameters,
            )
            bag_hyperparameters = copy.deepcopy(self._node_bag_hyperparameters)
            if bag_hyperparameters.get("fold_fitting_strategy", "auto") == "auto":
                bag_hyperparameters["fold_fitting_strategy"] = "sequential_local"
            for strategy in ("fold_fitting_strategy_cpu", "fold_fitting_strategy_gpu"):
                if bag_hyperparameters.get(strategy) == "auto":
                    bag_hyperparameters[strategy] = "sequential_local"
            bag_hyperparameters["model_random_seed"] = self._hierarchy_seed + index
            bag_hyperparameters["ag.max_time_limit"] = self._node_time_limit
            bag = BaggedEnsembleModel(
                model_base=child,
                path=os.path.join(self.path, "nodes"),
                name=node.bag_name,
                random_state=self._bag_random_state,
                hyperparameters=bag_hyperparameters,
            )
            # HF's default API fits each node in a separately seeded process.
            # Reproduce that stream locally without changing the caller's RNG.
            with _node_random_context(
                device=hyperparameters.get("device") or ("cuda" if num_gpus else "cpu"),
                seed=self._hierarchy_seed + index,
            ):
                bag.fit(
                    X=X_node,
                    y=y_node,
                    X_val=X_val_node,
                    y_val=y_val_node,
                    k_fold=self._node_folds,
                    n_repeats=self._node_repeats,
                    time_limit=remaining,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    feature_metadata=node.feature_generator.feature_metadata,
                    verbosity=kwargs.get("verbosity", 2),
                )
            if root_validation_indices is None:
                root_validation_indices = bag.get_oof_fold_val_idx(X=X_node, y=y_node)
            if len(bag.models) != len(root_validation_indices):
                raise ValueError("Hierarchy nodes must use the same number of bag folds.")
            node_oof[node.bag_name] = self._node_oof(
                bag, train_mask, node.feature_generator.transform(X), root_validation_indices
            )
            bag.save(verbose=False)
            del bag
        self._oof = self.model.combine_probabilities(len(X), lambda node: node_oof[node.bag_name]).astype(np.float32)
        self._official_validation = None

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X)

        def predict_node(node):
            bag = self._load_node(node)
            return self._as_multiclass(bag.predict_proba(node.feature_generator.transform(X)))

        # HF re-predicts only the test tail after adding each held-out fold to
        # support. A validation prefix would change capped support sampling.
        return self.model.combine_probabilities(len(X), predict_node)

    def predict_proba_oof(self, X=None, y=None, **kwargs):
        if X is not None and len(X) != len(self._oof):
            raise ValueError("OOF predictions correspond to the original training rows.")
        return self._oof

    def save(self, path=None, verbose=True):
        target = path or self.path
        original = getattr(self, "_node_model_root", self.path)
        if os.path.abspath(original) != os.path.abspath(target):
            shutil.copytree(
                os.path.join(original, "nodes"),
                os.path.join(target, "nodes"),
                dirs_exist_ok=True,
            )
        self._node_model_root = target
        try:
            return super().save(path=target, verbose=verbose)
        finally:
            self._node_model_root = original

    @classmethod
    def load(cls, path, reset_paths=True, verbose=True):
        model = super().load(path, reset_paths=reset_paths, verbose=verbose)
        model._node_model_root = model.path
        return model

    def reduce_memory_size(self, remove_fit=True, remove_info=False, requires_save=True, **kwargs):
        super().reduce_memory_size(
            remove_fit=remove_fit, remove_info=remove_info, requires_save=requires_save, **kwargs
        )
        if self.is_fit() and remove_fit and requires_save:
            for node in self.model.nodes:
                bag = self._load_node(node)
                bag.reduce_memory_size(remove_fit=True, remove_info=remove_info, requires_save=True)
                bag.save(verbose=False)


class MitraStackerEnsembleModel(StackerEnsembleModel):
    """Retain ordinary bagging except for HF's externally validated hierarchy."""

    def _get_fold_fitting_strategy(self, model_base, num_gpus):
        from autogluon.core.models.ensemble.fold_fitting_strategy import SequentialLocalFoldFittingStrategy

        from .fold_fitting_strategy import MitraSequentialLocalFoldFittingStrategy

        strategy = super()._get_fold_fitting_strategy(model_base=model_base, num_gpus=num_gpus)
        if strategy is SequentialLocalFoldFittingStrategy and isinstance(model_base, MitraV2Model):
            return MitraSequentialLocalFoldFittingStrategy
        return strategy

    def _fit_single(self, X, y, model_base, use_child_oof, skip_oof=False, **kwargs):
        if not self._node_bagging or not use_child_oof or skip_oof:
            return super()._fit_single(
                X=X, y=y, model_base=model_base, use_child_oof=use_child_oof, skip_oof=skip_oof, **kwargs
            )
        # The hierarchy has already assembled its OOF from node bags. The
        # generic path estimates OOF cost by timing predict_proba on 500 rows;
        # it can reject a completed fit even though retrieving cached OOF
        # predictions is immediate.
        super()._fit_single(X=X, y=y, model_base=model_base, use_child_oof=True, skip_oof=True, **kwargs)
        started = time.monotonic()
        self._oof_pred_proba = model_base.predict_proba_oof(X=X, y=y)
        self._oof_pred_model_repeats = np.ones(len(X), dtype=np.uint8)
        self._child_oof = True
        model_base.predict_time = time.monotonic() - started
        model_base.val_score = model_base.score_with_y_pred_proba(y=y, y_pred_proba=self._oof_pred_proba)
        self.predict_time = (self.predict_time or 0) + model_base.predict_time
        if self.low_memory:
            self.save_child(model_base)

    def _preprocess_fit_args(self, **kwargs):
        self._node_bagging = (
            self.problem_type == "multiclass"
            and self.num_classes is not None
            and self.num_classes > 10
            and kwargs.get("X_val") is not None
            and kwargs.get("groups") is None
            and kwargs.get("X_pseudo") is None
            and self.params.get("custom_splits") is None
            and (kwargs.get("k_fold") is None or kwargs["k_fold"] >= 2)
            and kwargs.get("k_fold_start", 0) == 0
            and kwargs.get("n_repeat_start", 0) == 0
            and kwargs.get("k_fold_end") in (None, kwargs.get("k_fold"))
        )
        if "max_time_limit" not in self._user_params_aux:
            count = hierarchy_node_count(self.num_classes) if self._node_bagging else 1
            self.params_aux["max_time_limit"] = 3600 * count
        return super()._preprocess_fit_args(**kwargs)

    def _fit(self, X, y, X_val=None, y_val=None, k_fold=None, n_repeats=1, **kwargs):
        if not self._node_bagging:
            return super()._fit(
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                k_fold=k_fold,
                n_repeats=n_repeats,
                **kwargs,
            )
        if self.is_fit():
            return self
        if kwargs.get("k_fold_start", 0) or kwargs.get("n_repeat_start", 0):
            raise ValueError("A Mitra hierarchy must start with the first bag fold and repeat.")
        base = self._get_model_base()
        parameters = base.get_params()
        node_parameters = copy.deepcopy(parameters["hyperparameters"])
        hierarchy = _MitraHierarchyModel(**parameters)
        hierarchy._official_validation = (self.preprocess(X_val, preprocess_nonadaptive=False, fit=False), y_val)
        hierarchy._node_hyperparameters = node_parameters
        hierarchy._node_bag_hyperparameters = copy.deepcopy(self.params)
        hierarchy._node_bag_hyperparameters.pop("use_child_oof", None)
        hierarchy._node_bag_hyperparameters.pop("refit_folds", None)
        hierarchy._node_bag_hyperparameters["save_bag_folds"] = True
        hierarchy._node_folds = k_fold or 5
        hierarchy._node_repeats = n_repeats
        hierarchy._node_time_limit = 3600
        hierarchy._bag_random_state = self._random_state
        hierarchy._hierarchy_seed = node_parameters.get("seed", self.params["model_random_seed"])
        self._child_type = _MitraHierarchyModel
        self.model_base = hierarchy
        self.params["use_child_oof"] = True
        return super()._fit(
            X=X,
            y=y,
            k_fold=k_fold,
            n_repeats=n_repeats,
            X_val=X_val,
            y_val=y_val,
            **kwargs,
        )
