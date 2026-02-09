from __future__ import annotations

import copy
import logging
import os
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.types import R_FLOAT, S_STACK
from autogluon.common.utils.distribute_utils import DistributedContext
from autogluon.common.utils.lite import disable_if_lite_mode
from autogluon.common.utils.log_utils import convert_time_in_s_to_log_friendly, reset_logger_for_remote_call
from autogluon.common.utils.resource_utils import ResourceManager, get_resource_manager
from autogluon.common.utils.try_import import try_import_ray, try_import_torch
from autogluon.core.augmentation.distill_utils import augment_data, format_distillation_labels
from autogluon.core.calibrate import calibrate_decision_threshold
from autogluon.core.calibrate.conformity_score import compute_conformity_score
from autogluon.core.calibrate.temperature_scaling import apply_temperature_scaling, tune_temperature_scaling
from autogluon.core.callbacks import AbstractCallback
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REFIT_FULL_NAME, REGRESSION, SOFTCLASS
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerMulticlassToBinary
from autogluon.core.metrics import Scorer, compute_metric, get_metric
from autogluon.core.models import (
    AbstractModel,
    BaggedEnsembleModel,
    GreedyWeightedEnsembleModel,
    SimpleWeightedEnsembleModel,
    StackerEnsembleModel,
    WeightedEnsembleModel,
)
from autogluon.core.pseudolabeling.pseudolabeling import assert_pseudo_column_match
from autogluon.core.ray.distributed_jobs_managers import ParallelFitManager
from autogluon.core.trainer import AbstractTrainer
from autogluon.core.trainer.utils import process_hyperparameters
from autogluon.core.utils import (
    compute_permutation_feature_importance,
    convert_pred_probas_to_df,
    default_holdout_frac,
    extract_column,
    generate_train_test_split,
    get_pred_from_proba,
    infer_eval_metric,
)
from autogluon.core.utils.exceptions import (
    InsufficientTime,
    NoGPUError,
    NoStackFeatures,
    NotEnoughCudaMemoryError,
    NotEnoughMemoryError,
    NotValidStacker,
    NoValidFeatures,
    TimeLimitExceeded,
)
from autogluon.core.utils.feature_selection import FeatureSelector
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl

logger = logging.getLogger(__name__)


class AbstractTabularTrainer(AbstractTrainer[AbstractModel]):
    """
    AbstractTabularTrainer contains logic to train a variety of models under a variety of constraints and automatically generate a multi-layer stack ensemble.
    Beyond the basic functionality, it also has support for model refitting, distillation, pseudo-labelling, unlabeled data, and much more.

    It is not recommended to directly use Trainer. Instead, use Predictor or Learner which internally uses Trainer.
    This documentation is for developers. Users should avoid this class.

    Due to the complexity of the logic within this class, a text description will not give the full picture.
    It is recommended to carefully read the code and use a debugger to understand how it works.

    AbstractTabularTrainer makes much fewer assumptions about the problem than Learner and Predictor.
    It expects these ambiguities to have already been resolved upstream. For example, problem_type, feature_metadata, num_classes, etc.

    Parameters
    ----------
    path : str
        Path to save and load trainer artifacts to disk.
        Path should end in `/` or `os.path.sep()`.
    problem_type : str
        One of ['binary', 'multiclass', 'regression', 'quantile', 'softclass']
    num_classes : int
        The number of classes in the problem.
        If problem_type is in ['regression', 'quantile'], this must be None.
        If problem_type is 'binary', this must be 2.
        If problem_type is in ['multiclass', 'softclass'], this must be >= 2.
    feature_metadata : FeatureMetadata
        FeatureMetadata for X. Sent to each model during fit.
    eval_metric : Scorer, default = None
        Metric to optimize. If None, a default metric is used depending on the problem_type.
    quantile_levels : list[float] | np.ndarray, default = None
        # TODO: Add documentation, not documented in Predictor.
        Only used when problem_type=quantile
    low_memory : bool, default = True
        Deprecated parameter, likely to be removed in future versions.
        If True, caches models to disk separately instead of containing all models within memory.
        If False, may cause a variety of bugs.
    k_fold : int, default = 0
        If <2, then non-bagged mode is used.
        If >= 2, then bagged mode is used with num_bag_folds == k_fold for each model.
        Bagged mode changes the way models are trained and ensembled.
        Bagged mode enables multi-layer stacking and repeated bagging.
    n_repeats : int, default = 1
        The maximum repeats of bagging to do when in bagged mode.
        Larger values take linearly longer to train and infer, but improves quality slightly.
    sample_weight : str, default = None
        Column name of the sample weight in X
    weight_evaluation : bool, default = False
        If True, the eval_metric is calculated with sample_weight incorporated into the score.
    save_data : bool, default = True
        Whether to cache the data (X, y, X_val, y_val) to disk.
        Required for a variety of advanced post-fit functionality.
        It is recommended to keep as True.
    random_state : int, default = 0
        Random state for data splitting in bagged mode.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
    raise_on_model_failure : bool, default = False
        If True, Trainer will raise on any exception during model training.
            This is ideal when using a debugger during development.
        If False, Trainer will try to skip to the next model if an exception occurred during model training.

        .. versionadded:: 1.3.0
    """

    distill_stackname = "distill"  # name of stack-level for distilled student models

    def __init__(
        self,
        path: str,
        *,
        problem_type: str,
        num_classes: int | None = None,
        feature_metadata: FeatureMetadata | None = None,
        eval_metric: Scorer | None = None,
        quantile_levels: list[float] | np.ndarray | None = None,
        low_memory: bool = True,
        k_fold: int = 0,
        n_repeats: int = 1,
        sample_weight: str | None = None,
        weight_evaluation: bool = False,
        save_data: bool = False,
        random_state: int = 0,
        verbosity: int = 2,
        raise_on_model_failure: bool = False,
    ):
        super().__init__(
            path=path,
            low_memory=low_memory,
            save_data=save_data,
        )
        self._validate_num_classes(num_classes=num_classes, problem_type=problem_type)
        self._validate_quantile_levels(quantile_levels=quantile_levels, problem_type=problem_type)
        self.problem_type = problem_type
        self.feature_metadata = feature_metadata

        #: Integer value added to the stack level to get the random_state for kfold splits or the train/val split if bagging is disabled
        self.random_state = random_state
        self.verbosity = verbosity
        self.raise_on_model_failure = raise_on_model_failure

        # TODO: consider redesign where Trainer doesn't need sample_weight column name and weights are separate from X
        self.sample_weight = sample_weight
        self.weight_evaluation = weight_evaluation
        if eval_metric is not None:
            self.eval_metric = eval_metric
        else:
            self.eval_metric = infer_eval_metric(problem_type=self.problem_type)

        logger.log(
            20, f"AutoGluon will gauge predictive performance using evaluation metric: '{self.eval_metric.name}'"
        )
        if not self.eval_metric.greater_is_better_internal:
            logger.log(
                20,
                "\tThis metric's sign has been flipped to adhere to being higher_is_better. "
                "The metric score can be multiplied by -1 to get the metric value.",
            )
        if not (self.eval_metric.needs_pred or self.eval_metric.needs_quantile):
            logger.log(
                20,
                "\tThis metric expects predicted probabilities rather than predicted class labels, "
                "so you'll need to use predict_proba() instead of predict()",
            )

        logger.log(20, "\tTo change this, specify the eval_metric parameter of Predictor()")
        self.num_classes = num_classes
        self.quantile_levels = quantile_levels

        #: will be set to True if feature-pruning is turned on.
        self.feature_prune = False

        self.bagged_mode = True if k_fold >= 2 else False
        if self.bagged_mode:
            #: int number of folds to do model bagging, < 2 means disabled
            self.k_fold = k_fold
            self.n_repeats = n_repeats
        else:
            self.k_fold = 0
            self.n_repeats = 1

        #: Internal float of the total time limit allowed for a given fit call. Used in logging statements.
        self._time_limit = None
        #: Internal timestamp of the time training started for a given fit call. Used in logging statements.
        self._time_train_start = None
        #: Same as `self._time_train_start` except it is not reset to None after the fit call completes.
        self._time_train_start_last = None

        self._num_rows_train = None
        self._num_cols_train = None
        self._num_rows_val = None
        self._num_rows_test = None

        self.is_data_saved = False
        self._X_saved = False
        self._y_saved = False
        self._X_val_saved = False
        self._y_val_saved = False

        #: custom split indices
        self._groups = None

        #: whether to treat regression predictions as class-probabilities (during distillation)
        self._regress_preds_asprobas = False

        #: dict of model name -> model failure metadata
        self._models_failed_to_train_errors = dict()

        # self._exceptions_list = []  # TODO: Keep exceptions list for debugging during benchmarking.

        self.callbacks: list[AbstractCallback] = []
        self._callback_early_stop = False

    @property
    def _path_attr(self) -> str:
        """Path to cached model graph attributes"""
        return os.path.join(self.path_utils, "attr")

    @property
    def has_val(self) -> bool:
        """Whether the trainer uses validation data"""
        return self._num_rows_val is not None

    @property
    def num_rows_val_for_calibration(self) -> int:
        """The number of rows available to optimize model calibration"""
        if self._num_rows_val is not None:
            return self._num_rows_val
        elif self.bagged_mode:
            assert self._num_rows_train is not None
            return self._num_rows_train
        else:
            return 0

    @property
    def time_left(self) -> float | None:
        """
        Remaining time left in the fit call.
        None if time_limit was unspecified.
        """
        if self._time_train_start is None:
            return None
        elif self._time_limit is None:
            return None
        time_elapsed = time.time() - self._time_train_start
        time_left = self._time_limit - time_elapsed
        return time_left

    @property
    def logger(self) -> logging.Logger:
        return logger

    def log(self, level: int, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)

    def load_X(self):
        if self._X_saved:
            path = os.path.join(self.path_data, "X.pkl")
            return load_pkl.load(path=path)
        return None

    def load_X_val(self):
        if self._X_val_saved:
            path = os.path.join(self.path_data, "X_val.pkl")
            return load_pkl.load(path=path)
        return None

    def load_y(self):
        if self._y_saved:
            path = os.path.join(self.path_data, "y.pkl")
            return load_pkl.load(path=path)
        return None

    def load_y_val(self):
        if self._y_val_saved:
            path = os.path.join(self.path_data, "y_val.pkl")
            return load_pkl.load(path=path)
        return None

    def load_data(self):
        X = self.load_X()
        y = self.load_y()
        X_val = self.load_X_val()
        y_val = self.load_y_val()

        return X, y, X_val, y_val

    def save_X(self, X, verbose=True):
        path = os.path.join(self.path_data, "X.pkl")
        save_pkl.save(path=path, object=X, verbose=verbose)
        self._X_saved = True

    def save_X_val(self, X, verbose=True):
        path = os.path.join(self.path_data, "X_val.pkl")
        save_pkl.save(path=path, object=X, verbose=verbose)
        self._X_val_saved = True

    def save_X_test(self, X, verbose=True):
        path = os.path.join(self.path_data, "X_test.pkl")
        save_pkl.save(path=path, object=X, verbose=verbose)
        self._X_test_saved = True

    def save_y(self, y, verbose=True):
        path = os.path.join(self.path_data, "y.pkl")
        save_pkl.save(path=path, object=y, verbose=verbose)
        self._y_saved = True

    def save_y_val(self, y, verbose=True):
        path = os.path.join(self.path_data, "y_val.pkl")
        save_pkl.save(path=path, object=y, verbose=verbose)
        self._y_val_saved = True

    def save_y_test(self, y, verbose=True):
        path = os.path.join(self.path_data, "y_test.pkl")
        save_pkl.save(path=path, object=y, verbose=verbose)
        self._y_test_saved = True

    def get_model_names(
        self,
        stack_name: list[str] | str | None = None,
        level: list[int] | int | None = None,
        can_infer: bool | None = None,
        models: list[str] | None = None,
    ) -> list[str]:
        if models is None:
            models = list(self.model_graph.nodes)
        if stack_name is not None:
            if not isinstance(stack_name, list):
                stack_name = [stack_name]
            node_attributes: dict = self.get_models_attribute_dict(attribute="stack_name", models=models)
            models = [model_name for model_name in models if node_attributes[model_name] in stack_name]
        if level is not None:
            if not isinstance(level, list):
                level = [level]
            node_attributes: dict = self.get_models_attribute_dict(attribute="level", models=models)
            models = [model_name for model_name in models if node_attributes[model_name] in level]
        # TODO: can_infer is technically more complicated, if an ancestor can't infer then the model can't infer.
        if can_infer is not None:
            node_attributes = self.get_models_attribute_full(attribute="can_infer", models=models, func=min)
            models = [model for model in models if node_attributes[model] == can_infer]
        return models

    def get_max_level(self, stack_name: str | None = None, models: list[str] | None = None) -> int:
        models = self.get_model_names(stack_name=stack_name, models=models)
        models_attribute_dict = self.get_models_attribute_dict(attribute="level", models=models)
        if models_attribute_dict:
            return max(list(models_attribute_dict.values()))
        else:
            return -1

    def construct_model_templates(self, hyperparameters: dict[str, Any]) -> tuple[list[AbstractModel], dict]:
        """Constructs a list of unfit models based on the hyperparameters dict."""
        raise NotImplementedError

    def construct_model_templates_distillation(
        self, hyperparameters: dict, **kwargs
    ) -> tuple[list[AbstractModel], dict]:
        """Constructs a list of unfit models based on the hyperparameters dict for softclass distillation."""
        raise NotImplementedError

    def get_model_level(self, model_name: str) -> int:
        return self.get_model_attribute(model=model_name, attribute="level")

    def fit(self, X, y, hyperparameters: dict, X_val=None, y_val=None, **kwargs):
        raise NotImplementedError

    # TODO: Enable easier re-mapping of trained models -> hyperparameters input (They don't share a key since name can change)
    def train_multi_levels(
        self,
        X,
        y,
        hyperparameters: dict,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        X_unlabeled=None,
        base_model_names: list[str] | None = None,
        core_kwargs: dict | None = None,
        aux_kwargs: dict | None = None,
        level_start=1,
        level_end=1,
        time_limit=None,
        name_suffix: str | None = None,
        relative_stack=True,
        level_time_modifier=0.333,
        infer_limit=None,
        infer_limit_batch_size=None,
        callbacks: list[AbstractCallback] | None = None,
    ) -> list[str]:
        """
        Trains a multi-layer stack ensemble using the input data on the hyperparameters dict input.
            hyperparameters is used to determine the models used in each stack layer.
        If continuing a stack ensemble with level_start>1, ensure that base_model_names is set to the appropriate base models that will be used by the level_start level models.
        Trains both core and aux models.
            core models are standard models which are fit on the data features. Core models will also use model predictions if base_model_names was specified or if level != 1.
            aux models are ensemble models which only use the predictions of core models as features. These models never use the original features.

        level_time_modifier : float, default 0.333
            The amount of extra time given relatively to early stack levels compared to later stack levels.
            If 0, then all stack levels are given 100%/L of the time, where L is the number of stack levels.
            If 1, then all stack levels are given 100% of the time, meaning if the first level uses all of the time given to it, the other levels won't train.
            Time given to a level = remaining_time / remaining_levels * (1 + level_time_modifier), capped by total remaining time.

        Returns a list of the model names that were trained from this method call, in order of fit.
        """
        self._fit_setup(time_limit=time_limit, callbacks=callbacks)
        time_train_start = self._time_train_start
        assert time_train_start is not None

        if self.callbacks:
            callback_classes = [c.__class__.__name__ for c in self.callbacks]
            logger.log(20, f"User-specified callbacks ({len(self.callbacks)}): {callback_classes}")

        hyperparameters = self._process_hyperparameters(hyperparameters=hyperparameters)

        if relative_stack:
            if level_start != 1:
                raise AssertionError(
                    f"level_start must be 1 when `relative_stack=True`. (level_start = {level_start})"
                )
            level_add = 0
            if base_model_names:
                max_base_model_level = self.get_max_level(models=base_model_names)
                level_start = max_base_model_level + 1
                level_add = level_start - 1
                level_end += level_add
            if level_start != 1:
                hyperparameters_relative = {}
                for key in hyperparameters:
                    if isinstance(key, int):
                        hyperparameters_relative[key + level_add] = hyperparameters[key]
                    else:
                        hyperparameters_relative[key] = hyperparameters[key]
                hyperparameters = hyperparameters_relative

        core_kwargs = {} if core_kwargs is None else core_kwargs.copy()
        aux_kwargs = {} if aux_kwargs is None else aux_kwargs.copy()

        self._callbacks_setup(
            X=X,
            y=y,
            hyperparameters=hyperparameters,
            X_val=X_val,
            y_val=y_val,
            X_unlabeled=X_unlabeled,
            level_start=level_start,
            level_end=level_end,
            time_limit=time_limit,
            base_model_names=base_model_names,
            core_kwargs=core_kwargs,
            aux_kwargs=aux_kwargs,
            name_suffix=name_suffix,
            level_time_modifier=level_time_modifier,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
        )
        # TODO: Add logic for callbacks to specify that the rest of the trainer logic should be skipped in the case where they are overriding the trainer logic.

        model_names_fit = []
        if level_start != level_end:
            logger.log(
                20,
                f"AutoGluon will fit {level_end - level_start + 1} stack levels (L{level_start} to L{level_end}) ...",
            )
        for level in range(level_start, level_end + 1):
            core_kwargs_level = core_kwargs.copy()
            aux_kwargs_level = aux_kwargs.copy()
            full_weighted_ensemble = (
                aux_kwargs_level.pop("fit_full_last_level_weighted_ensemble", True)
                and (level == level_end)
                and (level > 1)
            )
            additional_full_weighted_ensemble = (
                aux_kwargs_level.pop("full_weighted_ensemble_additionally", False) and full_weighted_ensemble
            )
            if time_limit is not None:
                time_train_level_start = time.time()
                levels_left = level_end - level + 1
                time_left = time_limit - (time_train_level_start - time_train_start)
                time_limit_for_level = min(time_left / levels_left * (1 + level_time_modifier), time_left)
                time_limit_core = time_limit_for_level
                time_limit_aux = max(
                    time_limit_for_level * 0.1, min(time_limit, 360)
                )  # Allows aux to go over time_limit, but only by a small amount
                core_kwargs_level["time_limit"] = core_kwargs_level.get("time_limit", time_limit_core)
                aux_kwargs_level["time_limit"] = aux_kwargs_level.get("time_limit", time_limit_aux)
            base_model_names, aux_models = self.stack_new_level(
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                X_unlabeled=X_unlabeled,
                models=hyperparameters,
                level=level,
                base_model_names=base_model_names,
                core_kwargs=core_kwargs_level,
                aux_kwargs=aux_kwargs_level,
                name_suffix=name_suffix,
                infer_limit=infer_limit,
                infer_limit_batch_size=infer_limit_batch_size,
                full_weighted_ensemble=full_weighted_ensemble,
                additional_full_weighted_ensemble=additional_full_weighted_ensemble,
            )
            model_names_fit += base_model_names + aux_models
        if (self.model_best is None or infer_limit is not None) and len(model_names_fit) != 0:
            self.model_best = self.get_model_best(infer_limit=infer_limit, infer_limit_as_child=True)
        self._callbacks_conclude()
        self._fit_cleanup()
        self.save()
        return model_names_fit

    def _fit_setup(
        self, time_limit: float | None = None, callbacks: list[AbstractCallback | list | tuple] | None = None
    ):
        """
        Prepare the trainer state at the start of / prior to a fit call.
        Should be paired with a `self._fit_cleanup()` at the conclusion of the fit call.
        """
        self._time_train_start = time.time()
        self._time_train_start_last = self._time_train_start
        self._time_limit = time_limit
        self.reset_callbacks()
        callbacks_new = []
        if callbacks is not None:
            assert isinstance(callbacks, list), f"`callbacks` must be a list. Found invalid type: `{type(callbacks)}`."
            for callback in callbacks:
                if isinstance(callback, (list, tuple)):
                    assert len(callback) == 2, (
                        f"Callback must either be an initialized object or a tuple/list of length 2, found: {callback}"
                    )
                    callback_cls = callback[0]
                    if isinstance(callback_cls, str):
                        from autogluon.core.callbacks._early_stopping_callback import EarlyStoppingCallback
                        from autogluon.core.callbacks._early_stopping_count_callback import EarlyStoppingCountCallback
                        from autogluon.core.callbacks._early_stopping_ensemble_callback import (
                            EarlyStoppingEnsembleCallback,
                        )

                        _callback_cls_lst = [
                            EarlyStoppingCallback,
                            EarlyStoppingCountCallback,
                            EarlyStoppingEnsembleCallback,
                        ]

                        _callback_cls_name_map = {c.__name__: c for c in _callback_cls_lst}

                        assert callback_cls in _callback_cls_name_map.keys(), (
                            f"Unknown callback class: {callback_cls}. "
                            f"Valid classes: {list(_callback_cls_name_map.keys())}"
                        )
                        callback_cls = _callback_cls_name_map[callback_cls]

                    callback_kwargs = callback[1]
                    assert isinstance(callback_kwargs, dict), (
                        f"Callback kwargs must be a dictionary, found: {callback_kwargs}"
                    )
                    callback = callback_cls(**callback_kwargs)
                else:
                    assert isinstance(callback, AbstractCallback), (
                        f"Elements in `callbacks` must be of type AbstractCallback. Found invalid type: `{type(callback)}`."
                    )
                callbacks_new.append(callback)
        else:
            callbacks_new = []
        self.callbacks = callbacks_new

    def _fit_cleanup(self):
        """
        Cleanup the trainer state after fit call completes.
        This ensures that future fit calls are not corrupted by prior fit calls.
        Should be paired with an earlier `self._fit_setup()` call.
        """
        self._time_limit = None
        self._time_train_start = None
        self.reset_callbacks()

    def _callbacks_setup(self, **kwargs):
        for callback in self.callbacks:
            callback.before_trainer_fit(trainer=self, **kwargs)

    def _callbacks_conclude(self):
        for callback in self.callbacks:
            callback.after_trainer_fit(trainer=self)

    def reset_callbacks(self):
        """Deletes callback objects and resets `self._callback_early_stop` to False."""
        self.callbacks = []
        self._callback_early_stop = False

    # TODO: Consider better greedy approximation method such as via fitting a weighted ensemble to evaluate the value of a subset.
    def _filter_base_models_via_infer_limit(
        self,
        base_model_names: list[str],
        infer_limit: float | None,
        infer_limit_modifier: float = 1.0,
        as_child: bool = True,
        verbose: bool = True,
    ) -> list[str]:
        """
        Returns a subset of base_model_names whose combined prediction time for 1 row of data does not exceed infer_limit seconds.
        With the goal of selecting the best valid subset that is most valuable to stack ensembles who use them as base models,
        this is a variant of the constrained knapsack problem and is NP-Hard and infeasible to exactly solve even with fewer than 10 models.
        For practical purposes, this method applies a greedy approximation approach to selecting the subset
        by simply removing models in reverse order of validation score until the remaining subset is valid.

        Parameters
        ----------
        base_model_names: list[str]
            list of model names. These models must already be added to the trainer.
        infer_limit: float, optional
            Inference limit in seconds for 1 row of data. This is compared against values pre-computed during fit for the models.
        infer_limit_modifier: float, default = 1.0
            Modifier to multiply infer_limit by.
            Set to <1.0 to provide headroom for stack models who take the returned subset as base models
            so that the stack models are less likely to exceed infer_limit.
        as_child: bool, default = True
            If True, use the inference time of only 1 child model for bags instead of the overall inference time of the bag.
            This is useful if the intent is to refit the models, as this will best estimate the inference time of the refit model.
        verbose: bool, default = True
            Whether to log the models that are removed.

        Returns
        -------
        Returns valid subset of models that satisfy constraints.
        """
        if infer_limit is None or not base_model_names:
            return base_model_names

        base_model_names = base_model_names.copy()
        num_models_og = len(base_model_names)
        infer_limit_threshold = infer_limit * infer_limit_modifier  # Add headroom

        if as_child:
            attribute = "predict_1_child_time"
        else:
            attribute = "predict_1_time"

        predict_1_time_full_set = self.get_model_attribute_full(model=base_model_names, attribute=attribute)

        messages_to_log = []

        base_model_names_copy = base_model_names.copy()
        # Prune models that by themselves have larger inference latency than the infer_limit, as they can never be valid
        for base_model_name in base_model_names_copy:
            predict_1_time_full = self.get_model_attribute_full(model=base_model_name, attribute=attribute)
            if predict_1_time_full >= infer_limit_threshold:
                predict_1_time_full_set_old = predict_1_time_full_set
                base_model_names.remove(base_model_name)
                predict_1_time_full_set = self.get_model_attribute_full(model=base_model_names, attribute=attribute)
                if verbose:
                    predict_1_time_full_set_log, time_unit = convert_time_in_s_to_log_friendly(
                        time_in_sec=predict_1_time_full_set
                    )
                    predict_1_time_full_set_old_log, time_unit_old = convert_time_in_s_to_log_friendly(
                        time_in_sec=predict_1_time_full_set_old
                    )
                    messages_to_log.append(
                        f"\t{round(predict_1_time_full_set_old_log, 3)}{time_unit_old}\t-> {round(predict_1_time_full_set_log, 3)}{time_unit}\t({base_model_name})"
                    )

        score_val_dict = self.get_models_attribute_dict(attribute="val_score", models=base_model_names)

        # Penalize score by 1e-5 * predict_1_time
        # This ensures that if two models have the same score, the one with faster inference time is preferred
        # because we sort by ascending score, removing the lowest scores first.
        # A higher penalty makes the model more likely to be removed.
        # A slower model gets a larger penalty subtracted from its score, making its "effective score" lower,
        # thus making it fail the check earlier and be removed.
        score_val_dict_penalized = {}
        for m, score in score_val_dict.items():
            predict_time = self.get_model_attribute_full(model=m, attribute=attribute)
            # Ensure predict_time is not None (should not happen if attribute is correctly fetched)
            if predict_time is None:
                predict_time = 0

            # Penalty factor: 1e-4 reduces score by 0.0001 per second of inference time.
            # Example:
            # Model A: Score 0.8, Time 0.1s -> 0.8 - 0.00001 = 0.79999
            # Model B: Score 0.8, Time 10s  -> 0.8 - 0.001   = 0.799
            # Model B is removed first.
            penalty = 1e-4 * predict_time
            score_val_dict_penalized[m] = score - penalty

        sorted_scores = sorted(score_val_dict_penalized.items(), key=lambda x: x[1])
        i = 0
        # Prune models by ascending validation score until the remaining subset's combined inference latency satisfies infer_limit
        while base_model_names and (predict_1_time_full_set >= infer_limit_threshold):
            # TODO: Incorporate score vs inference speed tradeoff in a smarter way
            base_model_to_remove = sorted_scores[i][0]
            predict_1_time_full_set_old = predict_1_time_full_set
            base_model_names.remove(base_model_to_remove)
            i += 1
            predict_1_time_full_set = self.get_model_attribute_full(model=base_model_names, attribute=attribute)
            if verbose:
                predict_1_time_full_set_log, time_unit = convert_time_in_s_to_log_friendly(
                    time_in_sec=predict_1_time_full_set
                )
                predict_1_time_full_set_old_log, time_unit_old = convert_time_in_s_to_log_friendly(
                    time_in_sec=predict_1_time_full_set_old
                )
                messages_to_log.append(
                    f"\t{round(predict_1_time_full_set_old_log, 3)}{time_unit_old}\t-> {round(predict_1_time_full_set_log, 3)}{time_unit}\t({base_model_to_remove})"
                )

        if messages_to_log:
            infer_limit_threshold_log, time_unit_threshold = convert_time_in_s_to_log_friendly(
                time_in_sec=infer_limit_threshold
            )
            logger.log(
                20,
                f"Removing {len(messages_to_log)}/{num_models_og} base models to satisfy inference constraint "
                f"(constraint={round(infer_limit_threshold_log, 3)}{time_unit_threshold}) ...",
            )
            for msg in messages_to_log:
                logger.log(20, msg)

        return base_model_names

    def stack_new_level(
        self,
        X,
        y,
        models: list[AbstractModel] | dict,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        X_unlabeled=None,
        level=1,
        base_model_names: list[str] | None = None,
        core_kwargs: dict | None = None,
        aux_kwargs: dict | None = None,
        name_suffix: str | None = None,
        infer_limit=None,
        infer_limit_batch_size=None,
        full_weighted_ensemble: bool = False,
        additional_full_weighted_ensemble: bool = False,
    ) -> tuple[list[str], list[str]]:
        """
        Similar to calling self.stack_new_level_core, except auxiliary models will also be trained via a call to self.stack_new_level_aux, with the models trained from self.stack_new_level_core used as base models.
        """
        if base_model_names is None:
            base_model_names = []
        core_kwargs = {} if core_kwargs is None else core_kwargs.copy()
        aux_kwargs = {} if aux_kwargs is None else aux_kwargs.copy()
        if level < 1:
            raise AssertionError(f"Stack level must be >= 1, but level={level}.")
        if base_model_names and level == 1:
            raise AssertionError(
                f"Stack level 1 models cannot have base models, but base_model_names={base_model_names}."
            )
        if name_suffix:
            core_kwargs["name_suffix"] = core_kwargs.get("name_suffix", "") + name_suffix
            aux_kwargs["name_suffix"] = aux_kwargs.get("name_suffix", "") + name_suffix
        core_models = self.stack_new_level_core(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_unlabeled=X_unlabeled,
            models=models,
            level=level,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            base_model_names=base_model_names,
            **core_kwargs,
        )

        aux_models = []
        if full_weighted_ensemble:
            full_aux_kwargs = aux_kwargs.copy()
            if additional_full_weighted_ensemble:
                full_aux_kwargs["name_extra"] = "_ALL"
            all_base_model_names = self.get_model_names(
                stack_name="core"
            )  # Fit weighted ensemble on all previously fitted core models
            aux_models += self._stack_new_level_aux(
                X_val, y_val, X, y, all_base_model_names, level, infer_limit, infer_limit_batch_size, **full_aux_kwargs
            )

        if (not full_weighted_ensemble) or additional_full_weighted_ensemble:
            aux_models += self._stack_new_level_aux(
                X_val, y_val, X, y, core_models, level, infer_limit, infer_limit_batch_size, **aux_kwargs
            )

        return core_models, aux_models

    def stack_new_level_core(
        self,
        X,
        y,
        models: list[AbstractModel] | dict,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        X_unlabeled=None,
        level=1,
        base_model_names: list[str] | None = None,
        fit_strategy: Literal["sequential", "parallel"] = "sequential",
        stack_name="core",
        ag_args=None,
        ag_args_fit=None,
        ag_args_ensemble=None,
        included_model_types=None,
        excluded_model_types=None,
        ensemble_type=StackerEnsembleModel,
        name_suffix: str | None = None,
        get_models_func=None,
        refit_full=False,
        infer_limit=None,
        infer_limit_batch_size=None,
        **kwargs,
    ) -> list[str]:
        """
        Trains all models using the data provided.
        If level > 1, then the models will use base model predictions as additional features.
            The base models used can be specified via base_model_names.
        If self.bagged_mode, then models will be trained as StackerEnsembleModels.
        The data provided in this method should not contain stack features, as they will be automatically generated if necessary.
        """
        if self._callback_early_stop:
            return []
        if get_models_func is None:
            get_models_func = self.construct_model_templates
        if base_model_names is None:
            base_model_names = []
        if not self.bagged_mode and level != 1:
            raise ValueError("Stack Ensembling is not valid for non-bagged mode.")

        base_model_names = self._filter_base_models_via_infer_limit(
            base_model_names=base_model_names,
            infer_limit=infer_limit,
            infer_limit_modifier=0.8,
        )
        if ag_args_fit is None:
            ag_args_fit = {}
        ag_args_fit = ag_args_fit.copy()
        if infer_limit_batch_size is not None:
            ag_args_fit["predict_1_batch_size"] = infer_limit_batch_size

        if isinstance(models, dict):
            get_models_kwargs = dict(
                level=level,
                name_suffix=name_suffix,
                ag_args=ag_args,
                ag_args_fit=ag_args_fit,
                included_model_types=included_model_types,
                excluded_model_types=excluded_model_types,
            )

            if self.bagged_mode:
                if level == 1:
                    (base_model_names, base_model_paths, base_model_types) = (None, None, None)
                elif level > 1:
                    base_model_names, base_model_paths, base_model_types = self._get_models_load_info(
                        model_names=base_model_names
                    )
                    if len(base_model_names) == 0:  # type: ignore
                        logger.log(20, f"No base models to train on, skipping stack level {level}...")
                        return []
                else:
                    raise AssertionError(f"Stack level cannot be less than 1! level = {level}")

                ensemble_kwargs = {
                    "base_model_names": base_model_names,
                    "base_model_paths_dict": base_model_paths,
                    "base_model_types_dict": base_model_types,
                    "base_model_types_inner_dict": self.get_models_attribute_dict(
                        attribute="type_inner", models=base_model_names
                    ),
                    "base_model_performances_dict": self.get_models_attribute_dict(
                        attribute="val_score", models=base_model_names
                    ),
                    "random_state": level + self.random_state,
                }
                get_models_kwargs.update(
                    dict(
                        ag_args_ensemble=ag_args_ensemble,
                        ensemble_type=ensemble_type,
                        ensemble_kwargs=ensemble_kwargs,
                    )
                )
            models, model_args_fit = get_models_func(hyperparameters=models, **get_models_kwargs)
            if model_args_fit:
                hyperparameter_tune_kwargs = {
                    model_name: model_args_fit[model_name]["hyperparameter_tune_kwargs"]
                    for model_name in model_args_fit
                    if "hyperparameter_tune_kwargs" in model_args_fit[model_name]
                }
                kwargs["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs

        logger.log(
            10 if ((not refit_full) and DistributedContext.is_distributed_mode()) else 20,
            f'Fitting {len(models)} L{level} models, fit_strategy="{fit_strategy}" ...',
        )

        X_init = self.get_inputs_to_stacker(X, base_models=base_model_names, fit=True)
        feature_metadata = self.get_feature_metadata(use_orig_features=True, base_models=base_model_names)
        if X_val is not None:
            X_val = self.get_inputs_to_stacker(X_val, base_models=base_model_names, fit=False, use_val_cache=True)
        if X_test is not None:
            X_test = self.get_inputs_to_stacker(X_test, base_models=base_model_names, fit=False, use_val_cache=False)
        compute_score = not refit_full
        if refit_full and X_val is not None:
            X_init = pd.concat([X_init, X_val])
            y = pd.concat([y, y_val])
            X_val = None
            y_val = None
        if X_unlabeled is not None:
            X_unlabeled = self.get_inputs_to_stacker(X_unlabeled, base_models=base_model_names, fit=False)

        fit_kwargs = dict(
            num_classes=self.num_classes,
            feature_metadata=feature_metadata,
        )

        # FIXME: TODO: v0.1 X_unlabeled isn't cached so it won't be available during refit_full or fit_extra.
        return self._train_multi(
            X=X_init,
            y=y,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_unlabeled=X_unlabeled,
            models=models,
            level=level,
            stack_name=stack_name,
            compute_score=compute_score,
            fit_kwargs=fit_kwargs,
            fit_strategy=fit_strategy,
            **kwargs,
        )

    def _stack_new_level_aux(
        self, X_val, y_val, X, y, core_models, level, infer_limit, infer_limit_batch_size, **kwargs
    ):
        if X_val is None:
            aux_models = self.stack_new_level_aux(
                X=X,
                y=y,
                base_model_names=core_models,
                level=level + 1,
                infer_limit=infer_limit,
                infer_limit_batch_size=infer_limit_batch_size,
                **kwargs,
            )
        else:
            aux_models = self.stack_new_level_aux(
                X=X_val,
                y=y_val,
                fit=False,
                base_model_names=core_models,
                level=level + 1,
                infer_limit=infer_limit,
                infer_limit_batch_size=infer_limit_batch_size,
                **kwargs,
            )
        return aux_models

    # TODO: Consider making level be auto-determined based off of max(base_model_levels)+1
    # TODO: Remove name_suffix, hacked in
    # TODO: X can be optional because it isn't needed if fit=True
    def stack_new_level_aux(
        self,
        X,
        y,
        base_model_names: list[str],
        level: int | str = "auto",
        fit=True,
        stack_name="aux1",
        time_limit=None,
        name_suffix: str | None = None,
        get_models_func=None,
        check_if_best=True,
        infer_limit=None,
        infer_limit_batch_size=None,
        use_val_cache=True,
        fit_weighted_ensemble: bool = True,
        name_extra: str | None = None,
        total_resources: dict | None = None,
    ) -> list[str]:
        """
        Trains auxiliary models (currently a single weighted ensemble) using the provided base models.
        Level must be greater than the level of any of the base models.
        Auxiliary models never use the original features and only train with the predictions of other models as features.
        """
        if self._callback_early_stop:
            return []
        if fit_weighted_ensemble is False:
            # Skip fitting of aux models
            return []

        base_model_names = self._filter_base_models_via_infer_limit(
            base_model_names=base_model_names, infer_limit=infer_limit, infer_limit_modifier=0.95
        )

        if len(base_model_names) == 0:
            logger.log(20, f"No base models to train on, skipping auxiliary stack level {level}...")
            return []

        if isinstance(level, str):
            assert level == "auto", f"level must be 'auto' if str, found: {level}"
            levels_dict = self.get_models_attribute_dict(attribute="level", models=base_model_names)
            base_model_level_max = None
            for k, v in levels_dict.items():
                if base_model_level_max is None or v > base_model_level_max:
                    base_model_level_max = v
            level = base_model_level_max + 1

        if infer_limit_batch_size is not None:
            ag_args_fit = dict()
            ag_args_fit["predict_1_batch_size"] = infer_limit_batch_size
        else:
            ag_args_fit = None
        X_stack_preds = self.get_inputs_to_stacker(
            X, base_models=base_model_names, fit=fit, use_orig_features=False, use_val_cache=use_val_cache
        )
        if self.weight_evaluation:
            X, w = extract_column(
                X, self.sample_weight
            )  # TODO: consider redesign with w as separate arg instead of bundled inside X
            if w is not None:
                X_stack_preds[self.sample_weight] = w.values / w.mean()
        child_hyperparameters = None
        if name_extra is not None:
            child_hyperparameters = {"ag_args": {"name_suffix": name_extra}}
        return self.generate_weighted_ensemble(
            X=X_stack_preds,
            y=y,
            level=level,
            base_model_names=base_model_names,
            k_fold=1,
            n_repeats=1,
            ag_args_fit=ag_args_fit,
            stack_name=stack_name,
            time_limit=time_limit,
            name_suffix=name_suffix,
            get_models_func=get_models_func,
            check_if_best=check_if_best,
            child_hyperparameters=child_hyperparameters,
            total_resources=total_resources,
        )

    def predict(self, X: pd.DataFrame, model: str | None = None) -> np.ndarray:
        if model is None:
            model = self._get_best()
        return self._predict_model(X=X, model=model)

    def predict_proba(self, X: pd.DataFrame, model: str | None = None) -> np.ndarray:
        if model is None:
            model = self._get_best()
        return self._predict_proba_model(X=X, model=model)

    def _get_best(self) -> str:
        if self.model_best is not None:
            return self.model_best
        else:
            return self.get_model_best()

    # Note: model_pred_proba_dict is mutated in this function to minimize memory usage
    def get_inputs_to_model(
        self,
        model: str | AbstractModel,
        X: pd.DataFrame,
        model_pred_proba_dict: dict[str, np.ndarray] | None = None,
        fit: bool = False,
        preprocess_nonadaptive: bool = False,
    ) -> pd.DataFrame:
        """
        For output X:
            If preprocess_nonadaptive=False, call model.predict(X)
            If preprocess_nonadaptive=True, call model.predict(X, preprocess_nonadaptive=False)
        """
        if isinstance(model, str):
            # TODO: Remove unnecessary load when no stacking
            model = self.load_model(model)
        model_level = self.get_model_level(model.name)
        if model_level > 1 and isinstance(model, StackerEnsembleModel):
            if fit:
                model_pred_proba_dict = None
            else:
                model_set = self.get_minimum_model_set(model)
                model_set = [
                    m for m in model_set if m != model.name
                ]  # TODO: Can probably be faster, get this result from graph
                model_pred_proba_dict = self.get_model_pred_proba_dict(
                    X=X, models=model_set, model_pred_proba_dict=model_pred_proba_dict
                )
            X = model.preprocess(
                X=X,
                preprocess_nonadaptive=preprocess_nonadaptive,
                fit=fit,
                model_pred_proba_dict=model_pred_proba_dict,
            )
        elif preprocess_nonadaptive:
            X = model.preprocess(X=X, preprocess_stateful=False)
        return X

    def score(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model: str | None = None,
        metric: Scorer | None = None,
        weights: np.ndarray | None = None,
        as_error: bool = False,
    ) -> float:
        if metric is None:
            metric = self.eval_metric
        if metric.needs_pred or metric.needs_quantile:
            y_pred = self.predict(X=X, model=model)
            y_pred_proba = None
        else:
            y_pred = None
            y_pred_proba = self.predict_proba(X=X, model=model)
        return compute_metric(
            y=y,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            metric=metric,
            weights=weights,
            weight_evaluation=self.weight_evaluation,
            as_error=as_error,
            quantile_levels=self.quantile_levels,
        )

    def score_with_y_pred_proba(
        self,
        y: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: Scorer | None = None,
        weights: np.ndarray | None = None,
        as_error: bool = False,
    ) -> float:
        if metric is None:
            metric = self.eval_metric
        if metric.needs_pred or metric.needs_quantile:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            y_pred_proba = None
        else:
            y_pred = None
        return compute_metric(
            y=y,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            metric=metric,
            weights=weights,
            weight_evaluation=self.weight_evaluation,
            as_error=as_error,
            quantile_levels=self.quantile_levels,
        )

    def score_with_y_pred(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray | None = None,
        metric: Scorer | None = None,
        as_error: bool = False,
    ) -> float:
        if metric is None:
            metric = self.eval_metric
        return compute_metric(
            y=y,
            y_pred=y_pred,
            y_pred_proba=None,
            metric=metric,
            weights=weights,
            weight_evaluation=self.weight_evaluation,
            as_error=as_error,
            quantile_levels=self.quantile_levels,
        )

    # TODO: Slow if large ensemble with many models, could cache output result to speed up during inference
    def _construct_model_pred_order(self, models: list[str]) -> list[str]:
        """
        Constructs a list of model names in order of inference calls required to infer on all the models.

        Parameters
        ----------
        models : list[str]
            The list of models to construct the prediction order from.
            If a model has dependencies, the dependency models will be put earlier in the output list.
            Models explicitly mentioned in the `models` input will be placed as early as possible in the output list.
            Models earlier in `models` will attempt to be placed earlier in the output list than those later in `models`.
                It is recommended that earlier elements do not have dependency models that are listed later in `models`.

        Returns
        -------
        Returns list of models in inference call order, including dependency models of those specified in the input.
        """
        model_set = set()
        model_order = []
        for model in models:
            if model in model_set:
                continue
            min_models_set = set(self.get_minimum_model_set(model))
            models_to_load = list(min_models_set.difference(model_set))
            subgraph = nx.subgraph(self.model_graph, models_to_load)
            model_pred_order = list(nx.lexicographical_topological_sort(subgraph))
            model_order += [m for m in model_pred_order if m not in model_set]
            model_set = set(model_order)
        return model_order

    def _construct_model_pred_order_with_pred_dict(
        self, models: list[str], models_to_ignore: list[str] | None = None
    ) -> list[str]:
        """
        Constructs a list of model names in order of inference calls required to infer on all the models.
        Unlike `_construct_model_pred_order`, this method's output is in undefined order when multiple models are valid to infer at the same time.

        Parameters
        ----------
        models : list[str]
            The list of models to construct the prediction order from.
            If a model has dependencies, the dependency models will be put earlier in the output list.
        models_to_ignore : list[str], optional
            A list of models that have already been computed and can be ignored.
            Models in this list and their dependencies (if not depended on by other models in `models`) will be pruned from the final output.

        Returns
        -------
        Returns list of models in inference call order, including dependency models of those specified in the input.
        """
        model_set = set()
        for model in models:
            if model in model_set:
                continue
            min_model_set = set(self.get_minimum_model_set(model))
            model_set = model_set.union(min_model_set)
        if models_to_ignore is not None:
            model_set = model_set.difference(set(models_to_ignore))
        models_to_load = list(model_set)
        subgraph = nx.DiGraph(nx.subgraph(self.model_graph, models_to_load))  # Wrap subgraph in DiGraph to unfreeze it
        # For model in models_to_ignore, remove model node from graph and all ancestors that have no remaining descendants and are not in `models`
        models_to_ignore = [
            model for model in models_to_load if (model not in models) and (not list(subgraph.successors(model)))
        ]
        while models_to_ignore:
            model = models_to_ignore[0]
            predecessors = list(subgraph.predecessors(model))
            subgraph.remove_node(model)
            models_to_ignore = models_to_ignore[1:]
            for predecessor in predecessors:
                if (
                    (predecessor not in models)
                    and (not list(subgraph.successors(predecessor)))
                    and (predecessor not in models_to_ignore)
                ):
                    models_to_ignore.append(predecessor)

        # Get model prediction order
        return list(nx.lexicographical_topological_sort(subgraph))

    def get_models_attribute_dict(self, attribute: str, models: list | None = None) -> dict[str, Any]:
        """Returns dictionary of model name -> attribute value for the provided attribute."""
        models_attribute_dict = nx.get_node_attributes(self.model_graph, attribute)
        if models is not None:
            model_names = []
            for model in models:
                if not isinstance(model, str):
                    model = model.name
                model_names.append(model)
            if attribute == "path":
                models_attribute_dict = {
                    key: os.path.join(*val) for key, val in models_attribute_dict.items() if key in model_names
                }
            else:
                models_attribute_dict = {key: val for key, val in models_attribute_dict.items() if key in model_names}
        return models_attribute_dict

    # TODO: Consider adding persist to disk functionality for pred_proba dictionary to lessen memory burden on large multiclass problems.
    #  For datasets with 100+ classes, this function could potentially run the system OOM due to each pred_proba numpy array taking significant amounts of space.
    #  This issue already existed in the previous level-based version but only had the minimum required predictions in memory at a time, whereas this has all model predictions in memory.
    # TODO: Add memory optimal topological ordering -> Minimize amount of pred_probas in memory at a time, delete pred probas that are no longer required
    def get_model_pred_proba_dict(
        self,
        X: pd.DataFrame,
        models: list[str],
        model_pred_proba_dict: dict | None = None,
        model_pred_time_dict: dict | None = None,
        record_pred_time: bool = False,
        use_val_cache: bool = False,
    ):
        """
        Optimally computes pred_probas (or predictions if regression) for each model in `models`.
        Will compute each necessary model only once and store predictions in a `model_pred_proba_dict` dictionary.
        Note: Mutates model_pred_proba_dict and model_pred_time_dict input if present to minimize memory usage.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to predict on.
        models : list[str]
            The list of models to predict with.
            Note that if models have dependency models, their dependencies will also be predicted with and included in the output.
        model_pred_proba_dict : dict, optional
            A dict of predict_probas that could have been computed by a prior call to `get_model_pred_proba_dict` to avoid redundant computations.
            Models already present in model_pred_proba_dict will not be predicted on.
            get_model_pred_proba_dict(X, models=['A', 'B', 'C']) is equivalent to
            get_model_pred_proba_dict(X, models=['C'], model_pred_proba_dict=get_model_pred_proba_dict(X, models=['A', 'B']))
            Note: Mutated in-place to minimize memory usage
        model_pred_time_dict : dict, optional
            If `record_pred_time==True`, this is a dict of model name to marginal time taken in seconds for the prediction of X.
            Must be specified alongside `model_pred_proba_dict` if `record_pred_time=True` and `model_pred_proba_dict != None`.
            Ignored if `record_pred_time=False`.
            Note: Mutated in-place to minimize memory usage
        record_pred_time : bool, default = False
            Whether to store marginal inference times of each model as an extra output `model_pred_time_dict`.
        use_val_cache : bool, default = False
            Whether to fetch cached val prediction probabilities for models instead of predicting on the data.
            Only set to True if X is equal to the validation data and you want to skip live predictions.

        Returns
        -------
        If `record_pred_time==True`, outputs tuple of dicts (model_pred_proba_dict, model_pred_time_dict), else output only model_pred_proba_dict
        """
        if model_pred_proba_dict is None:
            model_pred_proba_dict = {}
        if model_pred_time_dict is None:
            model_pred_time_dict = {}

        if use_val_cache:
            _, model_pred_proba_dict = self._update_pred_proba_dict_with_val_cache(
                model_set=set(models), model_pred_proba_dict=model_pred_proba_dict
            )
        if not model_pred_proba_dict:
            model_pred_order = self._construct_model_pred_order(models)
        else:
            model_pred_order = self._construct_model_pred_order_with_pred_dict(
                models, models_to_ignore=list(model_pred_proba_dict.keys())
            )
        if use_val_cache:
            model_set, model_pred_proba_dict = self._update_pred_proba_dict_with_val_cache(
                model_set=set(model_pred_order), model_pred_proba_dict=model_pred_proba_dict
            )
            model_pred_order = [model for model in model_pred_order if model in model_set]

        # Compute model predictions in topological order
        for model_name in model_pred_order:
            if record_pred_time:
                time_start = time.time()

            model = self.load_model(model_name=model_name)
            if isinstance(model, StackerEnsembleModel):
                preprocess_kwargs = dict(infer=False, model_pred_proba_dict=model_pred_proba_dict)
                model_pred_proba_dict[model_name] = model.predict_proba(X, **preprocess_kwargs)
            else:
                model_pred_proba_dict[model_name] = model.predict_proba(X)

            if record_pred_time:
                time_end = time.time()
                model_pred_time_dict[model_name] = time_end - time_start

        if record_pred_time:
            return model_pred_proba_dict, model_pred_time_dict
        else:
            return model_pred_proba_dict

    def get_model_oof_dict(self, models: list[str]) -> dict:
        """
        Returns a dictionary of out-of-fold prediction probabilities, keyed by model name
        """
        return {model: self.get_model_oof(model) for model in models}

    def get_model_pred_dict(self, X: pd.DataFrame, models: list[str], record_pred_time: bool = False, **kwargs):
        """
        Optimally computes predictions for each model in `models`.
        Will compute each necessary model only once and store predictions in a `model_pred_dict` dictionary.
        Note: Mutates model_pred_proba_dict and model_pred_time_dict input if present to minimize memory usage.

        Acts as a wrapper to `self.get_model_pred_proba_dict`, converting the output to predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to predict on.
        models : list[str]
            The list of models to predict with.
            Note that if models have dependency models, their dependencies will also be predicted with and included in the output.
        record_pred_time : bool, default = False
            Whether to store marginal inference times of each model as an extra output `model_pred_time_dict`.
        **kwargs : dict, optional
            Refer to `self.get_model_pred_proba_dict` for documentation of remaining arguments.
            This method shares identical arguments.

        Returns
        -------
        If `record_pred_time==True`, outputs tuple of dicts (model_pred_dict, model_pred_time_dict), else output only model_pred_dict
        """
        model_pred_proba_dict = self.get_model_pred_proba_dict(
            X=X, models=models, record_pred_time=record_pred_time, **kwargs
        )
        if record_pred_time:
            model_pred_proba_dict, model_pred_time_dict = model_pred_proba_dict
        else:
            model_pred_time_dict = None

        model_pred_dict = {}
        for m in model_pred_proba_dict:
            # Convert pred_proba to pred
            model_pred_dict[m] = get_pred_from_proba(
                y_pred_proba=model_pred_proba_dict[m], problem_type=self.problem_type
            )

        if record_pred_time:
            return model_pred_dict, model_pred_time_dict
        else:
            return model_pred_dict

    def get_model_oof(self, model: str, use_refit_parent: bool = False) -> np.ndarray:
        """
        Gets the out of fold prediction probabilities for a bagged ensemble model

        Parameters
        ----------
        model : str
            Name of the model to get OOF.
        use_refit_parent: bool = False
            If True and the model is a refit model, will instead return the parent model's OOF.
            If False and the model is a refit model, an exception will be raised.

        Returns
        -------
        np.ndarray
            model OOF prediction probabilities (if classification) or predictions (if regression)
        """
        if use_refit_parent and self.get_model_attribute(model=model, attribute="refit_full", default=False):
            model = self.get_model_attribute(model=model, attribute="refit_full_parent")
        model_type = self.get_model_attribute(model=model, attribute="type")
        if issubclass(model_type, BaggedEnsembleModel):
            model_path = self.get_model_attribute(model=model, attribute="path")
            return model_type.load_oof(path=os.path.join(self.path, model_path))
        else:
            raise AssertionError(f"Model {model} must be a BaggedEnsembleModel to return oof_pred_proba")

    def get_model_learning_curves(self, model: str) -> dict:
        model_type = self.get_model_attribute(model=model, attribute="type")
        model_path = self.get_model_attribute(model=model, attribute="path")
        return model_type.load_learning_curves(path=os.path.join(self.path, model_path))

    def _update_pred_proba_dict_with_val_cache(self, model_set: set, model_pred_proba_dict):
        """For each model in model_set, check if y_pred_proba_val is cached to disk. If so, load and add it to model_pred_proba_dict"""
        for model in model_set:
            y_pred_proba = self.get_model_attribute(model, attribute="cached_y_pred_proba_val", default=None)
            if isinstance(y_pred_proba, bool):
                if y_pred_proba:
                    try:
                        y_pred_proba = self._load_model_y_pred_proba_val(model)
                    except FileNotFoundError:
                        y_pred_proba = None
                else:
                    y_pred_proba = None
            if y_pred_proba is not None:
                model_pred_proba_dict[model] = y_pred_proba
        model_set = model_set.difference(set(model_pred_proba_dict.keys()))
        return model_set, model_pred_proba_dict

    def get_inputs_to_stacker(
        self,
        X: pd.DataFrame,
        *,
        model: str | None = None,
        base_models: list[str] | None = None,
        model_pred_proba_dict: dict | None = None,
        fit: bool = False,
        use_orig_features: bool = True,
        use_val_cache: bool = False,
    ) -> pd.DataFrame:
        """
        Returns the valid X input for a stacker model with base models equal to `base_models`.
        Pairs with `feature_metadata = self.get_feature_metadata(...)`. The contents of the returned `X` should reflect `feature_metadata`.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to augment.
        model : str, default = None
            The model to derive `base_models` from.
            Cannot be specified alongside `base_models`.
        base_models : list[str], default = None
            The list of base models to augment X with.
            Base models will add their prediction probabilities as extra features to X.
            Cannot be specified alongside `model`.
        model_pred_proba_dict : dict, optional
            A dict of predict_probas that could have been computed by a prior call to `get_model_pred_proba_dict` to avoid redundant computations.
            Models already present in model_pred_proba_dict will not be predicted on.
            Note: Mutated in-place to minimize memory usage
        fit : bool, default = False
            If True, X represents the training data and the models will return their out-of-fold prediction probabilities.
            If False, X represents validation or test data and the models will predict directly on X to generate their prediction probabilities.
        use_orig_features : bool, default = True
            If True, the output DataFrame will include X's original features in addition to the new stack features.
            If False, the output DataFrame will only contain the new stack features.
        use_val_cache : bool, default = False
            Whether to fetch cached val prediction probabilities for models instead of predicting on the data.
            Only set to True if X is equal to the validation data and you want to skip live predictions.

        Returns
        -------
        X : DataFrame, an updated DataFrame with the additional stack features from `base_models`.
        """
        if model is not None and base_models is not None:
            raise AssertionError("Only one of `model`, `base_models` is allowed to be set.")

        if model is not None and base_models is None:
            base_models = self.get_base_model_names(model)
        if not base_models:
            return X
        if fit:
            model_pred_proba_dict = self.get_model_oof_dict(models=base_models)
        else:
            model_pred_proba_dict = self.get_model_pred_proba_dict(
                X=X, models=base_models, model_pred_proba_dict=model_pred_proba_dict, use_val_cache=use_val_cache
            )
        pred_proba_list = [model_pred_proba_dict[model] for model in base_models]
        stack_column_names, _ = self._get_stack_column_names(models=base_models)
        X_stacker = convert_pred_probas_to_df(
            pred_proba_list=pred_proba_list, problem_type=self.problem_type, columns=stack_column_names, index=X.index
        )
        if use_orig_features:
            X = pd.concat([X_stacker, X], axis=1)
        else:
            X = X_stacker
        return X

    def get_feature_metadata(
        self, use_orig_features: bool = True, model: str | None = None, base_models: list[str] | None = None
    ) -> FeatureMetadata:
        """
        Returns the FeatureMetadata input to a `model.fit` call.
        Pairs with `X = self.get_inputs_to_stacker(...)`. The returned FeatureMetadata should reflect the contents of `X`.

        Parameters
        ----------
        use_orig_features : bool, default = True
            If True, will include the original features in the FeatureMetadata.
            If False, will only include the stack features in the FeatureMetadata.
        model : str, default = None
            If specified, it must be an already existing model.
            `base_models` will be set to the base models of `model`.
        base_models : list[str], default = None
            If specified, will add the stack features of the `base_models` to FeatureMetadata.

        Returns
        -------
        FeatureMetadata
            The FeatureMetadata that should be passed into a `model.fit` call.
        """
        if model is not None and base_models is not None:
            raise AssertionError("Only one of `model`, `base_models` is allowed to be set.")
        if model is not None and base_models is None:
            base_models = self.get_base_model_names(model)

        feature_metadata = None
        if use_orig_features:
            feature_metadata = self.feature_metadata
        if base_models:
            stack_column_names, _ = self._get_stack_column_names(models=base_models)
            stacker_type_map_raw = {column: R_FLOAT for column in stack_column_names}
            stacker_type_group_map_special = {S_STACK: stack_column_names}
            stacker_feature_metadata = FeatureMetadata(
                type_map_raw=stacker_type_map_raw, type_group_map_special=stacker_type_group_map_special
            )
            if feature_metadata is not None:
                feature_metadata = feature_metadata.join_metadata(stacker_feature_metadata)
            else:
                feature_metadata = stacker_feature_metadata
        if feature_metadata is None:
            feature_metadata = FeatureMetadata(type_map_raw={})
        return feature_metadata

    def _get_stack_column_names(self, models: list[str]) -> tuple[list[str], int]:
        """
        Get the stack column names generated when the provided models are used as base models in a stack ensemble.
        Additionally output the number of columns per model as an int.
        """
        if self.problem_type in [MULTICLASS, SOFTCLASS]:
            stack_column_names = [
                stack_column_prefix + "_" + str(cls)
                for stack_column_prefix in models
                for cls in range(self.num_classes)
            ]
            num_columns_per_model = self.num_classes
        elif self.problem_type == QUANTILE:
            stack_column_names = [
                stack_column_prefix + "_" + str(q) for stack_column_prefix in models for q in self.quantile_levels
            ]
            num_columns_per_model = len(self.quantile_levels)
        else:
            stack_column_names = models
            num_columns_per_model = 1
        return stack_column_names, num_columns_per_model

    # You must have previously called fit() with cache_data=True
    # Fits _FULL versions of specified models, but does NOT link them (_FULL stackers will still use normal models as input)
    def refit_single_full(
        self,
        X=None,
        y=None,
        X_val=None,
        y_val=None,
        X_unlabeled=None,
        models=None,
        fit_strategy: Literal["sequential", "parallel"] = "sequential",
        **kwargs,
    ) -> list[str]:
        if fit_strategy == "parallel":
            logger.log(
                30, f"Note: refit_full does not yet support fit_strategy='parallel', switching to 'sequential'..."
            )
            fit_strategy = "sequential"
        if X is None:
            X = self.load_X()
        if X_val is None:
            X_val = self.load_X_val()
        if y is None:
            y = self.load_y()
        if y_val is None:
            y_val = self.load_y_val()

        if models is None:
            models = self.get_model_names()

        model_levels = dict()
        ignore_models = []
        ignore_stack_names = [REFIT_FULL_NAME]
        for stack_name in ignore_stack_names:
            ignore_models += self.get_model_names(
                stack_name=stack_name
            )  # get_model_names returns [] if stack_name does not exist
        models = [model for model in models if model not in ignore_models]
        for model in models:
            model_level = self.get_model_level(model)
            if model_level not in model_levels:
                model_levels[model_level] = []
            model_levels[model_level].append(model)

        levels = sorted(model_levels.keys())
        models_trained_full = []
        model_refit_map = {}  # FIXME: is this even used, remove?

        if fit_strategy == "sequential":
            for level in levels:
                models_level = model_levels[level]
                for model in models_level:
                    model_name, models_trained = _detached_refit_single_full(
                        _self=self,
                        model=model,
                        X=X,
                        y=y,
                        X_val=X_val,
                        y_val=y_val,
                        X_unlabeled=X_unlabeled,
                        level=level,
                        kwargs=kwargs,
                        fit_strategy=fit_strategy,
                    )
                    if len(models_trained) == 1:
                        model_refit_map[model_name] = models_trained[0]
                    for model_trained in models_trained:
                        self._update_model_attr(
                            model_trained,
                            refit_full=True,
                            refit_full_parent=model_name,
                            refit_full_parent_val_score=self.get_model_attribute(model_name, "val_score"),
                        )
                    models_trained_full += models_trained
        elif fit_strategy == "parallel":
            # -- Parallel refit
            ray = try_import_ray()

            # FIXME: Need a common utility class for initializing ray so we don't duplicate code
            if not ray.is_initialized():
                ray.init(log_to_driver=False, logging_level=logging.ERROR)

            distributed_manager = ParallelFitManager(
                mode="refit",
                func=_remote_refit_single_full,
                func_kwargs=dict(fit_strategy=fit_strategy),
                func_put_kwargs=dict(
                    _self=self,
                    X=X,
                    y=y,
                    X_val=X_val,
                    y_val=y_val,
                    X_unlabeled=X_unlabeled,
                    kwargs=kwargs,
                ),
                # TODO: check if this is available in the kwargs
                num_cpus=kwargs.get("total_resources", {}).get("num_cpus", 1),
                num_gpus=kwargs.get("total_resources", {}).get("num_gpus", 0),
                get_model_attribute_func=self.get_model_attribute,
                X=X,
                y=y,
            )

            for level in levels:
                models_trained_full_level = []
                distributed_manager.job_kwargs["level"] = level
                models_level = model_levels[level]

                logger.log(
                    20, f"Scheduling distributed model-workers for refitting {len(models_level)} L{level} models..."
                )
                unfinished_job_refs = distributed_manager.schedule_jobs(models_to_fit=models_level)

                while unfinished_job_refs:
                    finished, unfinished_job_refs = ray.wait(unfinished_job_refs, num_returns=1)
                    refit_full_parent, model_trained, model_path, model_type = ray.get(finished[0])

                    self._add_model(
                        model_type.load(path=os.path.join(self.path, model_path), reset_paths=self.reset_paths),
                        stack_name=REFIT_FULL_NAME,
                        level=level,
                        _is_refit=True,
                    )
                    model_refit_map[refit_full_parent] = model_trained
                    self._update_model_attr(
                        model_trained,
                        refit_full=True,
                        refit_full_parent=refit_full_parent,
                        refit_full_parent_val_score=self.get_model_attribute(refit_full_parent, "val_score"),
                    )
                    models_trained_full_level.append(model_trained)

                    logger.log(20, f"Finished refit model for {refit_full_parent}")
                    unfinished_job_refs += distributed_manager.schedule_jobs()

                logger.log(20, f"Finished distributed refitting for {len(models_trained_full_level)} L{level} models.")
                models_trained_full += models_trained_full_level
                distributed_manager.clean_job_state(unfinished_job_refs=unfinished_job_refs)

            distributed_manager.clean_up_ray()
        else:
            raise ValueError(f"Invalid value for fit_strategy: '{fit_strategy}'")

        keys_to_del = []
        for model in model_refit_map.keys():
            if model_refit_map[model] not in models_trained_full:
                keys_to_del.append(model)
        for key in keys_to_del:
            del model_refit_map[key]
        self.save()  # TODO: This could be more efficient by passing in arg to not save if called by refit_ensemble_full since it saves anyways later.
        return models_trained_full

    # Fits _FULL models and links them in the stack so _FULL models only use other _FULL models as input during stacking
    # If model is specified, will fit all _FULL models that are ancestors of the provided model, automatically linking them.
    # If no model is specified, all models are refit and linked appropriately.
    def refit_ensemble_full(self, model: str | list[str] = "all", **kwargs) -> dict:
        if model == "all":
            ensemble_set = self.get_model_names()
        elif isinstance(model, list):
            ensemble_set = self.get_minimum_models_set(model)
        else:
            if model == "best":
                model = self.get_model_best()
            ensemble_set = self.get_minimum_model_set(model)
        existing_models = self.get_model_names()
        ensemble_set_valid = []
        model_refit_map = self.model_refit_map()
        for model in ensemble_set:
            if model in model_refit_map and model_refit_map[model] in existing_models:
                logger.log(
                    20,
                    f"Model '{model}' already has a refit _FULL model: '{model_refit_map[model]}', skipping refit...",
                )
            else:
                ensemble_set_valid.append(model)
        if ensemble_set_valid:
            models_trained_full = self.refit_single_full(models=ensemble_set_valid, **kwargs)
        else:
            models_trained_full = []

        model_refit_map = self.model_refit_map()
        for model_full in models_trained_full:
            # TODO: Consider moving base model info to a separate pkl file so that it can be edited without having to load/save the model again
            #  Downside: Slower inference speed when models are not persisted in memory prior.
            model_loaded = self.load_model(model_full)
            if isinstance(model_loaded, StackerEnsembleModel):
                for stack_column_prefix in model_loaded.stack_column_prefix_lst:
                    base_model = model_loaded.stack_column_prefix_to_model_map[stack_column_prefix]
                    new_base_model = model_refit_map[base_model]
                    new_base_model_type = self.get_model_attribute(model=new_base_model, attribute="type")
                    new_base_model_path = self.get_model_attribute(model=new_base_model, attribute="path")

                    model_loaded.base_model_paths_dict[new_base_model] = new_base_model_path
                    model_loaded.base_model_types_dict[new_base_model] = new_base_model_type
                    model_loaded.base_model_names.append(new_base_model)
                    model_loaded.stack_column_prefix_to_model_map[stack_column_prefix] = new_base_model

            model_loaded.save()  # TODO: Avoid this!

            # Remove old edges and add new edges
            edges_to_remove = list(self.model_graph.in_edges(model_loaded.name))
            self.model_graph.remove_edges_from(edges_to_remove)
            if isinstance(model_loaded, StackerEnsembleModel):
                for stack_column_prefix in model_loaded.stack_column_prefix_lst:
                    base_model_name = model_loaded.stack_column_prefix_to_model_map[stack_column_prefix]
                    self.model_graph.add_edge(base_model_name, model_loaded.name)

        self.save()
        return self.model_refit_map()

    def get_refit_full_parent(self, model: str) -> str:
        """Get refit full model's parent. If model does not have a parent, return `model`."""
        return self.get_model_attribute(model=model, attribute="refit_full_parent", default=model)

    def get_model_best(
        self,
        can_infer: bool | None = None,
        allow_full: bool = True,
        infer_limit: float | None = None,
        infer_limit_as_child: bool = False,
    ) -> str:
        """
        Returns the name of the model with the best validation score that satisfies all specified constraints.
        If no model satisfies the constraints, an AssertionError will be raised.

        Parameters
        ----------
        can_infer: bool, default = None
            If True, only consider models that can infer.
            If False, only consider models that can't infer.
            If None, consider all models.
        allow_full: bool, default = True
            If True, consider all models.
            If False, disallow refit_full models.
        infer_limit: float, default = None
            The maximum time in seconds per sample that a model is allowed to take during inference.
            If None, consider all models.
            If specified, consider only models that have a lower predict time per sample than `infer_limit`.
        infer_limit_as_child: bool, default = False
            If True, use the predict time per sample of the (theoretical) refit version of the model.
                If the model is already refit, the predict time per sample is unchanged.
            If False, use the predict time per sample of the model.

        Returns
        -------
        model: str
            The string name of the model with the best metric score that satisfies all constraints.
        """
        models = self.get_model_names(can_infer=can_infer)
        if not models:
            raise AssertionError("Trainer has no fit models that can infer.")
        models_full = self.get_models_attribute_dict(models=models, attribute="refit_full_parent")
        if not allow_full:
            models = [model for model in models if model not in models_full]

        predict_1_time_attribute = None
        if infer_limit is not None:
            if infer_limit_as_child:
                predict_1_time_attribute = "predict_1_child_time"
            else:
                predict_1_time_attribute = "predict_1_time"
            models_predict_1_time = self.get_models_attribute_full(models=models, attribute=predict_1_time_attribute)
            models_og = copy.deepcopy(models)
            for model_key in models_predict_1_time:
                if models_predict_1_time[model_key] is None or models_predict_1_time[model_key] > infer_limit:
                    models.remove(model_key)
            if models_og and not models:
                # get the fastest model
                models_predict_time_list = [models_predict_1_time[m] for m in models_og]
                min_time = np.array(models_predict_time_list).min()
                infer_limit_new = min_time * 1.2  # Give 20% lee-way
                logger.log(
                    30,
                    f"WARNING: Impossible to satisfy infer_limit constraint. Relaxing constraint from {infer_limit} to {infer_limit_new} ...",
                )
                models = models_og
                for model_key in models_predict_1_time:
                    if models_predict_1_time[model_key] > infer_limit_new:
                        models.remove(model_key)
        if not models:
            raise AssertionError(
                f"Trainer has no fit models that can infer while satisfying the constraints: (infer_limit={infer_limit}, allow_full={allow_full})."
            )
        model_performances = self.get_models_attribute_dict(models=models, attribute="val_score")

        predict_time_attr = predict_1_time_attribute if predict_1_time_attribute is not None else "predict_time"
        models_predict_time = self.get_models_attribute_full(models=models, attribute=predict_time_attr)

        perfs = [
            (m, model_performances[m], models_predict_time[m]) for m in models if model_performances[m] is not None
        ]
        if not perfs:
            models = [m for m in models if m in models_full]
            perfs = [
                (m, self.get_model_attribute(model=m, attribute="refit_full_parent_val_score"), models_predict_time[m])
                for m in models
            ]
            if not perfs:
                raise AssertionError(
                    "No fit models that can infer exist with a validation score to choose the best model."
                )
            elif not allow_full:
                raise AssertionError(
                    "No fit models that can infer exist with a validation score to choose the best model, but refit_full models exist. Set `allow_full=True` to get the best refit_full model."
                )
        return max(perfs, key=lambda i: (i[1], -i[2]))[0]

    def save_model(self, model, reduce_memory=True):
        # TODO: In future perhaps give option for the reduce_memory_size arguments, perhaps trainer level variables specified by user?
        if reduce_memory:
            model.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
        if self.low_memory:
            model.save()
        else:
            self.models[model.name] = model

    def save(self) -> None:
        models = self.models
        if self.low_memory:
            self.models = {}
        save_pkl.save(path=os.path.join(self.path, self.trainer_file_name), object=self)
        if self.low_memory:
            self.models = models

    def compile(self, model_names="all", with_ancestors=False, compiler_configs=None) -> list[str]:
        """
        Compile a list of models for accelerated prediction.

        Parameters
        ----------
        model_names : str or list
            A list of model names for model compilation. Alternatively, this can be 'all' or 'best'.
        compiler_configs: dict, default=None
            Model specific compiler options.
            This can be useful to specify the compiler backend for a specific model,
            e.g. {"RandomForest": {"compiler": "onnx"}}
        """
        if model_names == "all":
            model_names = self.get_model_names(can_infer=True)
        elif model_names == "best":
            if self.model_best is not None:
                model_names = [self.model_best]
            else:
                model_names = [self.get_model_best(can_infer=True)]
        if not isinstance(model_names, list):
            raise ValueError(f"model_names must be a list of model names. Invalid value: {model_names}")
        if with_ancestors:
            model_names = self.get_minimum_models_set(model_names)

        logger.log(20, f"Compiling {len(model_names)} Models ...")
        total_compile_time = 0

        model_names_to_compile = []
        model_names_to_configs_dict = dict()
        for model_name in model_names:
            model_type_inner = self.get_model_attribute(model_name, "type_inner")
            # Get model specific compiler options
            # Model type can be described with either model type, or model name as string
            if model_name in compiler_configs:
                config = compiler_configs[model_name]
            elif model_type_inner in compiler_configs:
                config = compiler_configs[model_type_inner]
            else:
                config = None
            if config is not None:
                model_names_to_compile.append(model_name)
                model_names_to_configs_dict[model_name] = config
            else:
                logger.log(20, f"Skipping compilation for {model_name} ... (No config specified)")
        for model_name in model_names_to_compile:
            model = self.load_model(model_name)
            config = model_names_to_configs_dict[model_name]

            # Check if already compiled, or if can't compile due to missing dependencies,
            # or if model hasn't implemented compiling.
            if "compiler" in config and model.get_compiler_name() == config["compiler"]:
                logger.log(
                    20,
                    f'Skipping compilation for {model_name} ... (Already compiled with "{model.get_compiler_name()}" backend)',
                )
            elif model.can_compile(compiler_configs=config):
                logger.log(20, f"Compiling model: {model.name} ... Config = {config}")
                compile_start_time = time.time()
                model.compile(compiler_configs=config)
                compile_end_time = time.time()
                model.compile_time = compile_end_time - compile_start_time
                compile_type = model.get_compiler_name()
                total_compile_time += model.compile_time

                # Update model_graph in order to put compile_time into leaderboard,
                # since models are saved right after training.
                self.model_graph.nodes[model.name]["compile_time"] = model.compile_time
                self.save_model(model, reduce_memory=False)
                logger.log(20, f'\tCompiled model with "{compile_type}" backend ...')
                logger.log(20, f"\t{round(model.compile_time, 2)}s\t = Compile    runtime")
            else:
                logger.log(
                    20,
                    f"Skipping compilation for {model.name} ... (Unable to compile with the provided config: {config})",
                )
        logger.log(20, f"Finished compiling models, total runtime = {round(total_compile_time, 2)}s.")
        self.save()
        return model_names

    def persist(self, model_names="all", with_ancestors=False, max_memory=None) -> list[str]:
        if model_names == "all":
            model_names = self.get_model_names()
        elif model_names == "best":
            if self.model_best is not None:
                model_names = [self.model_best]
            else:
                model_names = [self.get_model_best(can_infer=True)]
        if not isinstance(model_names, list):
            raise ValueError(f"model_names must be a list of model names. Invalid value: {model_names}")
        if with_ancestors:
            model_names = self.get_minimum_models_set(model_names)
        model_names_already_persisted = [model_name for model_name in model_names if model_name in self.models]
        if model_names_already_persisted:
            logger.log(
                30,
                f"The following {len(model_names_already_persisted)} models were already persisted and will be ignored in the model loading process: {model_names_already_persisted}",
            )
        model_names = [model_name for model_name in model_names if model_name not in model_names_already_persisted]
        if not model_names:
            logger.log(
                30,
                f"No valid unpersisted models were specified to be persisted, so no change in model persistence was performed.",
            )
            return []
        if max_memory is not None:

            @disable_if_lite_mode(ret=True)
            def _check_memory():
                info = self.get_models_info(model_names)
                model_mem_size_map = {model: info[model]["memory_size"] for model in model_names}
                for model in model_mem_size_map:
                    if "children_info" in info[model]:
                        for child in info[model]["children_info"].values():
                            model_mem_size_map[model] += child["memory_size"]
                total_mem_required = sum(model_mem_size_map.values())
                available_mem = ResourceManager.get_available_virtual_mem()
                memory_proportion = total_mem_required / available_mem
                if memory_proportion > max_memory:
                    logger.log(
                        30,
                        f"Models will not be persisted in memory as they are expected to require {round(memory_proportion * 100, 2)}% of memory, which is greater than the specified max_memory limit of {round(max_memory * 100, 2)}%.",
                    )
                    logger.log(
                        30,
                        f"\tModels will be loaded on-demand from disk to maintain safe memory usage, increasing inference latency. If inference latency is a concern, try to use smaller models or increase the value of max_memory.",
                    )
                    return False
                else:
                    logger.log(
                        20,
                        f"Persisting {len(model_names)} models in memory. Models will require {round(memory_proportion * 100, 2)}% of memory.",
                    )
                return True

            if not _check_memory():
                return []

        models = []
        for model_name in model_names:
            model = self.load_model(model_name)
            self.models[model.name] = model
            models.append(model)

        for model in models:
            # TODO: Move this to model code
            if isinstance(model, BaggedEnsembleModel):
                for fold, fold_model in enumerate(model.models):
                    if isinstance(fold_model, str):
                        model.models[fold] = model.load_child(fold_model)
        return model_names

    def unpersist(self, model_names="all") -> list:
        if model_names == "all":
            model_names = list(self.models.keys())
        if not isinstance(model_names, list):
            raise ValueError(f"model_names must be a list of model names. Invalid value: {model_names}")
        unpersisted_models = []
        for model in model_names:
            if model in self.models:
                self.models.pop(model)
                unpersisted_models.append(model)
        if unpersisted_models:
            logger.log(20, f"Unpersisted {len(unpersisted_models)} models: {unpersisted_models}")
        else:
            logger.log(
                30,
                f"No valid persisted models were specified to be unpersisted, so no change in model persistence was performed.",
            )
        return unpersisted_models

    def generate_weighted_ensemble(
        self,
        X,
        y,
        level,
        base_model_names,
        k_fold=1,
        n_repeats=1,
        stack_name=None,
        hyperparameters=None,
        ag_args_fit=None,
        time_limit=None,
        name_suffix: str | None = None,
        save_bag_folds=None,
        check_if_best=True,
        child_hyperparameters=None,
        get_models_func=None,
        total_resources: dict | None = None,
    ) -> list[str]:
        if get_models_func is None:
            get_models_func = self.construct_model_templates
        if len(base_model_names) == 0:
            logger.log(20, "No base models to train on, skipping weighted ensemble...")
            return []

        if child_hyperparameters is None:
            child_hyperparameters = {}

        if save_bag_folds is None:
            can_infer_dict = self.get_models_attribute_dict("can_infer", models=base_model_names)
            if False in can_infer_dict.values():
                save_bag_folds = False
            else:
                save_bag_folds = True

        feature_metadata = self.get_feature_metadata(use_orig_features=False, base_models=base_model_names)

        base_model_paths_dict = self.get_models_attribute_dict(attribute="path", models=base_model_names)
        base_model_paths_dict = {key: os.path.join(self.path, val) for key, val in base_model_paths_dict.items()}
        weighted_ensemble_model, _ = get_models_func(
            hyperparameters={
                "default": {
                    "ENS_WEIGHTED": [child_hyperparameters],
                }
            },
            ensemble_type=WeightedEnsembleModel,
            ensemble_kwargs=dict(
                base_model_names=base_model_names,
                base_model_paths_dict=base_model_paths_dict,
                base_model_types_dict=self.get_models_attribute_dict(attribute="type", models=base_model_names),
                base_model_types_inner_dict=self.get_models_attribute_dict(
                    attribute="type_inner", models=base_model_names
                ),
                base_model_performances_dict=self.get_models_attribute_dict(
                    attribute="val_score", models=base_model_names
                ),
                hyperparameters=hyperparameters,
                random_state=level + self.random_state,
            ),
            ag_args={"name_bag_suffix": ""},
            ag_args_fit=ag_args_fit,
            ag_args_ensemble={"save_bag_folds": save_bag_folds},
            name_suffix=name_suffix,
            level=level,
        )
        weighted_ensemble_model = weighted_ensemble_model[0]
        w = None
        if self.weight_evaluation:
            X, w = extract_column(X, self.sample_weight)
        models = self._train_multi(
            X=X,
            y=y,
            X_val=None,
            y_val=None,
            models=[weighted_ensemble_model],
            k_fold=k_fold,
            n_repeats=n_repeats,
            hyperparameter_tune_kwargs=None,
            stack_name=stack_name,
            level=level,
            time_limit=time_limit,
            ens_sample_weight=w,
            fit_kwargs=dict(
                feature_metadata=feature_metadata, num_classes=self.num_classes, groups=None
            ),  # FIXME: Is this the right way to do this?
            total_resources=total_resources,
        )
        for weighted_ensemble_model_name in models:
            if check_if_best and weighted_ensemble_model_name in self.get_model_names():
                if self.model_best is None:
                    self.model_best = weighted_ensemble_model_name
                else:
                    best_score = self.get_model_attribute(self.model_best, "val_score")
                    cur_score = self.get_model_attribute(weighted_ensemble_model_name, "val_score")
                    if best_score is not None and cur_score > best_score:
                        # new best model
                        self.model_best = weighted_ensemble_model_name
        return models

    def _train_single(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: AbstractModel,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        total_resources: dict = None,
        **model_fit_kwargs,
    ) -> AbstractModel:
        """
        Trains model but does not add the trained model to this Trainer.
        Returns trained model object.
        """
        model = model.fit(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            total_resources=total_resources,
            **model_fit_kwargs,
        )
        return model

    def _train_and_save(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: AbstractModel,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        X_pseudo: pd.DataFrame | None = None,
        y_pseudo: pd.DataFrame | None = None,
        time_limit: float | None = None,
        stack_name: str = "core",
        level: int = 1,
        compute_score: bool = True,
        total_resources: dict | None = None,
        errors: Literal["ignore", "raise"] = "ignore",
        errors_ignore: list | None = None,
        errors_raise: list | None = None,
        is_ray_worker: bool = False,
        **model_fit_kwargs,
    ) -> list[str]:
        """
        Trains model and saves it to disk, returning a list with a single element: The name of the model, or no elements if training failed.
        If the model name is returned:
            The model can be accessed via self.load_model(model.name).
            The model will have metadata information stored in self.model_graph.
            The model's name will be appended to self.models_level[stack_name][level]
            The model will be accessible and usable through any Trainer function that takes as input 'model' or 'model_name'.
        Note: self._train_and_save should not be used outside of self._train_single_full

        Parameters
        ----------
        errors: Literal["ignore", "raise"], default = "ignore"
            Determines how model fit exceptions are handled.
            If "ignore", will ignore all model exceptions during fit. If an exception occurs, an empty list is returned.
            If "raise", will raise the model exception if it occurs.
            Can be overwritten by `errors_ignore` and `errors_raise`.
        errors_ignore: list[str], optional
            The exception types specified in `errors_ignore` will be treated as if `errors="ignore"`.
        errors_raise: list[str], optional
            The exception types specified in `errors_raise` will be treated as if `errors="raise"`.

        """
        fit_start_time = time.time()
        model_names_trained = []
        y_pred_proba_val = None

        is_distributed_mode = DistributedContext.is_distributed_mode() or is_ray_worker

        fit_log_message = f"Fitting model: {model.name} ..."
        if time_limit is not None:
            time_left_total = time_limit
            not_enough_time = False
            if time_limit <= 0:
                not_enough_time = True
            elif self._time_limit is not None and self._time_train_start is not None:
                time_left_total = self._time_limit - (fit_start_time - self._time_train_start)
                # If only a very small amount of time remains, skip training
                min_time_required = min(self._time_limit * 0.01, 10)
                if (time_left_total < min_time_required) and (time_limit < min_time_required):
                    not_enough_time = True
            if not_enough_time:
                skip_msg = f"Skipping {model.name} due to lack of time remaining."
                not_enough_time_exception = InsufficientTime(skip_msg)
                if self._check_raise_exception(
                    exception=not_enough_time_exception,
                    errors=errors,
                    errors_ignore=errors_ignore,
                    errors_raise=errors_raise,
                ):
                    raise not_enough_time_exception
                else:
                    logger.log(15, skip_msg)
                    return []
            fit_log_message += (
                f" Training model for up to {time_limit:.2f}s of the {time_left_total:.2f}s of remaining time."
            )
        logger.log(10 if is_distributed_mode else 20, fit_log_message)

        if isinstance(model, BaggedEnsembleModel) and not compute_score:
            # Do not perform OOF predictions when we don't compute a score.
            model_fit_kwargs["_skip_oof"] = True
        if not isinstance(model, BaggedEnsembleModel):
            model_fit_kwargs.setdefault("log_resources", True)

        model_fit_kwargs = dict(
            model=model,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            time_limit=time_limit,
            total_resources=total_resources,
            **model_fit_kwargs,
        )

        # If model is not bagged model and not stacked then pseudolabeled data needs to be incorporated at this level
        # Bagged model does validation on the fit level where as single models do it separately. Hence this if statement
        # is required
        if (
            not isinstance(model, BaggedEnsembleModel)
            and X_pseudo is not None
            and y_pseudo is not None
            and X_pseudo.columns.equals(X.columns)
        ):
            assert_pseudo_column_match(X=X, X_pseudo=X_pseudo)
            # Needs .astype(X.dtypes) because pd.concat will convert categorical features to int/float unexpectedly. Need to convert them back to original.
            X_w_pseudo = pd.concat([X, X_pseudo], ignore_index=True).astype(X.dtypes)
            y_w_pseudo = pd.concat([y, y_pseudo], ignore_index=True)
            logger.log(15, f"{len(X_pseudo)} extra rows of pseudolabeled data added to training set for {model.name}")
            model_fit_kwargs["X"] = X_w_pseudo
            model_fit_kwargs["y"] = y_w_pseudo
        else:
            model_fit_kwargs["X"] = X
            model_fit_kwargs["y"] = y
            if level > 1:
                if X_pseudo is not None and y_pseudo is not None:
                    logger.log(15, f"Dropping pseudo in stacking layer due to missing out-of-fold predictions")
            else:
                model_fit_kwargs["X_pseudo"] = X_pseudo
                model_fit_kwargs["y_pseudo"] = y_pseudo

        exception = None
        try:
            model = self._train_single(**model_fit_kwargs)

            fit_end_time = time.time()
            if self.weight_evaluation:
                w = model_fit_kwargs.get("sample_weight", None)
                w_val = model_fit_kwargs.get("sample_weight_val", None)
            else:
                w = None
                w_val = None
            if not compute_score:
                score = None
                model.predict_time = None
            elif X_val is not None and y_val is not None:
                y_pred_proba_val = model.predict_proba(X_val, record_time=True)
                score = model.score_with_y_pred_proba(y=y_val, y_pred_proba=y_pred_proba_val, sample_weight=w_val)
            elif isinstance(model, BaggedEnsembleModel):
                if model.is_valid_oof() or isinstance(model, WeightedEnsembleModel):
                    score = model.score_with_oof(y=y, sample_weight=w)
                else:
                    score = None
            else:
                score = None
            pred_end_time = time.time()
            if model.fit_time is None:
                model.fit_time = fit_end_time - fit_start_time
            if model.predict_time is None and score is not None:
                model.predict_time = pred_end_time - fit_end_time
            model.val_score = score
            # TODO: Add recursive=True to avoid repeatedly loading models each time this is called for bagged ensembles (especially during repeated bagging)
            self.save_model(model=model)
        except Exception as exc:
            if self.raise_on_model_failure:
                # immediately raise instead of skipping to next model, useful for debugging during development
                logger.warning(
                    "Model failure occurred... Raising exception instead of continuing to next model. (raise_on_model_failure=True)"
                )
                raise exc
            exception = exc  # required to reference exc outside of `except` statement
            del_model = True
            if isinstance(exception, TimeLimitExceeded):
                logger.log(20, f"\tTime limit exceeded... Skipping {model.name}.")
            elif isinstance(exception, NotEnoughMemoryError):
                logger.warning(f"\tNot enough memory to train {model.name}... Skipping this model.")
            elif isinstance(exception, NoStackFeatures):
                logger.warning(f"\tNo stack features to train {model.name}... Skipping this model. {exception}")
            elif isinstance(exception, NotValidStacker):
                logger.warning(f"\tStacking disabled for {model.name}... Skipping this model. {exception}")
            elif isinstance(exception, NoValidFeatures):
                logger.warning(f"\tNo valid features to train {model.name}... Skipping this model.")
            elif isinstance(exception, NoGPUError):
                logger.warning(f"\tNo GPUs available to train {model.name}... Skipping this model.")
            elif isinstance(exception, NotEnoughCudaMemoryError):
                logger.warning(f"\tNot enough CUDA memory available to train {model.name}... Skipping this model.")
            elif isinstance(exception, ImportError):
                logger.error(
                    f"\tWarning: Exception caused {model.name} to fail during training (ImportError)... Skipping this model."
                )
                logger.error(f"\t\t{exception}")
                del_model = False
                if self.verbosity > 2:
                    logger.exception("Detailed Traceback:")
            else:  # all other exceptions
                logger.error(
                    f"\tWarning: Exception caused {model.name} to fail during training... Skipping this model."
                )
                logger.error(f"\t\t{exception}")
                if self.verbosity > 0:
                    logger.exception("Detailed Traceback:")
            crash_time = time.time()
            total_time = crash_time - fit_start_time
            tb = traceback.format_exc()
            model_info = self.get_model_info(model=model)
            self._models_failed_to_train_errors[model.name] = dict(
                exc_type=exception.__class__.__name__,
                exc_str=str(exception),
                exc_traceback=tb,
                model_info=model_info,
                total_time=total_time,
            )

            if del_model:
                del model
        else:
            self._add_model(
                model=model,
                stack_name=stack_name,
                level=level,
                y_pred_proba_val=y_pred_proba_val,
                is_ray_worker=is_ray_worker,
            )
            model_names_trained.append(model.name)
            if self.low_memory:
                del model
        if exception is not None:
            if self._check_raise_exception(
                exception=exception, errors=errors, errors_ignore=errors_ignore, errors_raise=errors_raise
            ):
                raise exception
        return model_names_trained

    # FIXME: v1.0 Move to AbstractModel for most fields
    def _get_model_metadata(self, model: AbstractModel, stack_name: str = "core", level: int = 1) -> dict[str, Any]:
        """
        Returns the model metadata used to initialize a node in the DAG (self.model_graph).
        """
        if isinstance(model, BaggedEnsembleModel):
            type_inner = model._child_type
        else:
            type_inner = type(model)
        num_children = len(model.models) if hasattr(model, "models") else 1
        predict_child_time = model.predict_time / num_children if model.predict_time is not None else None
        predict_1_child_time = model.predict_1_time / num_children if model.predict_1_time is not None else None
        fit_metadata = model.get_fit_metadata()

        model_param_aux = getattr(model, "_params_aux_child", model.params_aux)
        model_metadata = dict(
            fit_time=model.fit_time,
            compile_time=model.compile_time,
            predict_time=model.predict_time,
            predict_1_time=model.predict_1_time,
            predict_child_time=predict_child_time,
            predict_1_child_time=predict_1_child_time,
            predict_n_time_per_row=model.predict_n_time_per_row,
            predict_n_size=model.predict_n_size,
            val_score=model.val_score,
            eval_metric=model.eval_metric.name,
            stopping_metric=model.stopping_metric.name,
            path=os.path.relpath(model.path, self.path).split(os.sep),  # model's relative path to trainer
            type=type(model),  # Outer type, can be BaggedEnsemble, StackEnsemble (Type that is able to load the model)
            type_inner=type_inner,  # Inner type, if Ensemble then it is the type of the inner model (May not be able to load with this type)
            can_infer=model.can_infer(),
            can_fit=model.can_fit(),
            is_valid=model.is_valid(),
            stack_name=stack_name,
            level=level,
            num_children=num_children,
            fit_num_cpus=model.fit_num_cpus,
            fit_num_gpus=model.fit_num_gpus,
            fit_num_cpus_child=model.fit_num_cpus_child,
            fit_num_gpus_child=model.fit_num_gpus_child,
            refit_full_requires_gpu=(model.fit_num_gpus_child is not None)
            and (model.fit_num_gpus_child >= 1)
            and model._user_params.get("refit_folds", False),
            **fit_metadata,
        )
        return model_metadata

    def _add_model(
        self,
        model: AbstractModel,
        stack_name: str = "core",
        level: int = 1,
        y_pred_proba_val=None,
        _is_refit=False,
        is_distributed_main=False,
        is_ray_worker: bool = False,
    ) -> bool:
        """
        Registers the fit model in the Trainer object. Stores information such as model performance, save path, model type, and more.
        To use a model in Trainer, self._add_model must be called.
        If self.low_memory, then the model object will be deleted after this call. Use Trainer directly to leverage the model further.

        Parameters
        ----------
        model : AbstractModel
            Model which has been fit. This model will be registered to the Trainer.
        stack_name : str, default 'core'
            Stack name to assign the model to. This is used for advanced functionality.
        level : int, default 1
            Stack level of the stack name to assign the model to. This is used for advanced functionality.
            The model's name is appended to self.models_level[stack_name][level]
            The model's base_models (if it has any) must all be a lower level than the model.
        is_distributed_main: bool, default = False
            If True, the main process in distributed training is calling this function.
            This is used to avoid redundant logging in distributed training.

        Returns
        -------
        boolean, True if model was registered, False if model was found to be invalid and not registered.
        """
        if model.val_score is not None and np.isnan(model.val_score):
            msg = (
                f"WARNING: {model.name} has a val_score of {model.val_score} (NaN)! "
                f"This should never happen. The model will not be saved to avoid instability."
            )
            if self.raise_on_model_failure:
                raise AssertionError(
                    f"{msg} Raising an exception because `raise_on_model_failure={self.raise_on_model_failure}`."
                )
            else:
                logger.warning(msg)
            return False
        # TODO: Add to HPO

        node_attributes = self._get_model_metadata(model=model, stack_name=stack_name, level=level)
        if y_pred_proba_val is not None:
            # Cache y_pred_proba_val for later reuse to avoid redundant predict calls
            self._save_model_y_pred_proba_val(model=model.name, y_pred_proba_val=y_pred_proba_val)
            node_attributes["cached_y_pred_proba_val"] = True

        self.model_graph.add_node(
            model.name,
            **node_attributes,
        )
        if isinstance(model, StackerEnsembleModel):
            prior_models = self.get_model_names()
            # TODO: raise exception if no base models and level != 1?
            for stack_column_prefix in model.stack_column_prefix_lst:
                base_model_name = model.stack_column_prefix_to_model_map[stack_column_prefix]
                if base_model_name not in prior_models:
                    raise AssertionError(
                        f"Model '{model.name}' depends on model '{base_model_name}', but '{base_model_name}' is not registered as a trained model! Valid models: {prior_models}"
                    )
                elif level <= self.model_graph.nodes[base_model_name]["level"]:
                    raise AssertionError(
                        f"Model '{model.name}' depends on model '{base_model_name}', but '{base_model_name}' is not in a lower stack level. ('{model.name}' level: {level}, '{base_model_name}' level: {self.model_graph.nodes[base_model_name]['level']})"
                    )
                self.model_graph.add_edge(base_model_name, model.name)
        self._log_model_stats(
            model, _is_refit=_is_refit, is_distributed_main=is_distributed_main, is_ray_worker=is_ray_worker
        )
        if self.low_memory:
            del model
        return True

    def _path_attr_model(self, model: str):
        """Returns directory where attributes are cached"""
        return os.path.join(self._path_attr, model)

    def _path_to_model_attr(self, model: str, attribute: str):
        """Returns pkl file path for a cached model attribute"""
        return os.path.join(self._path_attr_model(model), f"{attribute}.pkl")

    def _save_model_y_pred_proba_val(self, model: str, y_pred_proba_val):
        """Cache y_pred_proba_val for later reuse to avoid redundant predict calls"""
        save_pkl.save(
            path=self._path_to_model_attr(model=model, attribute="y_pred_proba_val"), object=y_pred_proba_val
        )

    def _load_model_y_pred_proba_val(self, model: str):
        """Load cached y_pred_proba_val for a given model"""
        return load_pkl.load(path=self._path_to_model_attr(model=model, attribute="y_pred_proba_val"))

    # TODO: Once Python min-version is 3.8, can refactor to use positional-only argument for model
    #  https://peps.python.org/pep-0570/#empowering-library-authors
    #  Currently this method cannot accept the attribute key 'model' without making usage ugly.
    def _update_model_attr(self, model: str, **attributes):
        """Updates model node in graph with the input attributes dictionary"""
        if model not in self.model_graph:
            raise AssertionError(f'"{model}" is not a key in self.model_graph, cannot add attributes: {attributes}')
        self.model_graph.nodes[model].update(attributes)

    def _log_model_stats(self, model, _is_refit=False, is_distributed_main=False, is_ray_worker: bool = False):
        """Logs model fit time, val score, predict time, and predict_1_time"""
        model = self.load_model(model)
        print_weights = model._get_tags().get("print_weights", False)

        is_log_during_distributed_fit = DistributedContext.is_distributed_mode() and (not is_distributed_main)
        if is_ray_worker:
            is_log_during_distributed_fit = True
        log_level = 10 if is_log_during_distributed_fit else 20

        if print_weights:
            model_weights = model._get_model_weights()
            model_weights = {k: round(v, 3) for k, v in model_weights.items()}
            msg_weights = ""
            is_first = True
            for key, value in sorted(model_weights.items(), key=lambda x: x[1], reverse=True):
                if not is_first:
                    msg_weights += ", "
                msg_weights += f"'{key}': {value}"
                is_first = False
            logger.log(log_level, f"\tEnsemble Weights: {{{msg_weights}}}")
        if model.val_score is not None:
            if model.eval_metric.name != self.eval_metric.name:
                logger.log(log_level, f"\tNote: model has different eval_metric than default.")
            if not model.eval_metric.greater_is_better_internal:
                sign_str = "-"
            else:
                sign_str = ""
            logger.log(
                log_level, f"\t{round(model.val_score, 4)}\t = Validation score   ({sign_str}{model.eval_metric.name})"
            )
        if model.fit_time is not None:
            logger.log(log_level, f"\t{round(model.fit_time, 2)}s\t = Training   runtime")
        if model.predict_time is not None:
            logger.log(log_level, f"\t{round(model.predict_time, 2)}s\t = Validation runtime")
        predict_n_time_per_row = self.get_model_attribute_full(model=model.name, attribute="predict_n_time_per_row")
        predict_n_size = self.get_model_attribute_full(model=model.name, attribute="predict_n_size", func=min)
        if predict_n_time_per_row is not None and predict_n_size is not None:
            logger.log(
                15,
                f"\t{round(1 / (predict_n_time_per_row if predict_n_time_per_row else np.finfo(np.float16).eps), 1)}"
                f"\t = Inference  throughput (rows/s | {int(predict_n_size)} batch size)",
            )
        if model.predict_1_time is not None:
            fit_metadata = model.get_fit_metadata()
            predict_1_batch_size = fit_metadata.get("predict_1_batch_size", None)
            assert predict_1_batch_size is not None, (
                "predict_1_batch_size cannot be None if predict_1_time is not None"
            )

            if _is_refit:
                predict_1_time = self.get_model_attribute(model=model.name, attribute="predict_1_child_time")
                predict_1_time_full = self.get_model_attribute_full(model=model.name, attribute="predict_1_child_time")
            else:
                predict_1_time = model.predict_1_time
                predict_1_time_full = self.get_model_attribute_full(model=model.name, attribute="predict_1_time")

            predict_1_time_log, time_unit = convert_time_in_s_to_log_friendly(time_in_sec=predict_1_time)
            logger.log(
                log_level,
                f"\t{round(predict_1_time_log, 3)}{time_unit}\t = Validation runtime (1 row | {predict_1_batch_size} batch size | MARGINAL)",
            )

            predict_1_time_full_log, time_unit = convert_time_in_s_to_log_friendly(time_in_sec=predict_1_time_full)
            logger.log(
                log_level,
                f"\t{round(predict_1_time_full_log, 3)}{time_unit}\t = Validation runtime (1 row | {predict_1_batch_size} batch size)",
            )

            if not _is_refit:
                predict_1_time_child = self.get_model_attribute(model=model.name, attribute="predict_1_child_time")
                predict_1_time_child_log, time_unit = convert_time_in_s_to_log_friendly(
                    time_in_sec=predict_1_time_child
                )
                logger.log(
                    log_level,
                    f"\t{round(predict_1_time_child_log, 3)}{time_unit}\t = Validation runtime (1 row | {predict_1_batch_size} batch size | REFIT | MARGINAL)",
                )

                predict_1_time_full_child = self.get_model_attribute_full(
                    model=model.name, attribute="predict_1_child_time"
                )
                predict_1_time_full_child_log, time_unit = convert_time_in_s_to_log_friendly(
                    time_in_sec=predict_1_time_full_child
                )
                logger.log(
                    log_level,
                    f"\t{round(predict_1_time_full_child_log, 3)}{time_unit}\t = Validation runtime (1 row | {predict_1_batch_size} batch size | REFIT)",
                )

    # TODO: Split this to avoid confusion, HPO should go elsewhere?
    def _train_single_full(
        self,
        X,
        y,
        model: AbstractModel,
        X_unlabeled=None,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        X_pseudo=None,
        y_pseudo=None,
        hyperparameter_tune_kwargs=None,
        stack_name="core",
        k_fold=None,
        k_fold_start=0,
        k_fold_end=None,
        n_repeats=None,
        n_repeat_start=0,
        level=1,
        time_limit=None,
        fit_kwargs=None,
        compute_score=True,
        total_resources: dict | None = None,
        errors: Literal["ignore", "raise"] = "ignore",
        errors_ignore: list | None = None,
        errors_raise: list | None = None,
        is_ray_worker: bool = False,
        label_cleaner: None | LabelCleaner = None,
        **kwargs,
    ) -> list[str]:
        """
        Trains a model, with the potential to train multiple versions of this model with hyperparameter tuning and feature pruning.
        Returns a list of successfully trained and saved model names.
        Models trained from this method will be accessible in this Trainer.

        Parameters
        ----------
        errors: Literal["ignore", "raise"], default = "ignore"
            Determines how model fit exceptions are handled.
            If "ignore", will ignore all model exceptions during fit. If an exception occurs, an empty list is returned.
            If "raise", will raise the model exception if it occurs.
            Can be overwritten by `errors_ignore` and `errors_raise`.
        errors_ignore: list[str], optional
            The exception types specified in `errors_ignore` will be treated as if `errors="ignore"`.
        errors_raise: list[str], optional
            The exception types specified in `errors_raise` will be treated as if `errors="raise"`.
        """
        if self._callback_early_stop:
            return []
        check_callbacks = k_fold_start == 0 and n_repeat_start == 0 and not is_ray_worker
        skip_model = False
        if self.callbacks and check_callbacks:
            skip_model, time_limit = self._callbacks_before_fit(
                model=model,
                time_limit=time_limit,
                stack_name=stack_name,
                level=level,
            )
        if self._callback_early_stop or skip_model:
            return []

        model_fit_kwargs = self._get_model_fit_kwargs(
            X=X,
            X_val=X_val,
            time_limit=time_limit,
            k_fold=k_fold,
            fit_kwargs=fit_kwargs,
            ens_sample_weight=kwargs.get("ens_sample_weight", None),
            label_cleaner=label_cleaner,
        )
        exception = None
        if hyperparameter_tune_kwargs:
            if n_repeat_start != 0:
                raise ValueError(f"n_repeat_start must be 0 to hyperparameter_tune, value = {n_repeat_start}")
            elif k_fold_start != 0:
                raise ValueError(f"k_fold_start must be 0 to hyperparameter_tune, value = {k_fold_start}")
            # hpo_models (dict): keys = model_names, values = model_paths
            fit_log_message = f"Hyperparameter tuning model: {model.name} ..."
            if time_limit is not None:
                if time_limit <= 0:
                    logger.log(15, f"Skipping {model.name} due to lack of time remaining.")
                    return []
                fit_start_time = time.time()
                if self._time_limit is not None and self._time_train_start is not None:
                    time_left_total = self._time_limit - (fit_start_time - self._time_train_start)
                else:
                    time_left_total = time_limit
                fit_log_message += f" Tuning model for up to {round(time_limit, 2)}s of the {round(time_left_total, 2)}s of remaining time."
            logger.log(20, fit_log_message)
            try:
                if isinstance(model, BaggedEnsembleModel):
                    bagged_model_fit_kwargs = self._get_bagged_model_fit_kwargs(
                        k_fold=k_fold,
                        k_fold_start=k_fold_start,
                        k_fold_end=k_fold_end,
                        n_repeats=n_repeats,
                        n_repeat_start=n_repeat_start,
                    )
                    model_fit_kwargs.update(bagged_model_fit_kwargs)
                    hpo_models, hpo_results = model.hyperparameter_tune(
                        X=X,
                        y=y,
                        model=model,
                        X_val=X_val,
                        y_val=y_val,
                        X_unlabeled=X_unlabeled,
                        stack_name=stack_name,
                        level=level,
                        compute_score=compute_score,
                        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        total_resources=total_resources,
                        **model_fit_kwargs,
                    )
                else:
                    hpo_models, hpo_results = model.hyperparameter_tune(
                        X=X,
                        y=y,
                        X_val=X_val,
                        y_val=y_val,
                        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        total_resources=total_resources,
                        **model_fit_kwargs,
                    )
                if len(hpo_models) == 0:
                    logger.warning(
                        f"No model was trained during hyperparameter tuning {model.name}... Skipping this model."
                    )
            except Exception as exc:
                exception = exc  # required to provide exc outside of `except` statement
                if isinstance(exception, NoStackFeatures):
                    logger.warning(f"\tNo stack features to train {model.name}... Skipping this model. {exception}")
                elif isinstance(exception, NotValidStacker):
                    logger.warning(f"\tStacking disabled for {model.name}... Skipping this model. {exception}")
                elif isinstance(exception, NoValidFeatures):
                    logger.warning(f"\tNo valid features to train {model.name}... Skipping this model.")
                else:
                    logger.exception(
                        f"Warning: Exception caused {model.name} to fail during hyperparameter tuning... Skipping this model."
                    )
                    logger.warning(exception)
                del model
                model_names_trained = []
            else:
                # Commented out because it takes too much space (>>5 GB if run for an hour on a small-medium sized dataset)
                # self.hpo_results[model.name] = hpo_results
                model_names_trained = []
                self._extra_banned_names.add(model.name)
                for model_hpo_name, model_info in hpo_models.items():
                    model_hpo = self.load_model(
                        model_hpo_name, path=os.path.relpath(model_info["path"], self.path), model_type=type(model)
                    )
                    logger.log(20, f"Fitted model: {model_hpo.name} ...")
                    if self._add_model(model=model_hpo, stack_name=stack_name, level=level):
                        model_names_trained.append(model_hpo.name)
        else:
            model_fit_kwargs.update(dict(X_pseudo=X_pseudo, y_pseudo=y_pseudo))
            if isinstance(model, BaggedEnsembleModel):
                bagged_model_fit_kwargs = self._get_bagged_model_fit_kwargs(
                    k_fold=k_fold,
                    k_fold_start=k_fold_start,
                    k_fold_end=k_fold_end,
                    n_repeats=n_repeats,
                    n_repeat_start=n_repeat_start,
                )
                model_fit_kwargs.update(bagged_model_fit_kwargs)
            model_names_trained = self._train_and_save(
                X=X,
                y=y,
                model=model,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                X_unlabeled=X_unlabeled,
                stack_name=stack_name,
                level=level,
                compute_score=compute_score,
                total_resources=total_resources,
                errors=errors,
                errors_ignore=errors_ignore,
                errors_raise=errors_raise,
                is_ray_worker=is_ray_worker,
                **model_fit_kwargs,
            )
        if self.callbacks and check_callbacks:
            self._callbacks_after_fit(model_names=model_names_trained, stack_name=stack_name, level=level)
        self.save()
        if exception is not None:
            if self._check_raise_exception(
                exception=exception, errors=errors, errors_ignore=errors_ignore, errors_raise=errors_raise
            ):
                raise exception
        return model_names_trained

    # TODO: Move to a utility function outside of AbstractTabularTrainer
    @staticmethod
    def _check_raise_exception(
        exception: Exception,
        errors: Literal["ignore", "raise"] = "ignore",
        errors_ignore: list | None = None,
        errors_raise: list | None = None,
    ) -> bool:
        """
        Check if an exception should be raised based on the provided error handling logic.

        Parameters
        ----------
        exception: Exception
            The exception to check
        errors: Literal["ignore", "raise"], default = "ignore"
            Determines how exceptions are handled.
            If "ignore", will return False.
            If "raise", will return True.
            Can be overwritten by `errors_ignore` and `errors_raise`.
        errors_ignore: list[str], optional
            The exception types specified in `errors_ignore` will be treated as if `errors="ignore"`.
        errors_raise: list[str], optional
            The exception types specified in `errors_raise` will be treated as if `errors="raise"`.

        Returns
        -------
        raise_exception: bool
            If True, indicates that the exception should be raised based on the provided error handling rules.
        """
        raise_exception = None
        if errors_raise is not None:
            for err_type in errors_raise:
                if isinstance(exception, err_type):
                    raise_exception = True
                    break
        if errors_ignore is not None and raise_exception is None:
            for err_type in errors_ignore:
                if isinstance(exception, err_type):
                    raise_exception = False
                    break
        if raise_exception is None:
            if errors == "ignore":
                raise_exception = False
            elif errors == "raise":
                raise_exception = True
            else:
                raise ValueError(f"Invalid `errors` value: {errors} (valid values: ['ignore', 'raise']")
        return raise_exception

    def _callbacks_before_fit(
        self,
        *,
        model: AbstractModel,
        time_limit: float | None,
        stack_name: str,
        level: int,
    ):
        skip_model = False
        ts = time.time()
        for callback in self.callbacks:
            callback_early_stop, callback_skip_model = callback.before_model_fit(
                trainer=self,
                model=model,
                time_limit=time_limit,
                stack_name=stack_name,
                level=level,
            )
            if callback_early_stop:
                self._callback_early_stop = True
            if callback_skip_model:
                skip_model = True
            if time_limit is not None:
                te = time.time()
                time_limit -= te - ts
                ts = te
        return skip_model, time_limit

    def _callbacks_after_fit(
        self,
        *,
        model_names: list[str],
        stack_name: str,
        level: int,
    ):
        for callback in self.callbacks:
            callback_early_stop = callback.after_model_fit(
                self,
                model_names=model_names,
                stack_name=stack_name,
                level=level,
            )
            if callback_early_stop:
                self._callback_early_stop = True

    # TODO: How to deal with models that fail during this? They have trained valid models before, but should we still use those models or remove the entire model? Currently we still use models.
    # TODO: Time allowance can be made better by only using time taken during final model training and not during HPO and feature pruning.
    # TODO: Time allowance not accurate if running from fit_continue
    # TODO: Remove level and stack_name arguments, can get them automatically
    # TODO: Make sure that pretraining on X_unlabeled only happens 1 time rather than every fold of bagging. (Do during pretrain API work?)
    def _train_multi_repeats(
        self, X, y, models: list, n_repeats, n_repeat_start=1, time_limit=None, time_limit_total_level=None, **kwargs
    ) -> list[str]:
        """
        Fits bagged ensemble models with additional folds and/or bagged repeats.
        Models must have already been fit prior to entering this method.
        This method should only be called in self._train_multi
        Returns a list of successfully trained and saved model names.
        """
        if time_limit_total_level is None:
            time_limit_total_level = time_limit
        models_valid = models
        models_valid_next = []
        repeats_completed = 0
        time_start = time.time()
        for n in range(n_repeat_start, n_repeats):
            if not models_valid:
                break  # No models to repeat
            if time_limit is not None:
                time_start_repeat = time.time()
                time_left = time_limit - (time_start_repeat - time_start)
                if n == n_repeat_start:
                    time_required = time_limit_total_level * 0.575  # Require slightly over 50% to be safe
                else:
                    time_required = (time_start_repeat - time_start) / repeats_completed * (0.575 / 0.425)
                if time_left < time_required:
                    logger.log(15, "Not enough time left to finish repeated k-fold bagging, stopping early ...")
                    break
            logger.log(20, f"Repeating k-fold bagging: {n + 1}/{n_repeats}")
            for i, model in enumerate(models_valid):
                if self._callback_early_stop:
                    break
                if not self.get_model_attribute(model=model, attribute="can_fit"):
                    if isinstance(model, str):
                        models_valid_next.append(model)
                    else:
                        models_valid_next.append(model.name)
                    continue

                if isinstance(model, str):
                    model = self.load_model(model)
                if not isinstance(model, BaggedEnsembleModel):
                    raise AssertionError(
                        f"{model.name} must inherit from BaggedEnsembleModel to perform repeated k-fold bagging. Model type: {type(model).__name__}"
                    )
                if time_limit is None:
                    time_left = None
                else:
                    time_start_model = time.time()
                    time_left = time_limit - (time_start_model - time_start)

                models_valid_next += self._train_single_full(
                    X=X,
                    y=y,
                    model=model,
                    k_fold_start=0,
                    k_fold_end=None,
                    n_repeats=n + 1,
                    n_repeat_start=n,
                    time_limit=time_left,
                    **kwargs,
                )
            models_valid = copy.deepcopy(models_valid_next)
            models_valid_next = []
            repeats_completed += 1
        logger.log(20, f"Completed {n_repeat_start + repeats_completed}/{n_repeats} k-fold bagging repeats ...")
        return models_valid

    def _train_multi_initial(
        self,
        X,
        y,
        models: list[AbstractModel],
        k_fold,
        n_repeats,
        hyperparameter_tune_kwargs=None,
        time_limit=None,
        feature_prune_kwargs=None,
        **kwargs,
    ):
        """
        Fits models that have not previously been fit.
        This method should only be called in self._train_multi
        Returns a list of successfully trained and saved model names.
        """
        multi_fold_time_start = time.time()
        fit_args = dict(
            X=X,
            y=y,
            k_fold=k_fold,
        )
        fit_args.update(kwargs)

        hpo_enabled = False
        if hyperparameter_tune_kwargs:
            for key in hyperparameter_tune_kwargs:
                if hyperparameter_tune_kwargs[key] is not None:
                    hpo_enabled = True
                    break

        hpo_time_ratio = 0.9
        if hpo_enabled:
            time_split = True
        else:
            time_split = False
        k_fold_start = 0
        bagged = k_fold > 0
        if not bagged:
            time_ratio = hpo_time_ratio if hpo_enabled else 1
            models = self._train_multi_fold(
                models=models,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                time_limit=time_limit,
                time_split=time_split,
                time_ratio=time_ratio,
                **fit_args,
            )
        else:
            time_ratio = hpo_time_ratio if hpo_enabled else 1
            models = self._train_multi_fold(
                models=models,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                k_fold_start=0,
                k_fold_end=k_fold,
                n_repeats=n_repeats,
                n_repeat_start=0,
                time_limit=time_limit,
                time_split=time_split,
                time_ratio=time_ratio,
                **fit_args,
            )

        multi_fold_time_elapsed = time.time() - multi_fold_time_start
        if time_limit is not None:
            time_limit = time_limit - multi_fold_time_elapsed

        if feature_prune_kwargs is not None and len(models) > 0:
            feature_prune_time_start = time.time()
            model_fit_kwargs = self._get_model_fit_kwargs(
                X=X,
                X_val=kwargs.get("X_val", None),
                time_limit=None,
                k_fold=k_fold,
                fit_kwargs=kwargs.get("fit_kwargs", {}),
                ens_sample_weight=kwargs.get("ens_sample_weight"),
            )
            model_fit_kwargs.update(dict(X=X, y=y, X_val=kwargs.get("X_val", None), y_val=kwargs.get("y_val", None)))
            if bagged:
                bagged_model_fit_kwargs = self._get_bagged_model_fit_kwargs(
                    k_fold=k_fold, k_fold_start=k_fold_start, k_fold_end=k_fold, n_repeats=n_repeats, n_repeat_start=0
                )
                model_fit_kwargs.update(bagged_model_fit_kwargs)

            # FIXME: v1.3: X.columns incorrectly includes sample_weight column
            # FIXME: v1.3: Move sample_weight logic into fit_stack_core level methods, currently we are editing X too many times in self._get_model_fit_kwargs
            candidate_features = self._proxy_model_feature_prune(
                time_limit=time_limit,
                layer_fit_time=multi_fold_time_elapsed,
                level=kwargs["level"],
                features=X.columns.tolist(),
                model_fit_kwargs=model_fit_kwargs,
                **feature_prune_kwargs,
            )
            if time_limit is not None:
                time_limit = time_limit - (time.time() - feature_prune_time_start)

            fit_args["X"] = X[candidate_features]
            fit_args["X_val"] = (
                kwargs["X_val"][candidate_features]
                if isinstance(kwargs.get("X_val", None), pd.DataFrame)
                else kwargs.get("X_val", None)
            )

            if len(candidate_features) < len(X.columns):
                unfit_models = []
                original_prune_map = {}
                for model in models:
                    unfit_model = self.load_model(model).convert_to_template()
                    unfit_model.rename(f"{unfit_model.name}_Prune")
                    unfit_models.append(unfit_model)
                    original_prune_map[unfit_model.name] = model
                pruned_models = self._train_multi_fold(
                    models=unfit_models,
                    hyperparameter_tune_kwargs=None,
                    k_fold_start=k_fold_start,
                    k_fold_end=k_fold,
                    n_repeats=n_repeats,
                    n_repeat_start=0,
                    time_limit=time_limit,
                    **fit_args,
                )
                force_prune = feature_prune_kwargs.get("force_prune", False)
                models = self._retain_better_pruned_models(
                    pruned_models=pruned_models, original_prune_map=original_prune_map, force_prune=force_prune
                )
        return models

    # TODO: Ban KNN from being a Stacker model outside of aux. Will need to ensemble select on all stack layers ensemble selector to make it work
    # TODO: Robert dataset, LightGBM is super good but RF and KNN take all the time away from it on 1h despite being much worse
    # TODO: Add time_limit_per_model
    # TODO: Rename for v0.1
    def _train_multi_fold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: list[AbstractModel],
        time_limit: float | None = None,
        time_split: bool = False,
        time_ratio: float = 1,
        hyperparameter_tune_kwargs: dict | None = None,
        fit_strategy: Literal["sequential", "parallel"] = "sequential",
        **kwargs,
    ) -> list[str]:
        """
        Trains and saves a list of models sequentially.
        This method should only be called in self._train_multi_initial
        Returns a list of trained model names.
        """
        time_start = time.time()
        if time_limit is not None:
            time_limit = time_limit * time_ratio
        if time_limit is not None and len(models) > 0:
            time_limit_model_split = time_limit / len(models)
        else:
            time_limit_model_split = time_limit

        if fit_strategy == "parallel" and hyperparameter_tune_kwargs is not None and hyperparameter_tune_kwargs:
            for k, v in hyperparameter_tune_kwargs.items():
                if v is not None and (not isinstance(v, dict) or len(v) != 0):
                    logger.log(
                        30,
                        f"WARNING: fit_strategy='parallel', but `hyperparameter_tune_kwargs` is specified for model '{k}' with value {v}. "
                        f"Hyperparameter tuning does not yet support `parallel` fit_strategy. "
                        f"Falling back to fit_strategy='sequential' ... ",
                    )
                    fit_strategy = "sequential"
                    break
        if fit_strategy == "parallel":
            num_cpus = kwargs.get("total_resources", {}).get("num_cpus", "auto")
            if isinstance(num_cpus, str) and num_cpus == "auto":
                num_cpus = get_resource_manager().get_cpu_count()
            if num_cpus < 12:
                force_parallel = os.environ.get("AG_FORCE_PARALLEL", "False") == "True"
                if not force_parallel:
                    logger.log(
                        30,
                        f"Note: fit_strategy='parallel', but `num_cpus={num_cpus}`. "
                        f"Running parallel mode with fewer than 12 CPUs is not recommended and has been disabled. "
                        f'You can override this by specifying `os.environ["AG_FORCE_PARALLEL"] = "True"`. '
                        f"Falling back to fit_strategy='sequential' ...",
                    )
                    fit_strategy = "sequential"
        if fit_strategy == "parallel":
            num_gpus = kwargs.get("total_resources", {}).get("num_gpus", 0)
            if isinstance(num_gpus, str) and num_gpus == "auto":
                num_gpus = get_resource_manager().get_gpu_count()
            if isinstance(num_gpus, (float, int)) and num_gpus > 0:
                logger.log(
                    30,
                    f"WARNING: fit_strategy='parallel', but `num_gpus={num_gpus}` is specified. "
                    f"GPU is not yet supported for `parallel` fit_strategy. To enable parallel, ensure you specify `num_gpus=0` in the fit call. "
                    f"Falling back to fit_strategy='sequential' ... ",
                )
                fit_strategy = "sequential"
        if fit_strategy == "parallel":
            try:
                try_import_ray()
            except Exception as e:
                logger.log(
                    30,
                    f"WARNING: Exception encountered when trying to import ray (fit_strategy='parallel'). "
                    f"ray is required for 'parallel' fit_strategy. Falling back to fit_strategy='sequential' ... "
                    f"\n\tException details: {e.__class__.__name__}: {e}",
                )
                fit_strategy = "sequential"

        if fit_strategy == "sequential":
            models_valid = []
            for model in models:
                if self._callback_early_stop:
                    return models_valid

                models_valid += _detached_train_multi_fold(
                    _self=self,
                    model=model,
                    X=X,
                    y=y,
                    time_start=time_start,
                    time_split=time_split,
                    time_limit=time_limit,
                    time_limit_model_split=time_limit_model_split,
                    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                    is_ray_worker=False,
                    kwargs=kwargs,
                )
        elif fit_strategy == "parallel":
            models_valid = self._train_multi_fold_parallel(
                X=X,
                y=y,
                models=models,
                time_start=time_start,
                time_limit_model_split=time_limit_model_split,
                time_limit=time_limit,
                time_split=time_split,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid value for fit_strategy: '{fit_strategy}'")
        return models_valid

    def _train_multi_fold_parallel(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: list[AbstractModel],
        time_start: float,
        time_limit_model_split: float | None,
        time_limit: float | None = None,
        time_split: bool = False,
        hyperparameter_tune_kwargs: dict | None = None,
        **kwargs,
    ) -> list[str]:
        # -- Parallel or Distributed training
        ray = try_import_ray()

        # FIXME: Need a common utility class for initializing ray so we don't duplicate code
        if not ray.is_initialized():
            ray.init(log_to_driver=False, logging_level=logging.ERROR)

        models_valid = []

        if time_limit is not None:
            # Give models less than the full time limit to account for overheads (predict, cache, ray, etc.)
            time_limit_models = time_limit * 0.9
        else:
            time_limit_models = None

        logger.log(20, "Scheduling parallel model-workers for training...")
        distributed_manager = ParallelFitManager(
            mode="fit",
            X=X,  # FIXME: REMOVE
            y=y,  # FIXME: REMOVE
            func=_remote_train_multi_fold,
            func_kwargs=dict(
                time_split=time_split,
                time_limit_model_split=time_limit_model_split,
                time_limit=time_limit_models,
                time_start=time_start,
                errors="raise",
            ),
            func_put_kwargs=dict(
                _self=self,
                X=X,
                y=y,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                kwargs=kwargs,
            ),
            num_cpus=kwargs.get("total_resources", {}).get("num_cpus", 1),
            num_gpus=kwargs.get("total_resources", {}).get("num_gpus", 0),
            num_splits=kwargs.get("k_fold", 1) * kwargs.get("n_repeats", 1),
            problem_type=self.problem_type,  # FIXME: Should this be passed here?
            num_classes=self.num_classes,  # FIXME: Should this be passed here?
        )
        jobs_finished = 0
        jobs_total = len(models)

        ordered_model_names = [m.name for m in models]  # Use to ensure same model order is returned
        expected_model_names = set(ordered_model_names)
        unfinished_job_refs = distributed_manager.schedule_jobs(models_to_fit=models)

        timeout = None

        if time_limit is not None:
            # allow between 5 and 60 seconds overhead before force killing jobs to give some leniency to jobs with overhead.
            time_overhead = min(max(time_limit * 0.01, 5), 60)
            min_time_required_base = min(
                self._time_limit * 0.01, 10
            )  # This is checked in the worker thread, will skip if not satisfied
            # If time remaining is less than min_time_required, avoid scheduling new jobs and only wait for existing ones to finish.
            min_time_required = (
                min_time_required_base * 1.5 + 1
            )  # Add 50% buffer and 1 second to account for ray overhead
        else:
            time_overhead = None
            min_time_required = None

        can_schedule_jobs = True
        while unfinished_job_refs:
            if time_limit is not None:
                time_left = time_limit - (time.time() - time_start)
                timeout = int(time_left + time_overhead)  # include overhead.
                if timeout <= 0:
                    logger.log(20, "Ran into timeout while waiting for model training to finish. Stopping now.")
                    break
            finished, unfinished_job_refs = ray.wait(unfinished_job_refs, num_returns=1, timeout=timeout)

            if not finished:
                logger.log(20, "Ran into timeout while waiting for model training to finish. Stopping now.")
                break

            distributed_manager.deallocate_resources(job_ref=finished[0])
            model_name, model_path, model_type, exc, model_failure_info = ray.get(finished[0])
            assert model_name in expected_model_names, (
                f"Unexpected model name outputted during parallel fit: {model_name}\n"
                f"Valid Names: {expected_model_names}\n"
                f"This should never happen. Please create a GitHub Issue."
            )
            jobs_finished += 1

            if exc is not None or model_path is None:
                if exc is None:
                    if model_failure_info is not None:
                        exc_type = model_failure_info["exc_type"]
                        exc_str = model_failure_info["exc_str"]
                    else:
                        exc_type = None
                        exc_str = None
                else:
                    exc_type = exc.__class__
                    exc_str = str(exc)
                if exc_type is not None:
                    extra_log = f": {exc_type.__name__}: {exc_str}"
                else:
                    extra_log = ""
                if exc_type is not None and issubclass(exc_type, InsufficientTime):
                    logger.log(20, exc_str)
                else:
                    logger.log(
                        20,
                        f"Skipping {model_name if isinstance(model_name, str) else model_name.name} due to exception{extra_log}",
                    )
                if model_failure_info is not None:
                    self._models_failed_to_train_errors[model_name] = model_failure_info
            else:
                logger.log(20, f"Fitted {model_name}:")

                # TODO: figure out a way to avoid calling _add_model in the worker-process to save overhead time.
                #       - Right now, we need to call it within _add_model to be able to pass the model path to the main process without changing
                #         the return signature of _train_single_full. This can be a lot of work to change.
                # TODO: determine if y_pred_proba_val was cached in the worker-process. Right now, we re-do predictions for holdout data.
                # Self object is not permanently mutated during worker execution, so we need to add model to the "main" self (again).
                # This is the synchronization point between the distributed and main processes.
                if self._add_model(
                    model_type.load(path=os.path.join(self.path, model_path), reset_paths=self.reset_paths),
                    stack_name=kwargs["stack_name"],
                    level=kwargs["level"],
                ):
                    jobs_running = len(unfinished_job_refs)
                    if can_schedule_jobs:
                        remaining_task_word = "pending"
                    else:
                        remaining_task_word = "skipped"
                    parallel_status_log = (
                        f"\tJobs: {jobs_running} running, "
                        f"{jobs_total - (jobs_finished + jobs_running)} {remaining_task_word}, "
                        f"{jobs_finished}/{jobs_total} finished"
                    )
                    if time_limit is not None:
                        time_left = time_limit - (time.time() - time_start)
                        parallel_status_log += f" | {time_left:.0f}s remaining"
                    logger.log(20, parallel_status_log)
                    models_valid.append(model_name)
                else:
                    logger.log(
                        40,
                        f"Failed to add {model_name} to model graph. This should never happen. Please create a GitHub issue.",
                    )

            if not unfinished_job_refs and not distributed_manager.models_to_schedule:
                # Completed all jobs
                break

            # TODO: look into what this does / how this works for distributed training
            if self._callback_early_stop:
                logger.log(
                    20,
                    "Callback triggered in parallel setting. Stopping model training and cancelling remaining jobs.",
                )
                break

            # Stop due to time limit after adding model
            if time_limit is not None:
                time_elapsed = time.time() - time_start
                time_left = time_limit - time_elapsed
                time_left_models = time_limit_models - time_elapsed
                if (time_left + time_overhead) <= 0:
                    logger.log(
                        20,
                        "Time limit reached for this stacking layer. Stopping model training and cancelling remaining jobs.",
                    )
                    break
                elif time_left_models < min_time_required:
                    if can_schedule_jobs:
                        if len(distributed_manager.models_to_schedule) > 0:
                            logger.log(
                                20,
                                f"Low on time, skipping {len(distributed_manager.models_to_schedule)} "
                                f"pending jobs and waiting for running jobs to finish... ({time_left:.0f}s remaining time)",
                            )
                        can_schedule_jobs = False

            if can_schedule_jobs:
                # Re-schedule jobs
                unfinished_job_refs += distributed_manager.schedule_jobs()

        distributed_manager.clean_up_ray(unfinished_job_refs=unfinished_job_refs)
        logger.log(20, "Finished all parallel work for this stacking layer.")

        models_valid = set(models_valid)
        models_valid = [m for m in ordered_model_names if m in models_valid]  # maintain original order

        return models_valid

    def _train_multi(
        self,
        X,
        y,
        models: list[AbstractModel],
        hyperparameter_tune_kwargs=None,
        feature_prune_kwargs=None,
        k_fold=None,
        n_repeats=None,
        n_repeat_start=0,
        time_limit=None,
        delay_bag_sets: bool = False,
        **kwargs,
    ) -> list[str]:
        """
        Train a list of models using the same data.
        Assumes that input data has already been processed in the form the models will receive as input (including stack feature generation).
        Trained models are available in the trainer object.
        Note: Consider using public APIs instead of this.
        Returns a list of trained model names.
        """
        time_limit_total_level = time_limit
        if k_fold is None:
            k_fold = self.k_fold
        if n_repeats is None:
            n_repeats = self.n_repeats
        if (k_fold == 0) and (n_repeats != 1):
            raise ValueError(f"n_repeats must be 1 when k_fold is 0, values: ({n_repeats}, {k_fold})")
        if (time_limit is None and feature_prune_kwargs is None) or (not delay_bag_sets):
            n_repeats_initial = n_repeats
        else:
            n_repeats_initial = 1
        if n_repeat_start == 0:
            time_start = time.time()
            model_names_trained = self._train_multi_initial(
                X=X,
                y=y,
                models=models,
                k_fold=k_fold,
                n_repeats=n_repeats_initial,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                feature_prune_kwargs=feature_prune_kwargs,
                time_limit=time_limit,
                **kwargs,
            )
            n_repeat_start = n_repeats_initial
            if time_limit is not None:
                time_limit = time_limit - (time.time() - time_start)
        else:
            model_names_trained = models
        if (n_repeats > 1) and (n_repeat_start < n_repeats):
            model_names_trained = self._train_multi_repeats(
                X=X,
                y=y,
                models=model_names_trained,
                k_fold=k_fold,
                n_repeats=n_repeats,
                n_repeat_start=n_repeat_start,
                time_limit=time_limit,
                time_limit_total_level=time_limit_total_level,
                **kwargs,
            )
        return model_names_trained

    def _train_multi_and_ensemble(
        self,
        X,
        y,
        X_val,
        y_val,
        X_test=None,
        y_test=None,
        hyperparameters: dict | None = None,
        X_unlabeled=None,
        num_stack_levels=0,
        time_limit=None,
        groups=None,
        **kwargs,
    ) -> list[str]:
        """Identical to self.train_multi_levels, but also saves the data to disk. This should only ever be called once."""
        if time_limit is not None and time_limit <= 0:
            raise AssertionError(
                f"Not enough time left to train models. Consider specifying a larger time_limit. Time remaining: {round(time_limit, 2)}s"
            )
        if self.save_data and not self.is_data_saved:
            self.save_X(X)
            self.save_y(y)
            if X_val is not None:
                self.save_X_val(X_val)
                if y_val is not None:
                    self.save_y_val(y_val)
            if X_test is not None:
                self.save_X_test(X_test)
                if y_test is not None:
                    self.save_y_test(y_test)
            self.is_data_saved = True
        if self._groups is None:
            self._groups = groups
        self._num_rows_train = len(X)
        if X_val is not None:
            self._num_rows_val = len(X_val)
        if X_test is not None:
            self._num_rows_test = len(X_test)
        self._num_cols_train = len(list(X.columns))
        model_names_fit = self.train_multi_levels(
            X,
            y,
            hyperparameters=hyperparameters,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_unlabeled=X_unlabeled,
            level_start=1,
            level_end=num_stack_levels + 1,
            time_limit=time_limit,
            **kwargs,
        )
        if len(self.get_model_names()) == 0:
            # TODO v1.0: Add toggle to raise exception if no models trained
            logger.log(30, "Warning: AutoGluon did not successfully train any models")
        return model_names_fit

    def _predict_model(self, X: pd.DataFrame, model: str, model_pred_proba_dict: dict | None = None) -> np.ndarray:
        y_pred_proba = self._predict_proba_model(X=X, model=model, model_pred_proba_dict=model_pred_proba_dict)
        return get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)

    def _predict_proba_model(
        self, X: pd.DataFrame, model: str, model_pred_proba_dict: dict | None = None
    ) -> np.ndarray:
        model_pred_proba_dict = self.get_model_pred_proba_dict(
            X=X, models=[model], model_pred_proba_dict=model_pred_proba_dict
        )
        if not isinstance(model, str):
            model = model.name
        return model_pred_proba_dict[model]

    def _proxy_model_feature_prune(
        self,
        model_fit_kwargs: dict,
        time_limit: float,
        layer_fit_time: float,
        level: int,
        features: list[str],
        **feature_prune_kwargs: dict,
    ) -> list[str]:
        """
        Uses the best LightGBM-based base learner of this layer to perform time-aware permutation feature importance based feature pruning.
        If all LightGBM models fail, use the model that achieved the highest validation accuracy. Feature pruning gets the smaller of the
        remaining layer time limit and k times (default=2) it took to fit the base learners of this layer as its resource. Note that feature pruning can
        exit earlier based on arguments in feature_prune_kwargs. The method returns the list of feature names that survived the pruning procedure.

        Parameters
        ----------
        feature_prune_kwargs : dict
            Feature pruning kwarg arguments. Should contain arguments passed to FeatureSelector.select_features. One can optionally attach the following
            additional kwargs that are consumed at this level: 'proxy_model_class' to use a model of particular type with the highest validation score as the
            proxy model, 'feature_prune_time_limit' to manually specify how long we should perform the feature pruning procedure for, 'k' to specify how long
            we should perform feature pruning for if 'feature_prune_time_limit' has not been set (feature selection time budget is set to k * layer_fit_time),
            and 'raise_exception' to signify that AutoGluon should throw an exception if feature pruning errors out.
        time_limit : float
            Time limit left within the current stack layer in seconds. Feature pruning should never take more than this time.
        layer_fit_time : float
            How long it took to fit all the models in this layer once. Used to calculate how long to feature prune for.
        level : int
            Level of this stack layer.
        features: list[str]
            The list of feature names in the inputted dataset.

        Returns
        -------
        candidate_features : list[str]
            Feature names that survived the pruning procedure.
        """
        k = feature_prune_kwargs.pop("k", 2)
        proxy_model_class = feature_prune_kwargs.pop("proxy_model_class", self._get_default_proxy_model_class())
        feature_prune_time_limit = feature_prune_kwargs.pop("feature_prune_time_limit", None)
        raise_exception_on_fail = feature_prune_kwargs.pop("raise_exception", False)

        proxy_model = self._get_feature_prune_proxy_model(proxy_model_class=proxy_model_class, level=level)
        if proxy_model is None:
            return features

        if feature_prune_time_limit is not None:
            feature_prune_time_limit = min(max(time_limit - layer_fit_time, 0), feature_prune_time_limit)
        elif time_limit is not None:
            feature_prune_time_limit = min(
                max(time_limit - layer_fit_time, 0), max(k * layer_fit_time, 0.05 * time_limit)
            )
        else:
            feature_prune_time_limit = max(k * layer_fit_time, 300)

        if feature_prune_time_limit < 2 * proxy_model.fit_time:
            logger.warning(
                f"Insufficient time to train even a single feature pruning model (remaining: {feature_prune_time_limit}, "
                f"needed: {proxy_model.fit_time}). Skipping feature pruning."
            )
            return features
        selector = FeatureSelector(
            model=proxy_model,
            time_limit=feature_prune_time_limit,
            raise_exception=raise_exception_on_fail,
            problem_type=self.problem_type,
        )
        candidate_features = selector.select_features(**feature_prune_kwargs, **model_fit_kwargs)
        return candidate_features

    def _get_default_proxy_model_class(self):
        return None

    def _retain_better_pruned_models(
        self, pruned_models: list[str], original_prune_map: dict, force_prune: bool = False
    ) -> list[str]:
        """
        Compares models fit on the pruned set of features with their counterpart, models fit on full set of features.
        Take the model that achieved a higher validation set score and delete the other from self.model_graph.

        Parameters
        ----------
        pruned_models : list[str]
            A list of pruned model names.
        original_prune_map : dict
            A dictionary mapping the names of models fitted on pruned features to the names of models fitted on original features.
        force_prune : bool, default = False
            If set to true, force all base learners to work with the pruned set of features.

        Returns
        ----------
        models : list[str]
            A list of model names.
        """
        models = []
        for pruned_model in pruned_models:
            original_model = original_prune_map[pruned_model]
            leaderboard = self.leaderboard()
            original_score = leaderboard[leaderboard["model"] == original_model]["score_val"].item()
            pruned_score = leaderboard[leaderboard["model"] == pruned_model]["score_val"].item()
            score_str = f"({round(pruned_score, 4)} vs {round(original_score, 4)})"
            if force_prune:
                logger.log(
                    30,
                    f"Pruned score vs original score is {score_str}. Replacing original model since force_prune=True...",
                )
                self.delete_models(models_to_delete=original_model, dry_run=False)
                models.append(pruned_model)
            elif pruned_score > original_score:
                logger.log(
                    30,
                    f"Model trained with feature pruning score is better than original model's score {score_str}. Replacing original model...",
                )
                self.delete_models(models_to_delete=original_model, dry_run=False)
                models.append(pruned_model)
            else:
                logger.log(
                    30,
                    f"Model trained with feature pruning score is not better than original model's score {score_str}. Keeping original model...",
                )
                self.delete_models(models_to_delete=pruned_model, dry_run=False)
                models.append(original_model)
        return models

    # TODO: Enable raw=True for bagged models when X=None
    #  This is non-trivial to implement for multi-layer stacking ensembles on the OOF data.
    # TODO: Consider limiting X to 10k rows here instead of inside the model call
    def get_feature_importance(self, model=None, X=None, y=None, raw=True, **kwargs) -> pd.DataFrame:
        if model is None:
            model = self.model_best
        model: AbstractModel = self.load_model(model)
        if X is None and model.val_score is None:
            raise AssertionError(
                f"Model {model.name} is not valid for generating feature importances on original training data because no validation data was used during training, please specify new test data to compute feature importances."
            )

        if X is None:
            if isinstance(model, WeightedEnsembleModel):
                if self.bagged_mode:
                    if raw:
                        raise AssertionError(
                            "`feature_stage='transformed'` feature importance on the original training data is not yet supported when bagging is enabled, please specify new test data to compute feature importances."
                        )
                    X = None
                    is_oof = True
                else:
                    if raw:
                        X = self.load_X_val()
                    else:
                        X = None
                    is_oof = False
            elif isinstance(model, BaggedEnsembleModel):
                if raw:
                    raise AssertionError(
                        "`feature_stage='transformed'` feature importance on the original training data is not yet supported when bagging is enabled, please specify new test data to compute feature importances."
                    )
                X = self.load_X()
                X = self.get_inputs_to_model(model=model, X=X, fit=True)
                is_oof = True
            else:
                X = self.load_X_val()
                if not raw:
                    X = self.get_inputs_to_model(model=model, X=X, fit=False)
                is_oof = False
        else:
            is_oof = False
            if not raw:
                X = self.get_inputs_to_model(model=model, X=X, fit=False)

        if y is None and X is not None:
            if is_oof:
                y = self.load_y()
            else:
                y = self.load_y_val()

        if raw:
            return self._get_feature_importance_raw(X=X, y=y, model=model, **kwargs)
        else:
            if is_oof:
                kwargs["is_oof"] = is_oof
            return model.compute_feature_importance(X=X, y=y, **kwargs)

    # TODO: Can get feature importances of all children of model at no extra cost, requires scoring the values after predict_proba on each model
    #  Could solve by adding a self.score_all() function which takes model as input and also returns scores of all children models.
    #  This would be best solved after adding graph representation, it lives most naturally in AbstractModel
    # TODO: Can skip features which were pruned on all models that model depends on (Complex to implement, requires graph representation)
    # TODO: Note that raw importance will not equal non-raw importance for bagged models, even if raw features are identical to the model features.
    #  This is because for non-raw, we do an optimization where each fold model calls .compute_feature_importance(), and then the feature importances are averaged across the folds.
    #  This is different from raw, where the predictions of the folds are averaged and then feature importance is computed.
    #  Consider aligning these methods so they produce the same result.
    # The output of this function is identical to non-raw when model is level 1 and non-bagged
    def _get_feature_importance_raw(self, X, y, model, eval_metric=None, **kwargs) -> pd.DataFrame:
        if eval_metric is None:
            eval_metric = self.eval_metric
        if model is None:
            model = self._get_best()
        if eval_metric.needs_pred:
            predict_func = self.predict
        else:
            predict_func = self.predict_proba
        model: AbstractModel = self.load_model(model)
        predict_func_kwargs = dict(model=model)
        return compute_permutation_feature_importance(
            X=X,
            y=y,
            predict_func=predict_func,
            predict_func_kwargs=predict_func_kwargs,
            eval_metric=eval_metric,
            quantile_levels=self.quantile_levels,
            **kwargs,
        )

    def _get_models_load_info(self, model_names):
        model_names = copy.deepcopy(model_names)
        model_paths = self.get_models_attribute_dict(attribute="path", models=model_names)
        model_types = self.get_models_attribute_dict(attribute="type", models=model_names)
        return model_names, model_paths, model_types

    def get_model_attribute_full(self, model: str | list[str], attribute: str, func=sum) -> float | int:
        """
        Sums the attribute value across all models that the provided model depends on, including itself.
        For instance, this function can return the expected total predict_time of a model.
        attribute is the name of the desired attribute to be summed,
        or a dictionary of model name -> attribute value if the attribute is not present in the graph.
        """
        if isinstance(model, list):
            base_model_set = self.get_minimum_models_set(model)
        else:
            base_model_set = self.get_minimum_model_set(model)
        if isinstance(attribute, dict):
            is_dict = True
        else:
            is_dict = False
        if len(base_model_set) == 1:
            if is_dict:
                return attribute[model]
            else:
                return self.model_graph.nodes[base_model_set[0]][attribute]
        # attribute_full = 0
        attribute_lst = []
        for base_model in base_model_set:
            if is_dict:
                attribute_base_model = attribute[base_model]
            else:
                attribute_base_model = self.model_graph.nodes[base_model][attribute]
            if attribute_base_model is None:
                return None
            attribute_lst.append(attribute_base_model)
            # attribute_full += attribute_base_model
        if attribute_lst:
            attribute_full = func(attribute_lst)
        else:
            attribute_full = 0
        return attribute_full

    def get_models_attribute_full(self, models: list[str], attribute: str, func=sum):
        """
        For each model in models, returns the output of self.get_model_attribute_full mapped to a dict.
        """
        d = dict()
        for model in models:
            d[model] = self.get_model_attribute_full(model=model, attribute=attribute, func=func)
        return d

    # Gets the minimum set of models that the provided models depend on, including themselves
    # Returns a list of model names
    def get_minimum_models_set(self, models: list) -> list:
        models_set = set()
        for model in models:
            models_set = models_set.union(self.get_minimum_model_set(model))
        return list(models_set)

    # Gets the set of base models used directly by the provided model
    # Returns a list of model names
    def get_base_model_names(self, model) -> list:
        if not isinstance(model, str):
            model = model.name
        base_model_set = list(self.model_graph.predecessors(model))
        return base_model_set

    def model_refit_map(self, inverse=False) -> dict[str, str]:
        """
        Returns dict of parent model -> refit model

        If inverse=True, return dict of refit model -> parent model
        """
        model_refit_map = self.get_models_attribute_dict(attribute="refit_full_parent")
        if not inverse:
            model_refit_map = {parent: refit for refit, parent in model_refit_map.items()}
        return model_refit_map

    def model_exists(self, model: str) -> bool:
        return model in self.get_model_names()

    def _flatten_model_info(self, model_info: dict) -> dict:
        """
        Flattens the model_info nested dictionary into a shallow dictionary to convert to a pandas DataFrame row.

        Parameters
        ----------
        model_info: dict
            A nested dictionary of model metadata information

        Returns
        -------
        A flattened dictionary of model info.
        """
        model_info_keys = [
            "num_features",
            "model_type",
            "hyperparameters",
            "hyperparameters_fit",
            "ag_args_fit",
            "features",
            "is_initialized",
            "is_fit",
            "is_valid",
            "can_infer",
        ]
        model_info_flat = {k: v for k, v in model_info.items() if k in model_info_keys}

        custom_info = {}
        bagged_info = model_info.get("bagged_info", {})
        custom_info["num_models"] = bagged_info.get("num_child_models", 1)
        custom_info["memory_size"] = bagged_info.get("max_memory_size", model_info["memory_size"])
        custom_info["memory_size_min"] = bagged_info.get("min_memory_size", model_info["memory_size"])
        custom_info["compile_time"] = bagged_info.get("compile_time", model_info["compile_time"])
        custom_info["child_model_type"] = bagged_info.get("child_model_type", None)
        custom_info["child_hyperparameters"] = bagged_info.get("child_hyperparameters", None)
        custom_info["child_hyperparameters_fit"] = bagged_info.get("child_hyperparameters_fit", None)
        custom_info["child_ag_args_fit"] = bagged_info.get("child_ag_args_fit", None)

        model_info_keys = [
            "num_models",
            "memory_size",
            "memory_size_min",
            "compile_time",
            "child_model_type",
            "child_hyperparameters",
            "child_hyperparameters_fit",
            "child_ag_args_fit",
        ]
        for key in model_info_keys:
            model_info_flat[key] = custom_info[key]
        return model_info_flat

    def leaderboard(self, extra_info=False, refit_full: bool | None = None, set_refit_score_to_parent: bool = False):
        model_names = self.get_model_names()
        models_full_dict = self.get_models_attribute_dict(models=model_names, attribute="refit_full_parent")
        if refit_full is not None:
            if refit_full:
                model_names = [model for model in model_names if model in models_full_dict]
            else:
                model_names = [model for model in model_names if model not in models_full_dict]
        score_val = []
        eval_metric = []
        stopping_metric = []
        fit_time_marginal = []
        pred_time_val_marginal = []
        stack_level = []
        fit_time = []
        pred_time_val = []
        can_infer = []
        fit_order = list(range(1, len(model_names) + 1))
        score_val_dict = self.get_models_attribute_dict("val_score")
        eval_metric_dict = self.get_models_attribute_dict("eval_metric")
        stopping_metric_dict = self.get_models_attribute_dict("stopping_metric")
        fit_time_marginal_dict = self.get_models_attribute_dict("fit_time")
        predict_time_marginal_dict = self.get_models_attribute_dict("predict_time")
        fit_time_dict = self.get_models_attribute_full(attribute="fit_time", models=model_names, func=sum)
        pred_time_val_dict = self.get_models_attribute_full(attribute="predict_time", models=model_names, func=sum)
        can_infer_dict = self.get_models_attribute_full(attribute="can_infer", models=model_names, func=min)
        for model_name in model_names:
            if set_refit_score_to_parent and (model_name in models_full_dict):
                if models_full_dict[model_name] not in score_val_dict:
                    raise AssertionError(
                        f"Model parent is missing from leaderboard when `set_refit_score_to_parent=True`, "
                        f"this is invalid. The parent model may have been deleted. "
                        f"(model='{model_name}', parent='{models_full_dict[model_name]}')"
                    )
                score_val.append(score_val_dict[models_full_dict[model_name]])
            else:
                score_val.append(score_val_dict[model_name])
            eval_metric.append(eval_metric_dict[model_name])
            stopping_metric.append(stopping_metric_dict[model_name])
            fit_time_marginal.append(fit_time_marginal_dict[model_name])
            fit_time.append(fit_time_dict[model_name])
            pred_time_val_marginal.append(predict_time_marginal_dict[model_name])
            pred_time_val.append(pred_time_val_dict[model_name])
            stack_level.append(self.get_model_level(model_name))
            can_infer.append(can_infer_dict[model_name])

        model_info_dict = defaultdict(list)
        extra_info_dict = dict()
        if extra_info:
            # TODO: feature_metadata
            # TODO: disk size
            # TODO: load time
            # TODO: Add persist_if_mem_safe() function to persist in memory all models if reasonable memory size (or a specific model+ancestors)
            # TODO: Add is_persisted() function to check which models are persisted in memory
            # TODO: package_dependencies, package_dependencies_full

            info = self.get_info(include_model_info=True)
            model_info = info["model_info"]
            custom_model_info = {}
            for model_name in model_info:
                custom_info = {}
                bagged_info = model_info[model_name].get("bagged_info", {})
                custom_info["num_models"] = bagged_info.get("num_child_models", 1)
                custom_info["memory_size"] = bagged_info.get("max_memory_size", model_info[model_name]["memory_size"])
                custom_info["memory_size_min"] = bagged_info.get(
                    "min_memory_size", model_info[model_name]["memory_size"]
                )
                custom_info["compile_time"] = bagged_info.get("compile_time", model_info[model_name]["compile_time"])
                custom_info["child_model_type"] = bagged_info.get("child_model_type", None)
                custom_info["child_hyperparameters"] = bagged_info.get("child_hyperparameters", None)
                custom_info["child_hyperparameters_fit"] = bagged_info.get("child_hyperparameters_fit", None)
                custom_info["child_ag_args_fit"] = bagged_info.get("child_ag_args_fit", None)
                custom_model_info[model_name] = custom_info

            model_info_keys = [
                "num_features",
                "model_type",
                "hyperparameters",
                "hyperparameters_fit",
                "ag_args_fit",
                "features",
            ]
            model_info_sum_keys = []
            for key in model_info_keys:
                model_info_dict[key] = [model_info[model_name][key] for model_name in model_names]
                if key in model_info_sum_keys:
                    key_dict = {model_name: model_info[model_name][key] for model_name in model_names}
                    model_info_dict[key + "_full"] = [
                        self.get_model_attribute_full(model=model_name, attribute=key_dict)
                        for model_name in model_names
                    ]

            model_info_keys = [
                "num_models",
                "memory_size",
                "memory_size_min",
                "compile_time",
                "child_model_type",
                "child_hyperparameters",
                "child_hyperparameters_fit",
                "child_ag_args_fit",
            ]
            model_info_full_keys = {
                "memory_size": [("memory_size_w_ancestors", sum)],
                "memory_size_min": [("memory_size_min_w_ancestors", max)],
                "num_models": [("num_models_w_ancestors", sum)],
            }
            for key in model_info_keys:
                model_info_dict[key] = [custom_model_info[model_name][key] for model_name in model_names]
                if key in model_info_full_keys:
                    key_dict = {model_name: custom_model_info[model_name][key] for model_name in model_names}
                    for column_name, func in model_info_full_keys[key]:
                        model_info_dict[column_name] = [
                            self.get_model_attribute_full(model=model_name, attribute=key_dict, func=func)
                            for model_name in model_names
                        ]

            ancestors = [list(nx.dag.ancestors(self.model_graph, model_name)) for model_name in model_names]
            descendants = [list(nx.dag.descendants(self.model_graph, model_name)) for model_name in model_names]

            model_info_dict["num_ancestors"] = [len(ancestor_lst) for ancestor_lst in ancestors]
            model_info_dict["num_descendants"] = [len(descendant_lst) for descendant_lst in descendants]
            model_info_dict["ancestors"] = ancestors
            model_info_dict["descendants"] = descendants

            extra_info_dict = {
                "stopping_metric": stopping_metric,
            }

        df = pd.DataFrame(
            data={
                "model": model_names,
                "score_val": score_val,
                "eval_metric": eval_metric,
                "pred_time_val": pred_time_val,
                "fit_time": fit_time,
                "pred_time_val_marginal": pred_time_val_marginal,
                "fit_time_marginal": fit_time_marginal,
                "stack_level": stack_level,
                "can_infer": can_infer,
                "fit_order": fit_order,
                **extra_info_dict,
                **model_info_dict,
            }
        )
        df_sorted = df.sort_values(
            by=["score_val", "pred_time_val", "model"], ascending=[False, True, False]
        ).reset_index(drop=True)

        df_columns_lst = df_sorted.columns.tolist()
        explicit_order = [
            "model",
            "score_val",
            "eval_metric",
            "pred_time_val",
            "fit_time",
            "pred_time_val_marginal",
            "fit_time_marginal",
            "stack_level",
            "can_infer",
            "fit_order",
            "num_features",
            "num_models",
            "num_models_w_ancestors",
            "memory_size",
            "memory_size_w_ancestors",
            "memory_size_min",
            "memory_size_min_w_ancestors",
            "num_ancestors",
            "num_descendants",
            "model_type",
            "child_model_type",
        ]
        explicit_order = [column for column in explicit_order if column in df_columns_lst]
        df_columns_other = [column for column in df_columns_lst if column not in explicit_order]
        df_columns_new = explicit_order + df_columns_other
        df_sorted = df_sorted[df_columns_new]

        return df_sorted

    def model_failures(self) -> pd.DataFrame:
        """
        [Advanced] Get the model failures that occurred during the fitting of this predictor, in the form of a pandas DataFrame.

        This is useful for in-depth debugging of model failures and identifying bugs.

        Returns
        -------
        model_failures_df: pd.DataFrame
            A DataFrame of model failures. Each row corresponds to a model failure, and columns correspond to meta information about that model.
        """
        model_infos = dict()
        for i, (model_name, model_info) in enumerate(self._models_failed_to_train_errors.items()):
            model_info = copy.deepcopy(model_info)
            model_info_inner = model_info["model_info"]

            model_info_inner = self._flatten_model_info(model_info_inner)

            valid_keys = [
                "exc_type",
                "exc_str",
                "exc_traceback",
                "total_time",
            ]
            valid_keys_inner = [
                "model_type",
                "hyperparameters",
                "hyperparameters_fit",
                "is_initialized",
                "is_fit",
                "is_valid",
                "can_infer",
                "num_features",
                "memory_size",
                "num_models",
                "child_model_type",
                "child_hyperparameters",
                "child_hyperparameters_fit",
            ]
            model_info_out = {k: v for k, v in model_info.items() if k in valid_keys}
            model_info_inner_out = {k: v for k, v in model_info_inner.items() if k in valid_keys_inner}

            model_info_out.update(model_info_inner_out)
            model_info_out["model"] = model_name
            model_info_out["exc_order"] = i + 1

            model_infos[model_name] = model_info_out

        df = pd.DataFrame(
            data=model_infos,
        ).T

        explicit_order = [
            "model",
            "exc_type",
            "total_time",
            "model_type",
            "child_model_type",
            "is_initialized",
            "is_fit",
            "is_valid",
            "can_infer",
            "num_features",
            "num_models",
            "memory_size",
            "hyperparameters",
            "hyperparameters_fit",
            "child_hyperparameters",
            "child_hyperparameters_fit",
            "exc_str",
            "exc_traceback",
            "exc_order",
        ]

        df_columns_lst = list(df.columns)
        explicit_order = [column for column in explicit_order if column in df_columns_lst]
        df_columns_other = [column for column in df_columns_lst if column not in explicit_order]
        df_columns_new = explicit_order + df_columns_other
        df_sorted = df[df_columns_new]
        df_sorted = df_sorted.reset_index(drop=True)

        return df_sorted

    def get_info(self, include_model_info=False, include_model_failures=True) -> dict:
        num_models_trained = len(self.get_model_names())
        if self.model_best is not None:
            best_model = self.model_best
        else:
            try:
                best_model = self.get_model_best()
            except AssertionError:
                best_model = None
        if best_model is not None:
            best_model_score_val = self.get_model_attribute(model=best_model, attribute="val_score")
            best_model_stack_level = self.get_model_level(best_model)
        else:
            best_model_score_val = None
            best_model_stack_level = None
        # fit_time = None
        num_bag_folds = self.k_fold
        max_core_stack_level = self.get_max_level("core")
        max_stack_level = self.get_max_level()

        problem_type = self.problem_type
        eval_metric = self.eval_metric.name
        time_train_start = self._time_train_start_last
        num_rows_train = self._num_rows_train
        num_cols_train = self._num_cols_train
        num_rows_val = self._num_rows_val
        num_rows_test = self._num_rows_test
        num_classes = self.num_classes
        # TODO:
        #  Disk size of models
        #  Raw feature count
        #  HPO time
        #  Bag time
        #  Feature prune time
        #  Exception count / models failed count
        #  True model count (models * kfold)
        #  AutoGluon version fit on
        #  Max memory usage
        #  CPU count used / GPU count used

        info = {
            "time_train_start": time_train_start,
            "num_rows_train": num_rows_train,
            "num_cols_train": num_cols_train,
            "num_rows_val": num_rows_val,
            "num_rows_test": num_rows_test,
            "num_classes": num_classes,
            "problem_type": problem_type,
            "eval_metric": eval_metric,
            "best_model": best_model,
            "best_model_score_val": best_model_score_val,
            "best_model_stack_level": best_model_stack_level,
            "num_models_trained": num_models_trained,
            "num_bag_folds": num_bag_folds,
            "max_stack_level": max_stack_level,
            "max_core_stack_level": max_core_stack_level,
        }

        if include_model_info:
            info["model_info"] = self.get_models_info()
        if include_model_failures:
            info["model_info_failures"] = copy.deepcopy(self._models_failed_to_train_errors)

        return info

    def reduce_memory_size(
        self,
        remove_data=True,
        remove_fit_stack=False,
        remove_fit=True,
        remove_info=False,
        requires_save=True,
        reduce_children=False,
        **kwargs,
    ):
        if remove_data and self.is_data_saved:
            data_files = [
                os.path.join(self.path_data, "X.pkl"),
                os.path.join(self.path_data, "X_val.pkl"),
                os.path.join(self.path_data, "y.pkl"),
                os.path.join(self.path_data, "y_val.pkl"),
            ]
            for data_file in data_files:
                try:
                    os.remove(data_file)
                except FileNotFoundError:
                    pass
            if requires_save:
                self.is_data_saved = False
            try:
                os.rmdir(self.path_data)
            except OSError:
                pass
            shutil.rmtree(path=Path(self._path_attr), ignore_errors=True)
            try:
                os.rmdir(self.path_utils)
            except OSError:
                pass
        if remove_info and requires_save:
            # Remove model failure info artifacts
            self._models_failed_to_train_errors = dict()
        models = self.get_model_names()
        for model in models:
            model = self.load_model(model)
            model.reduce_memory_size(
                remove_fit_stack=remove_fit_stack,
                remove_fit=remove_fit,
                remove_info=remove_info,
                requires_save=requires_save,
                reduce_children=reduce_children,
                **kwargs,
            )
            if requires_save:
                self.save_model(model, reduce_memory=False)
        if requires_save:
            self.save()

    # TODO: Also enable deletion of models which didn't succeed in training (files may still be persisted)
    #  This includes the original HPO fold for stacking
    # Deletes specified models from trainer and from disk (if delete_from_disk=True).
    def delete_models(
        self,
        models_to_keep=None,
        models_to_delete=None,
        allow_delete_cascade=False,
        delete_from_disk=True,
        dry_run=True,
    ):
        if models_to_keep is not None and models_to_delete is not None:
            raise ValueError("Exactly one of [models_to_keep, models_to_delete] must be set.")
        if models_to_keep is not None:
            if not isinstance(models_to_keep, list):
                models_to_keep = [models_to_keep]
            minimum_model_set = set()
            for model in models_to_keep:
                minimum_model_set.update(self.get_minimum_model_set(model))
            minimum_model_set = list(minimum_model_set)
            models_to_remove = [model for model in self.get_model_names() if model not in minimum_model_set]
        elif models_to_delete is not None:
            if not isinstance(models_to_delete, list):
                models_to_delete = [models_to_delete]
            minimum_model_set = set(models_to_delete)
            minimum_model_set_orig = copy.deepcopy(minimum_model_set)
            for model in models_to_delete:
                minimum_model_set.update(nx.algorithms.dag.descendants(self.model_graph, model))
            if not allow_delete_cascade:
                if minimum_model_set != minimum_model_set_orig:
                    raise AssertionError(
                        "models_to_delete contains models which cause a delete cascade due to other models being dependent on them. Set allow_delete_cascade=True to enable the deletion."
                    )
            minimum_model_set = list(minimum_model_set)
            models_to_remove = [model for model in self.get_model_names() if model in minimum_model_set]
        else:
            raise ValueError("Exactly one of [models_to_keep, models_to_delete] must be set.")

        if dry_run:
            logger.log(30, f"Dry run enabled, AutoGluon would have deleted the following models: {models_to_remove}")
            if delete_from_disk:
                for model in models_to_remove:
                    model = self.load_model(model)
                    logger.log(30, f"\tDirectory {model.path} would have been deleted.")
            logger.log(30, "To perform the deletion, set dry_run=False")
            return

        if delete_from_disk:
            for model in models_to_remove:
                model = self.load_model(model)
                model.delete_from_disk()

        for model in models_to_remove:
            self._delete_model_from_graph(model=model)

        models_kept = self.get_model_names()

        if self.model_best is not None and self.model_best not in models_kept:
            try:
                self.model_best = self.get_model_best()
            except AssertionError:
                self.model_best = None

        # TODO: Delete from all the other model dicts
        self.save()

    def _delete_model_from_graph(self, model: str):
        self.model_graph.remove_node(model)
        if model in self.models:
            self.models.pop(model)
        path_attr_model = Path(self._path_attr_model(model))
        shutil.rmtree(path=path_attr_model, ignore_errors=True)

    @staticmethod
    def _process_hyperparameters(hyperparameters: dict) -> dict:
        return process_hyperparameters(hyperparameters=hyperparameters)

    def distill(
        self,
        X=None,
        y=None,
        X_val=None,
        y_val=None,
        X_unlabeled=None,
        time_limit=None,
        hyperparameters=None,
        holdout_frac=None,
        verbosity=None,
        models_name_suffix=None,
        teacher=None,
        teacher_preds="soft",
        augmentation_data=None,
        augment_method="spunge",
        augment_args={"size_factor": 5, "max_size": int(1e5)},
        augmented_sample_weight=1.0,
    ):
        """Various distillation algorithms.
        Args:
            X, y: pd.DataFrame and pd.Series of training data.
                If None, original training data used during predictor.fit() will be loaded.
                This data is split into train/validation if X_val, y_val are None.
            X_val, y_val: pd.DataFrame and pd.Series of validation data.
            time_limit, hyperparameters, holdout_frac: defined as in predictor.fit()
            teacher (None or str):
                If None, uses the model with the highest validation score as the teacher model, otherwise use the specified model name as the teacher.
            teacher_preds (None or str): If None, we only train with original labels (no data augmentation, overrides augment_method)
                If 'hard', labels are hard teacher predictions given by: teacher.predict()
                If 'soft', labels are soft teacher predictions given by: teacher.predict_proba()
                Note: 'hard' and 'soft' are equivalent for regression problems.
                If augment_method specified, teacher predictions are only used to label augmented data (training data keeps original labels).
                To apply label-smoothing: teacher_preds='onehot' will use original training data labels converted to one-hots for multiclass (no data augmentation).  # TODO: expose smoothing-hyperparameter.
            models_name_suffix (str): Suffix to append to each student model's name, new names will look like: 'MODELNAME_dstl_SUFFIX'
            augmentation_data: pd.DataFrame of additional data to use as "augmented data" (does not contain labels).
                When specified, augment_method, augment_args are ignored, and this is the only augmented data that is used (teacher_preds cannot be None).
            augment_method (None or str): specifies which augmentation strategy to utilize. Options: [None, 'spunge','munge']
                If None, no augmentation gets applied.
            }
            augment_args (dict): args passed into the augmentation function corresponding to augment_method.
            augmented_sample_weight (float): Nonnegative value indicating how much to weight augmented samples. This is only considered if sample_weight was initially specified in Predictor.
        """
        if verbosity is None:
            verbosity = self.verbosity

        if teacher is None:
            teacher = self._get_best()

        hyperparameter_tune = False  # TODO: add as argument with scheduler options.
        if augmentation_data is not None and teacher_preds is None:
            raise ValueError("augmentation_data must be None if teacher_preds is None")

        logger.log(
            20,
            f"Distilling with teacher='{teacher}', teacher_preds={str(teacher_preds)}, augment_method={str(augment_method)} ...",
        )
        if teacher not in self.get_model_names(can_infer=True):
            raise AssertionError(
                f"Teacher model '{teacher}' is not a valid teacher model! Either it does not exist or it cannot infer on new data.\n"
                f"Valid teacher models: {self.get_model_names(can_infer=True)}"
            )
        if X is None:
            if y is not None:
                raise ValueError("X cannot be None when y specified.")
            X = self.load_X()
            X_val = self.load_X_val()

        if y is None:
            y = self.load_y()
            y_val = self.load_y_val()

        if X_val is None:
            if y_val is not None:
                raise ValueError("X_val cannot be None when y_val specified.")
            if holdout_frac is None:
                holdout_frac = default_holdout_frac(len(X), hyperparameter_tune)
            X, X_val, y, y_val = generate_train_test_split(
                X, y, problem_type=self.problem_type, test_size=holdout_frac
            )

        y_val_og = y_val.copy()
        og_bagged_mode = self.bagged_mode
        og_verbosity = self.verbosity
        self.bagged_mode = False  # turn off bagging
        self.verbosity = verbosity  # change verbosity for distillation

        if self.sample_weight is not None:
            X, w = extract_column(X, self.sample_weight)

        if teacher_preds is None or teacher_preds == "onehot":
            augment_method = None
            logger.log(
                20,
                "Training students without a teacher model. Set teacher_preds = 'soft' or 'hard' to distill using the best AutoGluon predictor as teacher.",
            )

        if teacher_preds in ["onehot", "soft"]:
            y = format_distillation_labels(y, self.problem_type, self.num_classes)
            y_val = format_distillation_labels(y_val, self.problem_type, self.num_classes)

        if augment_method is None and augmentation_data is None:
            if teacher_preds == "hard":
                y_pred = pd.Series(self.predict(X, model=teacher))
                if (self.problem_type != REGRESSION) and (
                    len(y_pred.unique()) < len(y.unique())
                ):  # add missing labels
                    logger.log(
                        15, "Adding missing labels to distillation dataset by including some real training examples"
                    )
                    indices_to_add = []
                    for clss in y.unique():
                        if clss not in y_pred.unique():
                            logger.log(15, f"Fetching a row with label={clss} from training data")
                            clss_index = y[y == clss].index[0]
                            indices_to_add.append(clss_index)
                    X_extra = X.loc[indices_to_add].copy()
                    y_extra = y.loc[indices_to_add].copy()  # these are actually real training examples
                    X = pd.concat([X, X_extra])
                    y_pred = pd.concat([y_pred, y_extra])
                    if self.sample_weight is not None:
                        w = pd.concat([w, w[indices_to_add]])
                y = y_pred
            elif teacher_preds == "soft":
                y = self.predict_proba(X, model=teacher)
                if self.problem_type == MULTICLASS:
                    y = pd.DataFrame(y)
                else:
                    y = pd.Series(y)
        else:
            X_aug = augment_data(
                X=X,
                feature_metadata=self.feature_metadata,
                augmentation_data=augmentation_data,
                augment_method=augment_method,
                augment_args=augment_args,
            )
            if len(X_aug) > 0:
                if teacher_preds == "hard":
                    y_aug = pd.Series(self.predict(X_aug, model=teacher))
                elif teacher_preds == "soft":
                    y_aug = self.predict_proba(X_aug, model=teacher)
                    if self.problem_type == MULTICLASS:
                        y_aug = pd.DataFrame(y_aug)
                    else:
                        y_aug = pd.Series(y_aug)
                else:
                    raise ValueError(f"Unknown teacher_preds specified: {teacher_preds}")

                X = pd.concat([X, X_aug])
                y = pd.concat([y, y_aug])
                if self.sample_weight is not None:
                    w = pd.concat([w, pd.Series([augmented_sample_weight] * len(X_aug))])

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        if self.sample_weight is not None:
            w.reset_index(drop=True, inplace=True)
            X[self.sample_weight] = w

        name_suffix = "_DSTL"  # all student model names contain this substring
        if models_name_suffix is not None:
            name_suffix = name_suffix + "_" + models_name_suffix

        if hyperparameters is None:
            hyperparameters = {"GBM": {}, "CAT": {}, "NN_TORCH": {}, "RF": {}}
        hyperparameters = self._process_hyperparameters(
            hyperparameters=hyperparameters
        )  # TODO: consider exposing ag_args_fit, excluded_model_types as distill() arguments.
        if teacher_preds is not None and teacher_preds != "hard" and self.problem_type != REGRESSION:
            self._regress_preds_asprobas = True

        core_kwargs = {
            "stack_name": self.distill_stackname,
            "get_models_func": self.construct_model_templates_distillation,
        }
        aux_kwargs = {
            "get_models_func": self.construct_model_templates_distillation,
            "check_if_best": False,
        }

        # self.bagged_mode = True  # TODO: Add options for bagging
        models = self.train_multi_levels(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            hyperparameters=hyperparameters,
            time_limit=time_limit,  # FIXME: Also limit augmentation time
            name_suffix=name_suffix,
            core_kwargs=core_kwargs,
            aux_kwargs=aux_kwargs,
        )

        distilled_model_names = []
        w_val = None
        if self.weight_evaluation:
            X_val, w_val = extract_column(X_val, self.sample_weight)
        for model_name in models:  # finally measure original metric on validation data and overwrite stored val_scores
            model_score = self.score(X_val, y_val_og, model=model_name, weights=w_val)
            model_obj = self.load_model(model_name)
            model_obj.val_score = model_score
            model_obj.save()  # TODO: consider omitting for sake of efficiency
            self.model_graph.nodes[model_name]["val_score"] = model_score
            distilled_model_names.append(model_name)
        leaderboard = self.leaderboard()
        logger.log(20, "Distilled model leaderboard:")
        leaderboard_distilled = leaderboard[leaderboard["model"].isin(models)].reset_index(drop=True)
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            logger.log(20, leaderboard_distilled)

        # reset trainer to old state before distill() was called:
        self.bagged_mode = og_bagged_mode  # TODO: Confirm if safe to train future models after training models in both bagged and non-bagged modes
        self.verbosity = og_verbosity
        return distilled_model_names

    def _get_model_fit_kwargs(
        self,
        X: pd.DataFrame,
        X_val: pd.DataFrame,
        time_limit: float,
        k_fold: int,
        fit_kwargs: dict,
        ens_sample_weight: list | None = None,
        label_cleaner: None | LabelCleaner = None,
    ) -> dict:
        # Returns kwargs to be passed to AbstractModel's fit function
        if fit_kwargs is None:
            fit_kwargs = dict()

        model_fit_kwargs = dict(time_limit=time_limit, verbosity=self.verbosity, **fit_kwargs)
        if self.sample_weight is not None:
            X, w_train = extract_column(X, self.sample_weight)
            if w_train is not None:  # may be None for ensemble
                # TODO: consider moving weight normalization into AbstractModel.fit()
                model_fit_kwargs["sample_weight"] = (
                    w_train.values / w_train.mean()
                )  # normalization can affect gradient algorithms like boosting
            if X_val is not None:
                X_val, w_val = extract_column(X_val, self.sample_weight)
                if (
                    self.weight_evaluation and w_val is not None
                ):  # ignore validation sample weights unless weight_evaluation specified
                    model_fit_kwargs["sample_weight_val"] = w_val.values / w_val.mean()
            if ens_sample_weight is not None:
                model_fit_kwargs["sample_weight"] = (
                    ens_sample_weight  # sample weights to use for weighted ensemble only
                )
        if self._groups is not None and "groups" not in model_fit_kwargs:
            if k_fold == self.k_fold:  # don't do this on refit full
                model_fit_kwargs["groups"] = self._groups

        if label_cleaner is not None:
            model_fit_kwargs["label_cleaner"] = label_cleaner

        # FIXME: Sample weight `extract_column` is a hack, have to compute feature_metadata here because sample weight column could be in X upstream, extract sample weight column upstream instead.
        if "feature_metadata" not in model_fit_kwargs:
            raise AssertionError(f"Missing expected parameter 'feature_metadata'.")
        return model_fit_kwargs

    def _get_bagged_model_fit_kwargs(
        self, k_fold: int, k_fold_start: int, k_fold_end: int, n_repeats: int, n_repeat_start: int
    ) -> dict:
        # Returns additional kwargs (aside from _get_model_fit_kwargs) to be passed to BaggedEnsembleModel's fit function
        if k_fold is None:
            k_fold = self.k_fold
        if n_repeats is None:
            n_repeats = self.n_repeats
        return dict(
            k_fold=k_fold,
            k_fold_start=k_fold_start,
            k_fold_end=k_fold_end,
            n_repeats=n_repeats,
            n_repeat_start=n_repeat_start,
            compute_base_preds=False,
        )

    def _get_feature_prune_proxy_model(self, proxy_model_class: AbstractModel | None, level: int) -> AbstractModel:
        """
        Returns proxy model to be used for feature pruning - the base learner that has the highest validation score in a particular stack layer.
        Ties are broken by inference speed. If proxy_model_class is not None, take the best base learner belonging to proxy_model_class.
        proxy_model_class is an AbstractModel class (ex. LGBModel).
        """
        proxy_model = None
        if isinstance(proxy_model_class, str):
            raise AssertionError(
                f"proxy_model_class must be a subclass of AbstractModel. Was instead a string: {proxy_model_class}"
            )
        banned_models = [GreedyWeightedEnsembleModel, SimpleWeightedEnsembleModel]
        assert proxy_model_class not in banned_models, (
            "WeightedEnsemble models cannot be feature pruning proxy models."
        )

        leaderboard = self.leaderboard()
        banned_names = []
        candidate_model_rows = leaderboard[(~leaderboard["score_val"].isna()) & (leaderboard["stack_level"] == level)]
        candidate_models_type_inner = self.get_models_attribute_dict(
            attribute="type_inner", models=candidate_model_rows["model"]
        )
        for model_name, type_inner in candidate_models_type_inner.copy().items():
            if type_inner in banned_models:
                banned_names.append(model_name)
                candidate_models_type_inner.pop(model_name, None)
        banned_names = set(banned_names)
        candidate_model_rows = candidate_model_rows[~candidate_model_rows["model"].isin(banned_names)]
        if proxy_model_class is not None:
            candidate_model_names = [
                model_name
                for model_name, model_class in candidate_models_type_inner.items()
                if model_class == proxy_model_class
            ]
            candidate_model_rows = candidate_model_rows[candidate_model_rows["model"].isin(candidate_model_names)]
        if len(candidate_model_rows) == 0:
            if proxy_model_class is None:
                logger.warning(f"No models from level {level} have been successfully fit. Skipping feature pruning.")
            else:
                logger.warning(
                    f"No models of type {proxy_model_class} have finished training in level {level}. Skipping feature pruning."
                )
            return proxy_model
        best_candidate_model_rows = candidate_model_rows.loc[
            candidate_model_rows["score_val"] == candidate_model_rows["score_val"].max()
        ]
        return self.load_model(best_candidate_model_rows.loc[best_candidate_model_rows["fit_time"].idxmin()]["model"])

    def calibrate_model(
        self, model_name: str | None = None, lr: float = 0.1, max_iter: int = 200, init_val: float = 1.0
    ):
        """
        Applies temperature scaling to a model.
        Applies inverse softmax to predicted probs then trains temperature scalar
        on validation data to maximize negative log likelihood.
        Inversed softmaxes are divided by temperature scalar
        then softmaxed to return predicted probs.

        Parameters:
        -----------
        model_name: str: default = None
            model name to tune temperature scaling on.
            If None, will tune best model only. Best model chosen by validation score
        lr: float: default = 0.1
            The learning rate for temperature scaling algorithm
        max_iter: int: default = 200
            Number of iterations optimizer should take for
            tuning temperature scaler
        init_val: float: default = 1.0
            The initial value for temperature scalar term
        """
        # TODO: Note that temperature scaling is known to worsen calibration in the face of shifted test data.
        try:
            # FIXME: Avoid depending on torch for temp scaling
            try_import_torch()
        except ImportError:
            logger.log(30, "Warning: Torch is not installed, skipping calibration step...")
            return

        if model_name is None:
            if self.has_val:
                can_infer = True
            else:
                can_infer = None
            if self.model_best is not None:
                models = self.get_model_names(can_infer=can_infer)
                if self.model_best in models:
                    model_name = self.model_best
            if model_name is None:
                model_name = self.get_model_best(can_infer=can_infer)

        model_refit_map = self.model_refit_map()
        model_name_og = model_name
        for m, m_full in model_refit_map.items():
            if m_full == model_name:
                model_name_og = m
                break
        if self.has_val:
            X_val = self.load_X_val()
            y_val_probs = self.predict_proba(X_val, model_name_og)
            y_val = self.load_y_val().to_numpy()
        else:  # bagged mode
            y_val_probs = self.get_model_oof(model_name_og)
            y_val = self.load_y().to_numpy()

        y_val_probs_og = y_val_probs
        if self.problem_type == BINARY:
            # Convert one-dimensional array to be in the form of a 2-class multiclass predict_proba output
            y_val_probs = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_val_probs)

        model = self.load_model(model_name=model_name)
        if self.problem_type == QUANTILE:
            logger.log(15, f"Conformity scores being computed to calibrate model: {model_name}")
            conformalize = compute_conformity_score(
                y_val_pred=y_val_probs, y_val=y_val, quantile_levels=self.quantile_levels
            )
            model.conformalize = conformalize
            model.save()
        else:
            logger.log(15, f"Temperature scaling term being tuned for model: {model_name}")
            temp_scalar = tune_temperature_scaling(
                y_val_probs=y_val_probs, y_val=y_val, init_val=init_val, max_iter=max_iter, lr=lr
            )
            if temp_scalar is None:
                logger.log(
                    15,
                    f"Warning: Infinity found during calibration, skipping calibration on {model.name}! "
                    f"This can occur when the model is absolutely certain of a validation prediction (1.0 pred_proba).",
                )
            elif temp_scalar <= 0:
                logger.log(
                    30,
                    f"Warning: Temperature scaling found optimal at a negative value ({temp_scalar}). Disabling temperature scaling to avoid overfitting.",
                )
            else:
                # Check that scaling improves performance for the target metric
                score_without_temp = self.score_with_y_pred_proba(y=y_val, y_pred_proba=y_val_probs_og, weights=None)
                scaled_y_val_probs = apply_temperature_scaling(
                    y_val_probs, temp_scalar, problem_type=self.problem_type, transform_binary_proba=False
                )
                score_with_temp = self.score_with_y_pred_proba(y=y_val, y_pred_proba=scaled_y_val_probs, weights=None)

                if score_with_temp > score_without_temp:
                    logger.log(15, f"Temperature term found is: {temp_scalar}")
                    model.params_aux["temperature_scalar"] = temp_scalar
                    model.save()
                else:
                    logger.log(15, "Temperature did not improve performance, skipping calibration.")

    def calibrate_decision_threshold(
        self,
        X: pd.DataFrame | None = None,
        y: np.ndarray | None = None,
        metric: str | Scorer | None = None,
        model: str = "best",
        weights=None,
        decision_thresholds: int | list[float] = 25,
        secondary_decision_thresholds: int | None = 19,
        verbose: bool = True,
        **kwargs,
    ) -> float:
        # TODO: Docstring
        assert self.problem_type == BINARY, (
            f'calibrate_decision_threshold is only available for `problem_type="{BINARY}"`'
        )

        if metric is None:
            metric = self.eval_metric
        elif isinstance(metric, str):
            metric = get_metric(metric, self.problem_type, "eval_metric")

        if model == "best":
            model = self.get_model_best()

        if y is None:
            # If model is refit_full, use its parent to avoid over-fitting
            model_parent = self.get_refit_full_parent(model=model)
            if not self.model_exists(model_parent):
                raise AssertionError(
                    f"Unable to calibrate the decision threshold on the internal data because the "
                    f'model "{model}" is a refit_full model trained on all of the internal data, '
                    f'whose parent model "{model_parent}" does not exist or was deleted.\n'
                    f"It may have been deleted due to `predictor.fit(..., keep_only_best=True)`. "
                    f"Ensure `keep_only_best=False` to be able to calibrate refit_full models."
                )
            model = model_parent

            # TODO: Add helpful logging when data is not available, for example post optimize for deployment
            if self.has_val:
                # Use validation data
                X = self.load_X_val()
                if self.weight_evaluation:
                    X, weights = extract_column(X=X, col_name=self.sample_weight)
                y: np.array = self.load_y_val()
                y_pred_proba = self.predict_proba(X=X, model=model)
            else:
                # Use out-of-fold data
                if self.weight_evaluation:
                    X = self.load_X()
                    X, weights = extract_column(X=X, col_name=self.sample_weight)
                y: np.array = self.load_y()
                y_pred_proba = self.get_model_oof(model=model)
        else:
            y_pred_proba = self.predict_proba(X=X, model=model)

        if not metric.needs_pred:
            logger.warning(
                f'WARNING: The provided metric "{metric.name}" does not use class predictions for scoring, '
                f"and thus is invalid for decision threshold calibration. "
                f"Falling back to `decision_threshold=0.5`."
            )
            return 0.5

        return calibrate_decision_threshold(
            y=y,
            y_pred_proba=y_pred_proba,
            metric=lambda y, y_pred: self.score_with_y_pred(y=y, y_pred=y_pred, weights=weights, metric=metric),
            decision_thresholds=decision_thresholds,
            secondary_decision_thresholds=secondary_decision_thresholds,
            metric_name=metric.name,
            verbose=verbose,
            **kwargs,
        )

    @staticmethod
    def _validate_num_classes(num_classes: int | None, problem_type: str):
        if problem_type == BINARY:
            assert num_classes is not None and num_classes == 2, (
                f"num_classes must be 2 when problem_type='{problem_type}' (num_classes={num_classes})"
            )
        elif problem_type in [MULTICLASS, SOFTCLASS]:
            assert num_classes is not None and num_classes >= 2, (
                f"num_classes must be >=2 when problem_type='{problem_type}' (num_classes={num_classes})"
            )
        elif problem_type in [REGRESSION, QUANTILE]:
            assert num_classes is None, (
                f"num_classes must be None when problem_type='{problem_type}' (num_classes={num_classes})"
            )
        else:
            raise AssertionError(
                f"Unknown problem_type: '{problem_type}'. Valid problem types: {[BINARY, MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE]}"
            )

    @staticmethod
    def _validate_quantile_levels(quantile_levels: list[float] | np.ndarray | None, problem_type: str):
        if problem_type == QUANTILE:
            assert quantile_levels is not None, (
                f"quantile_levels must not be None when problem_type='{problem_type}' (quantile_levels={quantile_levels})"
            )
            assert isinstance(quantile_levels, (list, np.ndarray)), (
                f"quantile_levels must be a list or np.ndarray (quantile_levels={quantile_levels})"
            )
            assert len(quantile_levels) > 0, (
                f"quantile_levels must not be an empty list (quantile_levels={quantile_levels})"
            )
        else:
            assert quantile_levels is None, (
                f"quantile_levels must be None when problem_type='{problem_type}' (quantile_levels={quantile_levels})"
            )


def _detached_train_multi_fold(
    *,
    _self: AbstractTabularTrainer,
    model: str | AbstractModel,
    X: pd.DataFrame,
    y: pd.Series,
    time_split: bool,
    time_start: float,
    time_limit: float | None,
    time_limit_model_split: float | None,
    hyperparameter_tune_kwargs: dict,
    is_ray_worker: bool = False,
    kwargs: dict,
) -> list[str]:
    """Dedicated class-detached function to train a single model on multiple folds."""
    if isinstance(model, str):
        model = _self.load_model(model)
    elif _self.low_memory:
        model = copy.deepcopy(model)
    if hyperparameter_tune_kwargs is not None and isinstance(hyperparameter_tune_kwargs, dict):
        hyperparameter_tune_kwargs_model = hyperparameter_tune_kwargs.get(model.name, None)
    else:
        hyperparameter_tune_kwargs_model = None
    # TODO: Only update scores when finished, only update model as part of final models if finished!
    if time_split:
        time_left = time_limit_model_split
    else:
        if time_limit is None:
            time_left = None
        else:
            time_start_model = time.time()
            time_left = time_limit - (time_start_model - time_start)

    model_name_trained_lst = _self._train_single_full(
        X,
        y,
        model,
        time_limit=time_left,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs_model,
        is_ray_worker=is_ray_worker,
        **kwargs,
    )

    if _self.low_memory:
        del model

    return model_name_trained_lst


def _remote_train_multi_fold(
    *,
    _self: AbstractTabularTrainer,
    model: str | AbstractModel,
    X: pd.DataFrame,
    y: pd.Series,
    time_split: bool,
    time_start: float,
    time_limit: float | None,
    time_limit_model_split: float | None,
    hyperparameter_tune_kwargs: dict,
    kwargs: dict,
    errors: Literal["ignore", "raise"] | None = None,
) -> tuple[str, str | None, str | None, Exception | None, dict | None]:
    reset_logger_for_remote_call(verbosity=_self.verbosity)

    if errors is not None:
        kwargs["errors"] = errors

    exception = None
    try:
        model_name_list = _detached_train_multi_fold(
            _self=_self,
            model=model,
            X=X,
            y=y,
            time_start=time_start,
            time_split=time_split,
            time_limit=time_limit,
            time_limit_model_split=time_limit_model_split,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            is_ray_worker=True,
            kwargs=kwargs,
        )
    except Exception as exc:
        model_name_list = []
        if errors is not None and errors == "raise":
            # If training fails and exception is returned, collect the exception information and return
            exception = exc  # required to use in outer scope
        else:
            raise exc

    if not model_name_list:
        model_name = model if isinstance(model, str) else model.name
        # Get model_failure metadata if it exists
        model_failure_info = None
        if model_name in _self._models_failed_to_train_errors:
            model_failure_info = _self._models_failed_to_train_errors[model_name]
        return model_name, None, None, exception, model_failure_info

    # Fallback, return original model name if training failed.
    if not model_name_list:
        model_name = model if isinstance(model, str) else model.name
        return model_name, None, None, None, None
    model_name = model_name_list[0]
    return (
        model_name,
        _self.get_model_attribute(model=model_name, attribute="path"),
        _self.get_model_attribute(model=model_name, attribute="type"),
        None,
        None,
    )


def _detached_refit_single_full(
    *,
    _self: AbstractTabularTrainer,
    model: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_unlabeled: pd.DataFrame,
    level: int,
    kwargs: dict,
    fit_strategy: Literal["sequential", "parallel"] = "sequential",
) -> tuple[str, list[str]]:
    # TODO: loading the model is the reasons we must allocate GPU resources for this job in cases where models require GPU when loaded from disk
    model = _self.load_model(model)
    model_name = model.name
    reuse_first_fold = False

    if isinstance(model, BaggedEnsembleModel):
        # Reuse if model is already _FULL and no X_val
        if X_val is None:
            reuse_first_fold = not model._bagged_mode

    if not reuse_first_fold:
        if isinstance(model, BaggedEnsembleModel):
            can_refit_full = model._get_tags_child().get("can_refit_full", False)
        else:
            can_refit_full = model._get_tags().get("can_refit_full", False)
        reuse_first_fold = not can_refit_full

    if not reuse_first_fold:
        model_full = model.convert_to_refit_full_template()
        # Mitigates situation where bagged models barely had enough memory and refit requires more. Worst case results in OOM, but this lowers chance of failure.
        model_full._user_params_aux["max_memory_usage_ratio"] = model.params_aux["max_memory_usage_ratio"] * 1.15
        # Re-set user specified training resources.
        # FIXME: this is technically also a bug for non-distributed mode, but there it is good to use more/all resources per refit.
        # FIXME: Unsure if it is better to do model.fit_num_cpus or model.fit_num_cpus_child,
        #  (Nick): I'm currently leaning towards model.fit_num_cpus, it is also less memory intensive
        # Better to not specify this for sequential fits, since we want the models to use the optimal amount of resources,
        # which could be less than the available resources (ex: LightGBM fits faster using 50% of the cores)
        if fit_strategy == "parallel":
            # FIXME: Why use `model.fit_num_cpus_child` when we can use the same values as was passed to `ray` for the process, just pass those values as kwargs. Eliminates chance of inconsistency.
            if model.fit_num_cpus_child is not None:
                model_full._user_params_aux["num_cpus"] = model.fit_num_cpus_child
            if model.fit_num_gpus_child is not None:
                model_full._user_params_aux["num_gpus"] = model.fit_num_gpus_child
        # TODO: Do it for all models in the level at once to avoid repeated processing of data?
        base_model_names = _self.get_base_model_names(model_name)
        # FIXME: Logs for inference speed (1 row) are incorrect because
        #  parents are non-refit models in this sequence and later correct after logging.
        #  Avoiding fix at present to minimize hacks in the code.
        #  Return to this later when Trainer controls all stacking logic to map correct parent.
        models_trained = _self.stack_new_level_core(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            X_unlabeled=X_unlabeled,
            models=[model_full],
            base_model_names=base_model_names,
            level=level,
            stack_name=REFIT_FULL_NAME,
            hyperparameter_tune_kwargs=None,
            feature_prune=False,
            k_fold=1,
            n_repeats=1,
            ensemble_type=type(model),
            refit_full=True,
            **kwargs,
        )
        if len(models_trained) == 0:
            reuse_first_fold = True
            logger.log(
                30,
                f"WARNING: Refit training failure detected for '{model_name}'... "
                f"Falling back to using first fold to avoid downstream exception."
                f"\n\tThis is likely due to an out-of-memory error or other memory related issue. "
                f"\n\tPlease create a GitHub issue if this was triggered from a non-memory related problem.",
            )
            if not model.params.get("save_bag_folds", True):
                raise AssertionError(
                    f"Cannot avoid training failure during refit for '{model_name}' by falling back to "
                    f"copying the first fold because it does not exist! (save_bag_folds=False)"
                    f"\n\tPlease specify `save_bag_folds=True` in the `.fit` call to avoid this exception."
                )

    if reuse_first_fold:
        # Perform fallback black-box refit logic that doesn't retrain.
        model_full = model.convert_to_refit_full_via_copy()
        # FIXME: validation time not correct for infer 1 batch time, needed to hack _is_refit=True to fix
        logger.log(20, f"Fitting model: {model_full.name} | Skipping fit via cloning parent ...")
        _self._add_model(model_full, stack_name=REFIT_FULL_NAME, level=level, _is_refit=True)
        _self.save_model(model_full)
        models_trained = [model_full.name]

    return model_name, models_trained


def _remote_refit_single_full(
    *,
    _self: AbstractTabularTrainer,
    model: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_unlabeled: pd.DataFrame,
    level: int,
    kwargs: dict,
    fit_strategy: Literal["sequential", "parallel"],
) -> tuple[str, str, list[str], str, str]:
    reset_logger_for_remote_call(verbosity=_self.verbosity)

    model_name, models_trained = _detached_refit_single_full(
        _self=_self,
        model=model,
        X=X,
        y=y,
        X_val=X_val,
        y_val=y_val,
        X_unlabeled=X_unlabeled,
        level=level,
        kwargs=kwargs,
        fit_strategy=fit_strategy,
    )

    # We always just refit one model per call, so this must be the case.
    assert len(models_trained) == 1
    refitted_model_name = models_trained[0]
    return (
        model_name,
        refitted_model_name,
        _self.get_model_attribute(model=refitted_model_name, attribute="path"),
        _self.get_model_attribute(model=refitted_model_name, attribute="type"),
    )
