import copy
import inspect
import logging
import math
import os
import platform
import time
from collections import Counter
from statistics import mean
from typing import Dict, Type, Union

import numpy as np
import pandas as pd

from autogluon.common.utils.log_utils import DuplicateFilter
from .fold_fitting_strategy import AbstractFoldFittingStrategy, SequentialLocalFoldFittingStrategy, ParallelLocalFoldFittingStrategy
from ..abstract.abstract_model import AbstractModel
from ..abstract.model_trial import model_trial, skip_hpo
from ...constants import MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE, REFIT_FULL_SUFFIX
from ...hpo.exceptions import EmptySearchSpace
from ...utils.exceptions import TimeLimitExceeded
from ...utils.loaders import load_pkl
from ...utils.savers import save_pkl
from ...utils.try_import import try_import_ray
from ...utils.utils import CVSplitter, _compute_fi_with_stddev


logger = logging.getLogger(__name__)
dup_filter = DuplicateFilter()
logger.addFilter(dup_filter)


# TODO: Add metadata object with info like score on each model, train time on each model, etc.
class BaggedEnsembleModel(AbstractModel):
    """
    Bagged ensemble meta-model which fits a given model multiple times across different splits of the training data.

    For certain child models such as KNN, this may only train a single model and instead rely on the child model to generate out-of-fold predictions.

    Parameters
    ----------
    model_base : Union[AbstractModel, Type[AbstractModel]]
        The base model to repeatedly fit during bagging.
        If a AbstractModel class, then also provide model_base_kwargs which will be used to initialize the model via model_base(**model_base_kwargs).
    model_base_kwargs : Dict[str, any], default = None
        kwargs used to initialize model_base if model_base is a class.
    random_state : int, default = 0
        Random state used to split the data into cross-validation folds during fit.
    **kwargs
        Refer to AbstractModel documentation
    """
    _oof_filename = 'oof.pkl'

    def __init__(self, model_base: Union[AbstractModel, Type[AbstractModel]], model_base_kwargs: Dict[str, any] = None, random_state: int = 0, **kwargs):
        if inspect.isclass(model_base):
            if model_base_kwargs is None:
                model_base_kwargs = dict()
            self.model_base: AbstractModel = model_base(**model_base_kwargs)
        elif model_base_kwargs is not None:
            raise AssertionError(f'model_base_kwargs must be None if model_base was passed as an object! '
                                 f'(model_base: {model_base}, model_base_kwargs: {model_base_kwargs})')
        else:
            self.model_base: AbstractModel = model_base
        self._child_type = type(self.model_base)
        self.models = []
        self._oof_pred_proba = None
        self._oof_pred_model_repeats = None
        self._n_repeats = 0  # Number of n_repeats with at least 1 model fit, if kfold=5 and 8 models have been fit, _n_repeats is 2
        self._n_repeats_finished = 0  # Number of n_repeats finished, if kfold=5 and 8 models have been fit, _n_repeats_finished is 1
        self._k_fold_end = 0  # Number of models fit in current n_repeat (0 if completed), if kfold=5 and 8 models have been fit, _k_fold_end is 3
        self._k = None  # k models per n_repeat, equivalent to kfold value
        self._k_per_n_repeat = []  # k-fold used for each n_repeat. == [5, 10, 3] if first kfold was 5, second was 10, and third was 3
        self._random_state = random_state
        self.low_memory = True
        self._bagged_mode = None
        # _child_oof currently is only set to True for KNN models, that are capable of LOO prediction generation to avoid needing bagging.
        # TODO: Consider moving `_child_oof` logic to a separate class / refactor OOF logic.
        # FIXME: Avoid unnecessary refit during refit_full on `_child_oof=True` models, just re-use the original model.
        self._child_oof = False  # Whether the OOF preds were taken from a single child model (Assumes child can produce OOF preds without bagging).
        self._cv_splitters = []  # Keeps track of the CV splitter used for each bagged repeat.
        self._params_aux_child = None  # aux params of child model

        super().__init__(problem_type=self.model_base.problem_type, eval_metric=self.model_base.eval_metric, **kwargs)

    def _set_default_params(self):
        default_params = {
            # 'use_child_oof': False,  # [Advanced] Whether to defer to child model for OOF preds and only train a single child.
            'save_bag_folds': True,
            # 'refit_folds': False,  # [Advanced, Experimental] Whether to refit bags immediately to a refit_full model in a single .fit call.
            # 'num_folds' None,  # Number of bagged folds per set. If specified, overrides .fit `k_fold` value.
            # 'max_sets': None,  # Maximum bagged repeats to allow, if specified, will set `self.can_fit()` to `self._n_repeats_finished < max_repeats`
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            drop_unique=False,  # TODO: Get the value from child instead
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def is_valid(self):
        return self.is_fit() and (self._n_repeats == self._n_repeats_finished)

    def can_infer(self):
        return self.is_fit() and self.params.get('save_bag_folds', True)

    def is_stratified(self):
        if self.problem_type in [REGRESSION, QUANTILE, SOFTCLASS]:
            return False
        else:
            return True

    def is_fit(self):
        return len(self.models) != 0

    def can_fit(self) -> bool:
        if not self.is_fit():
            return True
        if not self._bagged_mode:
            return False
        # If max_sets is specified and the model has already fit >=max_sets, return False
        return self._get_model_params().get('max_sets', None) is None or self._get_model_params().get('max_sets') > self._n_repeats_finished

    def is_valid_oof(self):
        return self.is_fit() and (self._child_oof or self._bagged_mode)

    def get_oof_pred_proba(self, **kwargs):
        # TODO: Require is_valid == True (add option param to ignore is_valid)
        return self._oof_pred_proba_func(self._oof_pred_proba, self._oof_pred_model_repeats)

    @staticmethod
    def _oof_pred_proba_func(oof_pred_proba, oof_pred_model_repeats):
        oof_pred_model_repeats_without_0 = np.where(oof_pred_model_repeats == 0, 1, oof_pred_model_repeats)
        if oof_pred_proba.ndim == 2:
            oof_pred_model_repeats_without_0 = oof_pred_model_repeats_without_0[:, None]
        return oof_pred_proba / oof_pred_model_repeats_without_0

    def _init_misc(self, **kwargs):
        child = self._get_model_base().convert_to_template()
        child.initialize(**kwargs)
        self.eval_metric = child.eval_metric
        self.stopping_metric = child.stopping_metric
        self.quantile_levels = child.quantile_levels
        self.normalize_pred_probas = child.normalize_pred_probas
        self._params_aux_child = child.params_aux

    def preprocess(self, X, preprocess_nonadaptive=True, model=None, **kwargs):
        if preprocess_nonadaptive:
            if model is None:
                if not self.models:
                    return X
                model = self.models[0]
            model = self.load_child(model)
            return model.preprocess(X, preprocess_stateful=False)
        else:
            return X

    def _get_cv_splitter(self, n_splits, n_repeats, groups=None):
        return CVSplitter(n_splits=n_splits, n_repeats=n_repeats, groups=groups, stratified=self.is_stratified(), random_state=self._random_state)

    def _fit(self,
             X,
             y,
             X_val=None,
             y_val=None,
             X_pseudo=None,
             y_pseudo=None,
             k_fold=None,
             k_fold_start=0,
             k_fold_end=None,
             n_repeats=1,
             n_repeat_start=0,
             groups=None,
             **kwargs):
        use_child_oof = self.params.get('use_child_oof', False)
        if use_child_oof:
            if self.is_fit():
                # TODO: We may want to throw an exception instead and avoid calling fit more than once
                return self
            k_fold = 1
            k_fold_end = None
            groups = None
        else:
            k_fold, k_fold_end = self._update_k_fold(k_fold=k_fold, k_fold_end=k_fold_end)
        if k_fold is None and groups is None:
            k_fold = 5
        if k_fold is not None and k_fold < 1:
            k_fold = 1
        if k_fold is None or k_fold > 1:
            k_fold = self._get_cv_splitter(n_splits=k_fold, n_repeats=n_repeats, groups=groups).n_splits
        max_sets = self._get_model_params().get('max_sets', None)
        if max_sets is not None:
            if n_repeats > max_sets:
                n_repeats = max_sets
        self._validate_bag_kwargs(
            k_fold=k_fold,
            k_fold_start=k_fold_start,
            k_fold_end=k_fold_end,
            n_repeats=n_repeats,
            n_repeat_start=n_repeat_start,
            groups=groups,
        )
        if k_fold_end is None:
            k_fold_end = k_fold

        model_base = self._get_model_base()
        model_base.rename(name='')
        kwargs['feature_metadata'] = self.feature_metadata
        kwargs['num_classes'] = self.num_classes  # TODO: maybe don't pass num_classes to children

        if self.model_base is not None:
            self.save_model_base(self.model_base)
            self.model_base = None

        if self._oof_pred_proba is None and self.is_fit():
            self._load_oof()

        if (not self._get_tags_child().get('can_refit_full', False) or k_fold == 1) and not self.params.get('save_bag_folds', True):
            # TODO: This is a hack to avoid a very challenging situation:
            #  in high_quality preset we don't save fold models, but for models that don't support refit_full,
            #  they must copy the first fold model instead of fitting again.
            #  Therefore we must override save_bag_folds for these unsupported models so that the refit versions have a fold model to copy.
            #  This could be implemented better by only keeping the first fold model artifact and avoid saving the other fold model artifacts (lower disk usage)
            #  However, this is complex to code accounting for the fitting strategies and would be prone to difficult to diagnose bugs.
            self.params['save_bag_folds'] = True
            if k_fold != 1:
                # Only log in the situation where functionality is currently suboptimal
                logger.log(20, '\tForcing `save_bag_folds=True` because child model does not support `refit_full`.')

        save_bag_folds = self.params.get('save_bag_folds', True)
        if k_fold == 1:
            self._fit_single(X=X, y=y, model_base=model_base, use_child_oof=use_child_oof, **kwargs)
            return self
        else:
            refit_folds = self.params.get('refit_folds', False)
            if refit_folds:
                save_bag_folds = False
                if kwargs.get('time_limit', None) is not None:
                    fold_start = n_repeat_start * k_fold + k_fold_start
                    fold_end = (n_repeats - 1) * k_fold + k_fold_end
                    folds_to_fit = fold_end - fold_start
                    # Reserve time for final refit model
                    kwargs['time_limit'] = kwargs['time_limit'] * folds_to_fit / (folds_to_fit + 1.2)
            self._fit_folds(X=X, y=y, model_base=model_base, X_pseudo=X_pseudo, y_pseudo=y_pseudo,
                            k_fold=k_fold, k_fold_start=k_fold_start, k_fold_end=k_fold_end,
                            n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
            # FIXME: Don't save folds except for refit
            # FIXME: Cleanup self
            # FIXME: Don't add `_FULL` to name
            # FIXME: Support `can_refit_full=False` models
            if refit_folds:
                refit_template = self.convert_to_refit_full_template()
                refit_template.params['use_child_oof'] = False
                kwargs['time_limit'] = None
                refit_template.fit(X=X, y=y, k_fold=1, **kwargs)
                refit_template._oof_pred_proba = self._oof_pred_proba
                refit_template._oof_pred_model_repeats = self._oof_pred_model_repeats
                refit_template._child_oof = True
                refit_template.fit_time += self.fit_time + self.predict_time
                return refit_template
            else:
                return self

    def _update_k_fold(self, k_fold, k_fold_end=None, verbose=True):
        """Update k_fold and k_fold_end in case num_folds was specified"""
        k_fold_override = self.params.get('num_folds', None)
        if k_fold_override is not None:
            if k_fold is not None:
                if k_fold != k_fold_override and verbose:
                    logger.log(20, f'\tSetting folds to {k_fold_override}. Ignoring `k_fold={k_fold}` because `num_folds={k_fold_override}` overrides.')
                if k_fold_end is not None and k_fold_end == k_fold:
                    k_fold_end = k_fold_override
            k_fold = k_fold_override
        return k_fold, k_fold_end

    def _get_child_aux_val(self, key: str, default=None):
        assert self.is_initialized(), "Model must be initialized before calling self._get_child_aux_val!"
        return self._params_aux_child.get(key, default)

    def _validate_bag_kwargs(self, *,
                             k_fold,
                             k_fold_start,
                             k_fold_end,
                             n_repeats,
                             n_repeat_start,
                             groups):
        if groups is not None:
            if self._n_repeats_finished != 0:
                raise AssertionError('Bagged models cannot call fit with `groups` specified when a full k-fold set has already been fit.')
            if n_repeats > 1:
                raise AssertionError('Cannot perform repeated bagging with `groups` specified.')
            return

        if k_fold_end is None:
            k_fold_end = k_fold
        if k_fold is None:
            raise ValueError('k_fold cannot be None.')
        if k_fold < 1:
            raise ValueError(f'k_fold must be equal or greater than 1, value: ({k_fold})')
        if n_repeat_start != self._n_repeats_finished:
            raise ValueError(f'n_repeat_start must equal self._n_repeats_finished, values: ({n_repeat_start}, {self._n_repeats_finished})')
        if n_repeats <= n_repeat_start:
            raise ValueError(f'n_repeats must be greater than n_repeat_start, values: ({n_repeats}, {n_repeat_start})')
        if k_fold_start != self._k_fold_end:
            raise ValueError(f'k_fold_start must equal previous k_fold_end, values: ({k_fold_start}, {self._k_fold_end})')
        if k_fold_start >= k_fold_end:
            # TODO: Remove this limitation if n_repeats > 1
            raise ValueError(f'k_fold_end must be greater than k_fold_start, values: ({k_fold_end}, {k_fold_start})')
        if (n_repeats - n_repeat_start) > 1 and k_fold_end != k_fold:
            # TODO: Remove this limitation
            raise ValueError(f'k_fold_end must equal k_fold when (n_repeats - n_repeat_start) > 1, values: ({k_fold_end}, {k_fold})')
        if self._k is not None and self._k != k_fold:
            raise ValueError(f'k_fold must equal previously fit k_fold value for the current n_repeat, values: (({k_fold}, {self._k})')

    def predict_proba(self, X, normalize=None, **kwargs):
        model = self.load_child(self.models[0])
        X = self.preprocess(X, model=model, **kwargs)
        pred_proba = model.predict_proba(X=X, preprocess_nonadaptive=False, normalize=normalize)
        for model in self.models[1:]:
            model = self.load_child(model)
            pred_proba += model.predict_proba(X=X, preprocess_nonadaptive=False, normalize=normalize)
        pred_proba = pred_proba / len(self.models)

        if self.params_aux.get("temperature_scalar", None) is not None:
            pred_proba = self._apply_temperature_scaling(pred_proba)
        elif self.conformalize is not None:
            pred_proba = self._apply_conformalization(pred_proba)

        return pred_proba

    def _predict_proba(self, X, normalize=False, **kwargs):
        return self.predict_proba(X=X, normalize=normalize, **kwargs)

    def score_with_oof(self, y, sample_weight=None):
        self._load_oof()
        valid_indices = self._oof_pred_model_repeats > 0
        y = y[valid_indices]
        y_pred_proba = self.get_oof_pred_proba()[valid_indices]
        if sample_weight is not None:
            sample_weight = sample_weight[valid_indices]
        return self.score_with_y_pred_proba(y=y, y_pred_proba=y_pred_proba, sample_weight=sample_weight)

    def _fit_single(self, X, y, model_base, use_child_oof, time_limit=None, **kwargs):
        if self.is_fit():
            raise AssertionError('Model is already fit.')
        if self._n_repeats != 0:
            raise ValueError(f'n_repeats must equal 0 when fitting a single model with k_fold == 1, value: {self._n_repeats}')
        model_base.name = f'{model_base.name}S1F1'
        model_base.set_contexts(path_context=self.path + model_base.name + os.path.sep)
        time_start_fit = time.time()
        model_base.fit(X=X, y=y, time_limit=time_limit, **kwargs)
        model_base.fit_time = time.time() - time_start_fit
        model_base.predict_time = None
        X_len = len(X)

        # Check if pred_proba is going to take too long
        if time_limit is not None and X_len >= 10000:

            max_allowed_time = time_limit * 1.3  # allow some buffer
            time_left = max(
                max_allowed_time - model_base.fit_time,
                time_limit * 0.1,  # At least 10% of time_limit
                10,  # At least 10 seconds
            )
            # Sample at most 500 rows to estimate prediction time of all rows
            # TODO: Consider moving this into end of abstract model fit for all models.
            #  Currently this only fixes problem when in bagged mode, if not bagging, then inference could still be problematic
            n_sample = min(500, round(X_len * 0.1))
            frac = n_sample / X_len
            X_sample = X.sample(n=n_sample)
            time_start_predict = time.time()
            model_base.predict_proba(X_sample)
            time_predict_frac = time.time() - time_start_predict
            time_predict_estimate = time_predict_frac / frac
            logger.log(15, f'\t{round(time_predict_estimate, 2)}s\t= Estimated out-of-fold prediction time...')
            if time_predict_estimate > time_left:
                logger.warning(f'\tNot enough time to generate out-of-fold predictions for model. Estimated time required was {round(time_predict_estimate, 2)}s compared to {round(time_left, 2)}s of available time.')
                raise TimeLimitExceeded

        if use_child_oof:
            logger.log(15, '\t`use_child_oof` was specified for this model. It will function similarly to a bagged model, but will only fit one child model.')
            time_start_predict = time.time()
            if model_base._get_tags().get('valid_oof', False):
                self._oof_pred_proba = model_base.get_oof_pred_proba(X=X, y=y)
            else:
                logger.warning('\tWARNING: `use_child_oof` was specified but child model does not have a dedicated `get_oof_pred_proba` method. This model may have heavily overfit validation scores.')
                self._oof_pred_proba = model_base.predict_proba(X=X)
            self._child_oof = True
            model_base.predict_time = time.time() - time_start_predict
            model_base.val_score = model_base.score_with_y_pred_proba(y=y, y_pred_proba=self._oof_pred_proba)
        else:
            self._oof_pred_proba = model_base.predict_proba(X=X)  # TODO: Cheater value, will be overfit to valid set
        self._oof_pred_model_repeats = np.ones(shape=len(X), dtype=np.uint8)
        model_base.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
        if not self.params.get('save_bag_folds', True):
            model_base.model = None
        if self.low_memory:
            self.save_child(model_base)
        self.add_child(model=model_base, add_child_times=True)
        self._set_n_repeat_single()

    def _set_n_repeat_single(self):
        """Sets variables that track `n_repeats` to values that represent having only 1 child in the bag."""
        self._n_repeats = 1
        self._n_repeats_finished = 1
        self._k_per_n_repeat = [1]
        self._bagged_mode = False

    def _get_default_fold_fitting_strategy(self):
        # ray not working properly on macos: https://github.com/ray-project/ray/issues/20084
        # TODO: re-enable macos once this issue is addressed
        os_fitting_strategy_map = dict(
            Darwin='sequential_local',
            Windows='parallel_local',
            Linux='parallel_local',
        )
        current_os = platform.system()
        fold_fitting_strategy = os_fitting_strategy_map.get(current_os, 'sequential_local')
        if fold_fitting_strategy == 'sequential_local':
            warning_msg = f'\tWill use sequential fold fitting strategy because {current_os} OS does not yet support parallel folding.'
            dup_filter.attach_filter_targets(warning_msg)
            logger.warning(warning_msg)
        else:
            try:
                try_import_ray()
            except Exception as e:
                warning_msg = f'Will use sequential fold fitting strategy because import of ray failed. Reason: {str(e)}'
                dup_filter.attach_filter_targets(warning_msg)
                logger.warning(warning_msg)
                fold_fitting_strategy = 'sequential_local'
        assert fold_fitting_strategy in ['parallel_local', 'sequential_local']
        return fold_fitting_strategy

    def _fit_folds(self,
                   X,
                   y,
                   model_base,
                   X_pseudo=None,
                   y_pseudo=None,
                   k_fold=None,
                   k_fold_start=0,
                   k_fold_end=None,
                   n_repeats=1,
                   n_repeat_start=0,
                   time_limit=None,
                   sample_weight=None,
                   save_folds=True,
                   groups=None,
                   num_cpus=None,
                   num_gpus=None,
                   **kwargs):
        fold_fitting_strategy = self.params.get('fold_fitting_strategy', 'auto')
        if num_gpus is not None and not isinstance(num_gpus, str):
            # Use a specialized fitting strategy for CPU or GPU models if specified.
            if num_gpus > 0:
                fold_fitting_strategy = self.params.get('fold_fitting_strategy_gpu', fold_fitting_strategy)
            else:
                fold_fitting_strategy = self.params.get('fold_fitting_strategy_cpu', fold_fitting_strategy)
        if fold_fitting_strategy == 'auto':
            fold_fitting_strategy = self._get_default_fold_fitting_strategy()
        num_folds_parallel = self.params.get('num_folds_parallel', 'auto')
        disable_parallel_fitting = self.params.get('_disable_parallel_fitting', False)
        if fold_fitting_strategy == 'parallel_local':
            if disable_parallel_fitting:
                fold_fitting_strategy = SequentialLocalFoldFittingStrategy
                logger.log(20, f'\t{model_base.__class__.__name__} does not support parallel folding yet. Will use sequential folding instead')
            else:
                fold_fitting_strategy = ParallelLocalFoldFittingStrategy
        elif fold_fitting_strategy == 'sequential_local':
            fold_fitting_strategy = SequentialLocalFoldFittingStrategy
        else:
            raise ValueError(
                f'{fold_fitting_strategy} is not a valid option for fold_fitting_strategy'
                'Valid options are: parallel_local and sequential_local'
            )

        # TODO: Preprocess data here instead of repeatedly
        # FIXME: Raise exception if multiclass/binary and a single val fold contains all instances of a class. (Can happen if custom groups is specified)
        time_start = time.time()
        if k_fold_start != 0:
            cv_splitter = self._cv_splitters[n_repeat_start]
        else:
            cv_splitter = self._get_cv_splitter(n_splits=k_fold, n_repeats=n_repeats, groups=groups)
        if k_fold != cv_splitter.n_splits:
            k_fold = cv_splitter.n_splits
        if k_fold_end is None:
            k_fold_end = k_fold
        if cv_splitter.n_repeats < n_repeats:
            # If current cv_splitter doesn't have enough n_repeats for all folds, then create a new one.
            cv_splitter = self._get_cv_splitter(n_splits=k_fold, n_repeats=n_repeats, groups=groups)

        fold_fit_args_list, n_repeats_started, n_repeats_finished = self._generate_fold_configs(
            X=X,
            y=y,
            cv_splitter=cv_splitter,
            k_fold_start=k_fold_start,
            k_fold_end=k_fold_end,
            n_repeat_start=n_repeat_start,
            n_repeat_end=n_repeats,
        )

        fold_fit_args_list = [dict(fold_ctx=fold_ctx) for fold_ctx in fold_fit_args_list]

        oof_pred_proba, oof_pred_model_repeats = self._construct_empty_oof(X=X, y=y)
        models = []

        num_folds = len(fold_fit_args_list)
        if num_folds_parallel == 'auto':
            num_folds_parallel = num_folds
        fold_fitting_strategy_args = dict(
            model_base=model_base, model_base_kwargs=kwargs,
            bagged_ensemble_model=self, X=X, y=y, X_pseudo=X_pseudo, y_pseudo=y_pseudo, sample_weight=sample_weight,
            time_limit=time_limit, time_start=time_start, models=models,
            oof_pred_proba=oof_pred_proba, oof_pred_model_repeats=oof_pred_model_repeats,
            save_folds=save_folds, num_cpus=num_cpus, num_gpus=num_gpus
        )
        # noinspection PyCallingNonCallable
        if fold_fitting_strategy == ParallelLocalFoldFittingStrategy:
            fold_fitting_strategy_args['num_jobs'] = num_folds
            fold_fitting_strategy_args['num_folds_parallel'] = num_folds_parallel
        fold_fitting_strategy = fold_fitting_strategy(**fold_fitting_strategy_args)

        if type(fold_fitting_strategy) == ParallelLocalFoldFittingStrategy and not fold_fitting_strategy.is_mem_sufficient():
            # If memory is not sufficient, fall back to sequential fold strategy
            fold_fitting_strategy: AbstractFoldFittingStrategy = SequentialLocalFoldFittingStrategy(**fold_fitting_strategy_args)
            logger.log(30, f'\tMemory not enough to fit {model_base.__class__.__name__} folds in parallel. Will do sequential fitting instead. '
                           f'\tConsider decreasing folds trained in parallel by passing num_folds_parallel to ag_args_ensemble when calling predictor.fit')

        logger.log(20, f'\tFitting {len(fold_fit_args_list)} child models '
                       f'({fold_fit_args_list[0]["fold_ctx"]["model_name_suffix"]} - {fold_fit_args_list[-1]["fold_ctx"]["model_name_suffix"]}) | '
                       f'Fitting with {fold_fitting_strategy.__class__.__name__}')

        # noinspection PyCallingNonCallable
        for fold_fit_args in fold_fit_args_list:
            fold_fitting_strategy.schedule_fold_model_fit(**fold_fit_args)
        fold_fitting_strategy.after_all_folds_scheduled()

        for model in models:
            # No need to add child times or save child here as this already occurred in the fold_fitting_strategy
            self.add_child(model=model, add_child_times=False)
        self._bagged_mode = True

        if self._oof_pred_proba is None:
            self._oof_pred_proba = oof_pred_proba
            self._oof_pred_model_repeats = oof_pred_model_repeats
        else:
            self._oof_pred_proba += oof_pred_proba
            self._oof_pred_model_repeats += oof_pred_model_repeats

        self._cv_splitters += [cv_splitter for _ in range(n_repeats_started)]
        self._k_per_n_repeat += [k_fold for _ in range(n_repeats_finished)]
        self._n_repeats = n_repeats
        if k_fold == k_fold_end:
            self._k = None
            self._k_fold_end = 0
            self._n_repeats_finished = self._n_repeats
        else:
            self._k = k_fold
            self._k_fold_end = k_fold_end
            self._n_repeats_finished = self._n_repeats - 1

    @staticmethod
    def _generate_fold_configs(*,
                               X,
                               y,
                               cv_splitter,
                               k_fold_start,
                               k_fold_end,
                               n_repeat_start,
                               n_repeat_end) -> (list, int, int):
        """
        Generates fold configs given a cv_splitter, k_fold start-end and n_repeat start-end.
        Fold configs are used by inheritors of AbstractFoldFittingStrategy when fitting fold models.

        Returns a list of fold configs, the number of started repeats, and the number of finished repeats.
        """
        k_fold = cv_splitter.n_splits
        kfolds = cv_splitter.split(X=X, y=y)

        fold_start = n_repeat_start * k_fold + k_fold_start
        fold_end = (n_repeat_end - 1) * k_fold + k_fold_end
        folds_to_fit = fold_end - fold_start

        fold_fit_args_list = []
        n_repeats_started = 0
        n_repeats_finished = 0
        for repeat in range(n_repeat_start, n_repeat_end):  # For each repeat
            is_first_set = repeat == n_repeat_start
            is_last_set = repeat == (n_repeat_end - 1)
            if (not is_first_set) or (k_fold_start == 0):
                n_repeats_started += 1

            fold_in_set_start = k_fold_start if repeat == n_repeat_start else 0
            fold_in_set_end = k_fold_end if is_last_set else k_fold

            for fold_in_set in range(fold_in_set_start, fold_in_set_end):  # For each fold
                fold = fold_in_set + (repeat * k_fold)
                fold_ctx = dict(
                    model_name_suffix=f'S{repeat + 1}F{fold_in_set + 1}',  # S5F3 = 3rd fold of the 5th repeat set
                    fold=kfolds[fold],
                    is_last_fold=fold == (fold_end - 1),
                    folds_to_fit=folds_to_fit,
                    folds_finished=fold - fold_start,
                    folds_left=fold_end - fold,
                )

                fold_fit_args_list.append(fold_ctx)
            if fold_in_set_end == k_fold:
                n_repeats_finished += 1

        assert len(fold_fit_args_list) == folds_to_fit, "fold_fit_args_list is not the expected length!"

        return fold_fit_args_list, n_repeats_started, n_repeats_finished

    # TODO: Augment to generate OOF after shuffling each column in X (Batching), this is the fastest way.
    # TODO: Reduce logging clutter during OOF importance calculation (Currently logs separately for each child)
    # Generates OOF predictions from pre-trained bagged models, assuming X and y are in the same row order as used in .fit(X, y)
    def compute_feature_importance(self,
                                   X,
                                   y,
                                   features=None,
                                   silent=False,
                                   time_limit=None,
                                   is_oof=False,
                                   **kwargs) -> pd.DataFrame:
        if features is None:
            # FIXME: use FULL features (children can have different features)
            features = self.load_child(model=self.models[0]).features
        if not is_oof:
            return super().compute_feature_importance(X, y, features=features, time_limit=time_limit, silent=silent, **kwargs)
        fi_fold_list = []
        model_index = 0
        num_children = len(self.models)
        if time_limit is not None:
            time_limit_per_child = time_limit / num_children
        else:
            time_limit_per_child = None
        if not silent:
            logging_message = f'Computing feature importance via permutation shuffling for {len(features)} features using out-of-fold (OOF) data aggregated across {num_children} child models...'
            if time_limit is not None:
                logging_message = f'{logging_message} Time limit: {time_limit}s...'
            logger.log(20, logging_message)

        time_start = time.time()
        early_stop = False
        children_completed = 0
        log_final_suffix = ''
        for n_repeat, k in enumerate(self._k_per_n_repeat):
            if is_oof:
                if self._child_oof or not self._bagged_mode:
                    raise AssertionError('Model trained with no validation data cannot get feature importance on training data, please specify new test data to compute feature importances (model=%s)' % self.name)
                kfolds = self._cv_splitters[n_repeat].split(X=X, y=y)
                cur_kfolds = kfolds[n_repeat * k:(n_repeat + 1) * k]
            else:
                cur_kfolds = [(None, list(range(len(X))))] * k
            for i, fold in enumerate(cur_kfolds):
                _, test_index = fold
                model = self.load_child(self.models[model_index + i])
                fi_fold = model.compute_feature_importance(X=X.iloc[test_index, :], y=y.iloc[test_index], features=features, time_limit=time_limit_per_child,
                                                           silent=silent, log_prefix='\t', importance_as_list=True, **kwargs)
                fi_fold_list.append(fi_fold)

                children_completed += 1
                if time_limit is not None and children_completed != num_children:
                    time_now = time.time()
                    time_left = time_limit - (time_now - time_start)
                    time_child_average = (time_now - time_start) / children_completed
                    if time_left < (time_child_average * 1.1):
                        log_final_suffix = f' (Early stopping due to lack of time...)'
                        early_stop = True
                        break
            if early_stop:
                break
            model_index += k
        # TODO: DON'T THROW AWAY SAMPLES! USE LARGER N
        fi_list_dict = dict()
        for val in fi_fold_list:
            val = val['importance'].to_dict()  # TODO: Don't throw away stddev information of children
            for key in val:
                if key not in fi_list_dict:
                    fi_list_dict[key] = []
                fi_list_dict[key] += val[key]
        fi_df = _compute_fi_with_stddev(fi_list_dict)

        if not silent:
            logger.log(20, f'\t{round(time.time() - time_start, 2)}s\t= Actual runtime (Completed {children_completed} of {num_children} children){log_final_suffix}')

        return fi_df

    def get_features(self):
        assert self.is_fit(), "The model must be fit before calling the get_features method."
        return self.load_child(self.models[0]).get_features()

    def load_child(self, model: Union[AbstractModel, str], verbose=False) -> AbstractModel:
        if isinstance(model, str):
            child_path = self.create_contexts(self.path + model + os.path.sep)
            return self._child_type.load(path=child_path, verbose=verbose)
        else:
            return model

    def add_child(self, model: Union[AbstractModel, str], add_child_times=False):
        """
        Add a new fit child model to `self.models`

        Parameters
        ----------
        model : Union[AbstractModel, str]
            The child model to add. If str, it is the name of the model.
        add_child_times : bool, default = False
            Whether to add child metadata on times to the bag times.
            This includes fit_time, predict_time, and predict_1_time.
        """
        if self.models is None:
            self.models = []
        if isinstance(model, str):
            model_name = model
            model = None
        else:
            model_name = model.name
        if self.low_memory:
            self.models.append(model_name)
        else:
            if model is None:
                model = self.load_child(model=model_name, verbose=False)
            self.models.append(model)
        if add_child_times:
            if model is None:
                model = self.load_child(model=model_name, verbose=False)
            self._add_child_times_to_bag(model=model)

    def save_child(self, model: Union[AbstractModel, str], path=None, verbose=False):
        """Save child model to disk."""
        if path is None:
            path = self.path
        child = self.load_child(model)
        child.set_contexts(path + child.name + os.path.sep)
        child.save(verbose=verbose)

    def can_compile(self, compiler_configs=None):
        """Check if child models can compile"""
        if not self.is_fit():
            return False
        return self.load_child(self.models[0]).can_compile(compiler_configs=compiler_configs)

    def compile(self, compiler_configs=None):
        """Compile all child models"""
        assert self.is_fit(), "The model must be fit before calling the compile method."
        for child in self.models:
            child = self.load_child(child)
            child.compile(compiler_configs=compiler_configs)
            self.save_child(child)

    def get_compiler_name(self) -> str:
        assert self.is_fit(), "The model must be fit before calling the get_compiler_name method."
        return self.load_child(self.models[0]).get_compiler_name()

    # TODO: Multiply epochs/n_iterations by some value (such as 1.1) to account for having more training data than bagged models
    def convert_to_refit_full_template(self):
        init_args = self.get_params()
        init_args['hyperparameters']['save_bag_folds'] = True  # refit full models must save folds
        init_args['model_base'] = self.convert_to_refit_full_template_child()
        init_args['name'] = init_args['name'] + REFIT_FULL_SUFFIX
        model_full_template = self.__class__(**init_args)
        return model_full_template

    def convert_to_refit_full_template_child(self):
        refit_params_trained = self._get_compressed_params_trained()
        refit_params = copy.deepcopy(self._get_model_base().get_params())
        refit_params['hyperparameters'].update(refit_params_trained)
        refit_child_template = self._child_type(**refit_params)

        return refit_child_template

    def convert_to_refit_full_via_copy(self) -> AbstractModel:
        """
        Creates a new refit_full variant of the model, but instead of training it simply copies `self` while keeping only the first fold model.
        This method is for compatibility with models that have not implemented refit_full support as a fallback.
        """
        if not self.params.get('save_bag_folds', True):
            raise AssertionError('Cannot perform copy-based refit_full when save_bag_folds is False!')
        __models = self.models
        self.models = []
        model_full = copy.deepcopy(self)
        self.models = __models
        child_0 = self.load_child(self.models[0])
        model_full.fit_time = None
        model_full.predict_time = None
        model_full.predict_1_time = None
        model_full.val_score = None
        model_full.rename(model_full.name + REFIT_FULL_SUFFIX)
        if model_full.low_memory:
            model_full.save_child(child_0)
        model_full.add_child(model=child_0, add_child_times=True)
        model_full._set_n_repeat_single()
        return model_full

    def get_params(self):
        init_args = dict(
            model_base=self._get_model_base(),
            random_state=self._random_state,
        )
        init_args.update(super().get_params())
        init_args.pop('eval_metric')
        init_args.pop('problem_type')
        return init_args

    def convert_to_template_child(self):
        return self._get_model_base().convert_to_template()

    def _get_compressed_params(self, model_params_list=None):
        if model_params_list is None:
            model_params_list = [
                self.load_child(child).get_trained_params()
                for child in self.models
            ]

        model_params_compressed = dict()
        for param in model_params_list[0].keys():
            model_param_vals = [model_params[param] for model_params in model_params_list]
            if all(isinstance(val, bool) for val in model_param_vals):
                counter = Counter(model_param_vals)
                compressed_val = counter.most_common(1)[0][0]
            elif all(isinstance(val, int) for val in model_param_vals):
                compressed_val = round(mean(model_param_vals))
            elif all(isinstance(val, float) for val in model_param_vals):
                compressed_val = mean(model_param_vals)
            else:
                try:
                    counter = Counter(model_param_vals)
                    compressed_val = counter.most_common(1)[0][0]
                except TypeError:
                    compressed_val = model_param_vals[0]
            model_params_compressed[param] = compressed_val
        return model_params_compressed

    def _get_compressed_params_trained(self):
        model_params_list = [
            self.load_child(child).params_trained
            for child in self.models
        ]
        return self._get_compressed_params(model_params_list=model_params_list)

    def _get_model_base(self):
        if self.model_base is None:
            return self.load_model_base()
        else:
            return self.model_base

    def _add_child_times_to_bag(self, model):
        if self.fit_time is None:
            self.fit_time = model.fit_time
        else:
            self.fit_time += model.fit_time

        if self.predict_time is None:
            self.predict_time = model.predict_time
        else:
            self.predict_time += model.predict_time

        if self.predict_1_time is None:
            self.predict_1_time = model.predict_1_time
        else:
            self.predict_1_time += model.predict_1_time

    def _add_parallel_child_times(self, fit_time, predict_time, predict_1_time):
        if self.fit_time is None:
            self.fit_time = fit_time
        else:
            self.fit_time += fit_time

        if self.predict_time is None:
            self.predict_time = predict_time
        else:
            self.predict_time += predict_time

        if self.predict_1_time is None:
            self.predict_1_time = predict_1_time
        else:
            self.predict_1_time += predict_1_time

    @classmethod
    def load(cls, path: str, reset_paths=True, low_memory=True, load_oof=False, verbose=True):
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if not low_memory:
            model.persist_child_models(reset_paths=reset_paths)
        if load_oof:
            model._load_oof()
        return model

    @classmethod
    def load_oof(cls, path, verbose=True):
        try:
            oof = load_pkl.load(path=os.path.join(path + 'utils', cls._oof_filename), verbose=verbose)
            oof_pred_proba = oof['_oof_pred_proba']
            oof_pred_model_repeats = oof['_oof_pred_model_repeats']
        except FileNotFoundError:
            model = cls.load(path=path, reset_paths=True, verbose=verbose)
            model._load_oof()
            oof_pred_proba = model._oof_pred_proba
            oof_pred_model_repeats = model._oof_pred_model_repeats
        return cls._oof_pred_proba_func(oof_pred_proba=oof_pred_proba, oof_pred_model_repeats=oof_pred_model_repeats)

    def _load_oof(self):
        if self._oof_pred_proba is not None:
            pass
        else:
            oof = load_pkl.load(path=os.path.join(self.path + 'utils', self._oof_filename))
            self._oof_pred_proba = oof['_oof_pred_proba']
            self._oof_pred_model_repeats = oof['_oof_pred_model_repeats']

    def persist_child_models(self, reset_paths=True):
        for i, model_name in enumerate(self.models):
            if isinstance(model_name, str):
                child_path = self.create_contexts(self.path + model_name + os.path.sep)
                child_model = self._child_type.load(path=child_path, reset_paths=reset_paths, verbose=True)
                self.models[i] = child_model

    def unpersist_child_models(self):
        self.models = self._get_child_model_names(models=self.models)

    def _get_child_model_names(self, models: list) -> list:
        model_names = []
        for i, model in enumerate(models):
            if isinstance(model, str):
                model_names.append(model)
            else:
                model_names.append(model.name)
        return model_names

    def load_model_base(self):
        return load_pkl.load(path=os.path.join(self.path + 'utils', 'model_template.pkl'))

    def save_model_base(self, model_base):
        save_pkl.save(path=os.path.join(self.path + 'utils', 'model_template.pkl'), object=model_base)

    def save(self, path=None, verbose=True, save_oof=True, save_children=False) -> str:
        if path is None:
            path = self.path

        if save_children:
            for child in self.models:
                self.save_child(model=child, path=path, verbose=False)

        if save_oof and self._oof_pred_proba is not None:
            save_pkl.save(path=os.path.join(path + 'utils', self._oof_filename), object={
                '_oof_pred_proba': self._oof_pred_proba,
                '_oof_pred_model_repeats': self._oof_pred_model_repeats,
            })
            self._oof_pred_proba = None
            self._oof_pred_model_repeats = None

        _models = self.models
        if self.low_memory:
            self.models = self._get_child_model_names(self.models)
        path = super().save(path=path, verbose=verbose)
        self.models = _models
        return path

    # If `remove_fit_stack=True`, variables will be removed that are required to fit more folds and to fit new stacker models which use this model as a base model.
    #  This includes OOF variables.
    def reduce_memory_size(self, remove_fit_stack=False, remove_fit=True, remove_info=False, requires_save=True, reduce_children=False, **kwargs):
        super().reduce_memory_size(remove_fit=remove_fit, remove_info=remove_info, requires_save=requires_save, **kwargs)
        if remove_fit_stack:
            try:
                os.remove(os.path.join(self.path + 'utils', self._oof_filename))
            except FileNotFoundError:
                pass
            if requires_save:
                self._oof_pred_proba = None
                self._oof_pred_model_repeats = None
            try:
                os.remove(os.path.join(self.path + 'utils', 'model_template.pkl'))
            except FileNotFoundError:
                pass
            if requires_save:
                self.model_base = None
            try:
                os.rmdir(self.path + 'utils')
            except OSError:
                pass
        if reduce_children:
            for model in self.models:
                model = self.load_child(model)
                model.reduce_memory_size(remove_fit=remove_fit, remove_info=remove_info, requires_save=requires_save, **kwargs)
                if requires_save and self.low_memory:
                    self.save_child(model=model)

    def _get_model_names(self):
        model_names = []
        for model in self.models:
            if isinstance(model, str):
                model_names.append(model)
            else:
                model_names.append(model.name)
        return model_names

    def get_info(self):
        info = super().get_info()
        children_info = self._get_child_info()
        child_memory_sizes = [child['memory_size'] for child in children_info.values()]
        sum_memory_size_child = sum(child_memory_sizes)
        if child_memory_sizes:
            max_memory_size_child = max(child_memory_sizes)
        else:
            max_memory_size_child = 0
        if self.low_memory:
            max_memory_size = info['memory_size'] + sum_memory_size_child
            min_memory_size = info['memory_size'] + max_memory_size_child
        else:
            max_memory_size = info['memory_size']
            min_memory_size = info['memory_size'] - sum_memory_size_child + max_memory_size_child

        # Necessary if save_space is used as save_space deletes model_base.
        if len(self.models) > 0:
            child_model = self.load_child(self.models[0])
        else:
            child_model = self._get_model_base()
        child_hyperparameters = child_model.params
        child_ag_args_fit = child_model.params_aux

        bagged_info = dict(
            child_model_type=self._child_type.__name__,
            num_child_models=len(self.models),
            child_model_names=self._get_model_names(),
            _n_repeats=self._n_repeats,
            # _n_repeats_finished=self._n_repeats_finished,  # commented out because these are too technical
            # _k_fold_end=self._k_fold_end,
            # _k=self._k,
            _k_per_n_repeat=self._k_per_n_repeat,
            _random_state=self._random_state,
            low_memory=self.low_memory,  # If True, then model will attempt to use at most min_memory_size memory by having at most one child in memory. If False, model will use max_memory_size memory.
            bagged_mode=self._bagged_mode,
            max_memory_size=max_memory_size,  # Memory used when all children are loaded into memory at once.
            min_memory_size=min_memory_size,  # Memory used when only the largest child is loaded into memory.
            child_hyperparameters=child_hyperparameters,
            child_hyperparameters_fit=self._get_compressed_params_trained(),
            child_ag_args_fit=child_ag_args_fit,
        )
        info['bagged_info'] = bagged_info
        info['children_info'] = children_info

        child_features_full = list(set().union(*[child['features'] for child in children_info.values()]))
        info['features'] = child_features_full
        info['num_features'] = len(child_features_full)

        return info

    def get_memory_size(self):
        models = self.models
        self.models = None
        memory_size = super().get_memory_size()
        self.models = models
        return memory_size

    def validate_fit_resources(self, **kwargs):
        self._get_model_base().validate_fit_resources(**kwargs)

    def get_minimum_resources(self, **kwargs) -> Dict[str, int]:
        return self._get_model_base().get_minimum_resources(**kwargs)

    def _get_default_resources(self):
        return self._get_model_base()._get_default_resources()

    def _validate_fit_memory_usage(self, **kwargs):
        # memory is checked downstream on the child model
        pass

    def _get_child_info(self):
        child_info_dict = dict()
        for model in self.models:
            if isinstance(model, str):
                child_path = self.create_contexts(self.path + model + os.path.sep)
                child_info_dict[model] = self._child_type.load_info(child_path)
            else:
                child_info_dict[model.name] = model.get_info()
        return child_info_dict

    def _construct_empty_oof(self, X, y):
        if self.problem_type == MULTICLASS:
            oof_pred_proba = np.zeros(shape=(len(X), len(y.unique())), dtype=np.float32)
        elif self.problem_type == SOFTCLASS:
            oof_pred_proba = np.zeros(shape=y.shape, dtype=np.float32)
        elif self.problem_type == QUANTILE:
            oof_pred_proba = np.zeros(shape=(len(X), len(self.quantile_levels)), dtype=np.float32)
        else:
            oof_pred_proba = np.zeros(shape=len(X), dtype=np.float32)
        oof_pred_model_repeats = np.zeros(shape=len(X), dtype=np.uint8)
        return oof_pred_proba, oof_pred_model_repeats

    def _hyperparameter_tune(
        self,
        X,
        y,
        X_val,
        y_val,
        hpo_executor,
        k_fold=None,
        k_fold_end=None,
        **kwargs
    ):
        time_start = time.time()
        logger.log(15, "Starting generic AbstractModel hyperparameter tuning for %s model..." % self.name)
        k_fold, k_fold_end = self._update_k_fold(k_fold=k_fold, k_fold_end=k_fold_end)
        # initialize the model base to get necessary info for search space and estimating memory usage
        initialized_model_base = copy.deepcopy(self.model_base)
        model_init_args = self.model_base.get_params()
        model_init_args['feature_metadata'] = self.feature_metadata
        model_init_args['num_classes'] = self.num_classes
        initialized_model_base.initialize(X=X, y=y, **model_init_args)
        search_space = initialized_model_base._get_search_space()

        try:
            hpo_executor.validate_search_space(search_space, self.name)
        except EmptySearchSpace:
            return skip_hpo(X=X, y=y, X_val=X_val, y_val=y_val, **kwargs)

        # Use absolute path here because ray tune will change the working directory
        self.set_contexts(os.path.abspath(self.path) + os.path.sep)
        directory = self.path  # also create model directory if it doesn't exist
        # TODO: This will break on S3. Use tabular/utils/savers for datasets, add new function
        dataset_train_filename = 'dataset_train.pkl'
        train_path = os.path.join(directory, dataset_train_filename)
        save_pkl.save(path=train_path, object=(X, y))

        dataset_val_filename = 'dataset_val.pkl'
        val_path = os.path.join(directory, dataset_val_filename)
        save_pkl.save(path=val_path, object=(X_val, y_val))

        model_cls = self.__class__
        init_params = copy.deepcopy(self.get_params())
        model_base = self._get_model_base()

        if not inspect.isclass(model_base):
            init_params['model_base'] = init_params['model_base'].__class__
            init_params['model_base_kwargs'] = model_base.get_params()
        # Here the hyperparameters are unprocessed search space.
        # HPO Executor will handle passing in the correct parameters.
        # But we need to keep the ag_args_fit being passed to the base model
        if 'hyperparameters' in init_params['model_base_kwargs']:
            model_base_ag_args_fit = init_params['model_base_kwargs']['hyperparameters'].get('ag_args_fit', {})
            init_params['model_base_kwargs']['hyperparameters'] = {'ag_args_fit': model_base_ag_args_fit}
        # We set soft time limit to avoid trials being terminated directly by ray tune
        trial_soft_time_limit = None
        if hpo_executor.time_limit is not None:
            trial_soft_time_limit = max(hpo_executor.time_limit * 0.9, hpo_executor.time_limit - 5)  # 5 seconds max for buffer

        fit_kwargs = copy.deepcopy(kwargs)
        fit_kwargs['k_fold'] = k_fold
        fit_kwargs['k_fold_end'] = k_fold_end
        fit_kwargs['feature_metadata'] = self.feature_metadata
        fit_kwargs['num_classes'] = self.num_classes
        fit_kwargs['sample_weight'] = kwargs.get('sample_weight', None)
        fit_kwargs['sample_weight_val'] = kwargs.get('sample_weight_val', None)
        fit_kwargs.pop('time_limit', None)  # time_limit already set in hpo_executor
        train_fn_kwargs = dict(
            model_cls=model_cls,
            init_params=init_params,
            time_start=time_start,
            time_limit=trial_soft_time_limit,
            fit_kwargs=fit_kwargs,
            train_path=train_path,
            val_path=val_path,
            hpo_executor=hpo_executor,
            is_bagged_model=True,
        )
            
        minimum_resources_per_fold = self.get_minimum_resources(
            is_gpu_available=(hpo_executor.resources.get('num_gpus', 0) > 0)
        )
        minimum_cpu_per_fold = minimum_resources_per_fold.get('num_cpus', 1)
        minimum_gpu_per_fold = minimum_resources_per_fold.get('num_gpus', 0)

        hpo_executor.execute(
            model_trial=model_trial,
            train_fn_kwargs=train_fn_kwargs,
            directory=directory,
            minimum_cpu_per_trial=minimum_cpu_per_fold,
            minimum_gpu_per_trial=minimum_gpu_per_fold,
            model_estimate_memory_usage=None,  # Not needed as we've already calculated it above
            adapter_type='tabular',
            trainable_is_parallel=True,
        )

        hpo_results = hpo_executor.get_hpo_results(
            model_name=self.name,
            model_path_root=self.path_root,
            time_start=time_start,
        )

        return hpo_results

    def _more_tags(self):
        return {
            'valid_oof': True,
            'can_refit_full': True,
        }

    def _get_tags_child(self):
        """Gets the tags of the child model."""
        return self._get_model_base()._get_tags()
