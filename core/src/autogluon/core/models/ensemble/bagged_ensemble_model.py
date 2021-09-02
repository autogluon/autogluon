import copy
import logging
import os
import time
from collections import Counter
from statistics import mean

import numpy as np
import pandas as pd

from .fold_fitting_strategy import AbstractFoldFittingStrategy, SequentialLocalFoldFittingStrategy
from ..abstract.abstract_model import AbstractModel
from ...constants import MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE, REFIT_FULL_SUFFIX
from ...utils.exceptions import TimeLimitExceeded
from ...utils.loaders import load_pkl
from ...utils.savers import save_pkl
from ...utils.utils import CVSplitter, _compute_fi_with_stddev

logger = logging.getLogger(__name__)


# TODO: Add metadata object with info like score on each model, train time on each model, etc.
class BaggedEnsembleModel(AbstractModel):
    """
    Bagged ensemble meta-model which fits a given model multiple times across different splits of the training data.

    For certain child models such as KNN, this may only train a single model and instead rely on the child model to generate out-of-fold predictions.
    """
    _oof_filename = 'oof.pkl'

    def __init__(self, model_base: AbstractModel, random_state=0, **kwargs):
        self.model_base = model_base
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

        super().__init__(problem_type=self.model_base.problem_type, eval_metric=self.model_base.eval_metric, **kwargs)

    def _set_default_params(self):
        default_params = {
            # 'use_child_oof': False,  # [Advanced] Whether to defer to child model for OOF preds and only train a single child.
            'save_bag_folds': True,
            # 'refit_folds': False,  # [Advanced, Experimental] Whether to refit bags immediately to a refit_full model in a single .fit call.
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
        return not self.is_fit() or self._bagged_mode

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
        if k_fold is None and groups is None:
            k_fold = 5
        if k_fold is not None and k_fold < 1:
            k_fold = 1
        if k_fold is None or k_fold > 1:
            k_fold = self._get_cv_splitter(n_splits=k_fold, n_repeats=n_repeats, groups=groups).n_splits
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
            self._fit_folds(X=X, y=y, model_base=model_base, k_fold=k_fold, k_fold_start=k_fold_start, k_fold_end=k_fold_end,
                            n_repeats=n_repeats, n_repeat_start=n_repeat_start, save_folds=save_bag_folds, groups=groups, **kwargs)
            # FIXME: Don't save folds except for refit
            # FIXME: Cleanup self
            # FIXME: Don't add `_FULL` to name
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
            #  Currently this only fixes problem when in bagged mode, if not bagging, then inference could still be problamatic
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
        self._n_repeats = 1
        self._n_repeats_finished = 1
        self._k_per_n_repeat = [1]
        self._bagged_mode = False
        model_base.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
        if not self.params.get('save_bag_folds', True):
            model_base.model = None
        if self.low_memory:
            self.save_child(model_base, verbose=False)
            self.models = [model_base.name]
        else:
            self.models = [model_base]
        self._add_child_times_to_bag(model=model_base)

    def _fit_folds(self,
                   X,
                   y,
                   model_base,
                   k_fold=None,
                   k_fold_start=0,
                   k_fold_end=None,
                   n_repeats=1,
                   n_repeat_start=0,
                   time_limit=None,
                   sample_weight=None,
                   save_folds=True,
                   groups=None,
                   **kwargs):
        fold_fitting_strategy = self.params.get('fold_fitting_strategy', SequentialLocalFoldFittingStrategy)
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
        kfolds = cv_splitter.split(X=X, y=y)

        oof_pred_proba, oof_pred_model_repeats = self._construct_empty_oof(X=X, y=y)

        models = []
        fold_start = n_repeat_start * k_fold + k_fold_start
        fold_end = (n_repeats - 1) * k_fold + k_fold_end
        folds_to_fit = fold_end - fold_start
        # noinspection PyCallingNonCallable
        fold_fitting_strategy: AbstractFoldFittingStrategy = fold_fitting_strategy(
            self, X, y, sample_weight, time_limit, time_start, models, oof_pred_proba, oof_pred_model_repeats, save_folds=save_folds)
        for j in range(n_repeat_start, n_repeats):  # For each n_repeat
            if j != n_repeat_start or k_fold_start == 0:
                self._cv_splitters.append(cv_splitter)
            cur_repeat_count = j - n_repeat_start
            fold_start_n_repeat = fold_start + cur_repeat_count * k_fold
            fold_end_n_repeat = min(fold_start_n_repeat + k_fold, fold_end)

            for i in range(fold_start_n_repeat, fold_end_n_repeat):  # For each fold
                fold_num_in_repeat = i - (j * k_fold)  # The fold in the current repeat set (first fold in set = 0)

                fold_ctx = dict(
                    model_name_suffix=f'S{j + 1}F{fold_num_in_repeat + 1}',  # S5F3 = 3rd fold of the 5th repeat set
                    fold=kfolds[i],
                    is_last_fold=i != (fold_end - 1),
                    folds_to_fit=folds_to_fit,
                    folds_finished=i - fold_start,
                    folds_left=fold_end - i,
                )

                fold_fitting_strategy.schedule_fold_model_fit(model_base, fold_ctx, kwargs)
            if (fold_end_n_repeat != fold_end) or (k_fold == k_fold_end):
                self._k_per_n_repeat.append(k_fold)
        fold_fitting_strategy.after_all_folds_scheduled()
        self.models += models

        self._bagged_mode = True

        if self._oof_pred_proba is None:
            self._oof_pred_proba = oof_pred_proba
            self._oof_pred_model_repeats = oof_pred_model_repeats
        else:
            self._oof_pred_proba += oof_pred_proba
            self._oof_pred_model_repeats += oof_pred_model_repeats

        self._n_repeats = n_repeats
        if k_fold == k_fold_end:
            self._k = None
            self._k_fold_end = 0
            self._n_repeats_finished = self._n_repeats
        else:
            self._k = k_fold
            self._k_fold_end = k_fold_end
            self._n_repeats_finished = self._n_repeats - 1

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
                    raise AssertionError('Model trained with no validation data cannot get feature importances on training data, please specify new test data to compute feature importances (model=%s)' % self.name)
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

    def load_child(self, model, verbose=False) -> AbstractModel:
        if isinstance(model, str):
            child_path = self.create_contexts(self.path + model + os.path.sep)
            return self._child_type.load(path=child_path, verbose=verbose)
        else:
            return model

    def save_child(self, model, verbose=False):
        child = self.load_child(model)
        child.set_contexts(self.path + child.name + os.path.sep)
        child.save(verbose=verbose)

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
            oof = load_pkl.load(path=path + 'utils' + os.path.sep + cls._oof_filename, verbose=verbose)
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
            oof = load_pkl.load(path=self.path + 'utils' + os.path.sep + self._oof_filename)
            self._oof_pred_proba = oof['_oof_pred_proba']
            self._oof_pred_model_repeats = oof['_oof_pred_model_repeats']

    def persist_child_models(self, reset_paths=True):
        for i, model_name in enumerate(self.models):
            if isinstance(model_name, str):
                child_path = self.create_contexts(self.path + model_name + os.path.sep)
                child_model = self._child_type.load(path=child_path, reset_paths=reset_paths, verbose=True)
                self.models[i] = child_model

    def load_model_base(self):
        return load_pkl.load(path=self.path + 'utils' + os.path.sep + 'model_template.pkl')

    def save_model_base(self, model_base):
        save_pkl.save(path=self.path + 'utils' + os.path.sep + 'model_template.pkl', object=model_base)

    def save(self, path=None, verbose=True, save_oof=True, save_children=False) -> str:
        if path is None:
            path = self.path

        if save_children:
            model_names = []
            for child in self.models:
                child = self.load_child(child)
                child.set_contexts(path + child.name + os.path.sep)
                child.save(verbose=False)
                model_names.append(child.name)
            self.models = model_names

        if save_oof and self._oof_pred_proba is not None:
            save_pkl.save(path=path + 'utils' + os.path.sep + self._oof_filename, object={
                '_oof_pred_proba': self._oof_pred_proba,
                '_oof_pred_model_repeats': self._oof_pred_model_repeats,
            })
            self._oof_pred_proba = None
            self._oof_pred_model_repeats = None

        return super().save(path=path, verbose=verbose)

    # If `remove_fit_stack=True`, variables will be removed that are required to fit more folds and to fit new stacker models which use this model as a base model.
    #  This includes OOF variables.
    def reduce_memory_size(self, remove_fit_stack=False, remove_fit=True, remove_info=False, requires_save=True, reduce_children=False, **kwargs):
        super().reduce_memory_size(remove_fit=remove_fit, remove_info=remove_info, requires_save=requires_save, **kwargs)
        if remove_fit_stack:
            try:
                os.remove(self.path + 'utils' + os.path.sep + self._oof_filename)
            except FileNotFoundError:
                pass
            if requires_save:
                self._oof_pred_proba = None
                self._oof_pred_model_repeats = None
            try:
                os.remove(self.path + 'utils' + os.path.sep + 'model_template.pkl')
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

    def _preprocess_fit_resources(self, silent=False, **kwargs):
        """Pass along to child models to avoid altering up-front"""
        return kwargs

    # TODO: Currently double disk usage, saving model in HPO and also saving model in bag
    # FIXME: with use_bag_holdout=True, the fold-1 scores that are logged are of the inner validation score, not the holdout score.
    #  Fix this by passing X_val, y_val into this method
    def _hyperparameter_tune(self, X, y, k_fold, scheduler_options, preprocess_kwargs=None, groups=None, **kwargs):
        if len(self.models) != 0:
            raise ValueError('self.models must be empty to call hyperparameter_tune, value: %s' % self.models)

        kwargs['feature_metadata'] = self.feature_metadata
        kwargs['num_classes'] = self.num_classes  # TODO: maybe don't pass num_classes to children
        self.model_base.set_contexts(self.path + 'hpo' + os.path.sep)

        # TODO: Preprocess data here instead of repeatedly
        if preprocess_kwargs is None:
            preprocess_kwargs = dict()
        use_child_oof = self.params.get('use_child_oof', False)
        X = self.preprocess(X=X, preprocess=False, fit=True, **preprocess_kwargs)

        if use_child_oof:
            k_fold = 1
            X_fold = X
            y_fold = y
            X_val_fold = None
            y_val_fold = None
            train_index = list(range(len(X)))
            test_index = train_index
            cv_splitter = None
        else:
            cv_splitter = self._get_cv_splitter(n_splits=k_fold, n_repeats=1, groups=groups)
            if k_fold != cv_splitter.n_splits:
                k_fold = cv_splitter.n_splits

            kfolds = cv_splitter.split(X=X, y=y)

            train_index, test_index = kfolds[0]
            X_fold, X_val_fold = X.iloc[train_index, :], X.iloc[test_index, :]
            y_fold, y_val_fold = y.iloc[train_index], y.iloc[test_index]
        orig_time = scheduler_options[1]['time_out']
        if orig_time:
            scheduler_options[1]['time_out'] = orig_time * 0.8  # TODO: Scheduler doesn't early stop on final model, this is a safety net. Scheduler should be updated to early stop
        hpo_models, hpo_model_performances, hpo_results = self.model_base.hyperparameter_tune(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, scheduler_options=scheduler_options, **kwargs)
        scheduler_options[1]['time_out'] = orig_time

        bags = {}
        bags_performance = {}
        for i, (model_name, model_path) in enumerate(hpo_models.items()):
            child: AbstractModel = self._child_type.load(path=model_path)

            # TODO: Create new Ensemble Here
            bag = copy.deepcopy(self)
            bag.rename(f"{bag.name}{os.path.sep}T{i}")
            bag.set_contexts(self.path_root + bag.name + os.path.sep)

            oof_pred_proba, oof_pred_model_repeats = self._construct_empty_oof(X=X, y=y)

            if child._get_tags().get('valid_oof', False):
                y_pred_proba = child.get_oof_pred_proba(X=X, y=y)
                bag._n_repeats_finished = 1
                bag._k_per_n_repeat = [1]
                bag._bagged_mode = False
                bag._child_oof = True  # TODO: Consider a separate tag for refit_folds vs efficient OOF
            else:
                y_pred_proba = child.predict_proba(X_val_fold)

            oof_pred_proba[test_index] += y_pred_proba
            oof_pred_model_repeats[test_index] += 1

            bag.model_base = None
            child.rename('')
            child.set_contexts(bag.path + child.name + os.path.sep)
            bag.save_model_base(child.convert_to_template())

            bag._k = k_fold
            bag._k_fold_end = 1
            bag._n_repeats = 1
            bag._oof_pred_proba = oof_pred_proba
            bag._oof_pred_model_repeats = oof_pred_model_repeats
            child.rename('S1F1')
            child.set_contexts(bag.path + child.name + os.path.sep)
            if not self.params.get('save_bag_folds', True):
                child.model = None
            if bag.low_memory:
                bag.save_child(child, verbose=False)
                bag.models.append(child.name)
            else:
                bag.models.append(child)
            bag.val_score = child.val_score
            bag._add_child_times_to_bag(model=child)
            if cv_splitter is not None:
                bag._cv_splitters = [cv_splitter]

            bag.save()
            bags[bag.name] = bag.path
            bags_performance[bag.name] = bag.val_score

        # TODO: hpo_results likely not correct because no renames
        return bags, bags_performance, hpo_results

    def _more_tags(self):
        return {'valid_oof': True}
