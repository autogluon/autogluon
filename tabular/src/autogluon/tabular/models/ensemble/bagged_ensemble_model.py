import copy
import logging
import os
import time
from collections import Counter
from statistics import mean
from functools import reduce

import numpy as np
import pandas as pd

from autogluon.core.constants import MULTICLASS, REGRESSION, SOFTCLASS, REFIT_FULL_SUFFIX
from autogluon.core.utils.utils import generate_kfold
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils.utils import _compute_fi_with_stddev

from autogluon.core.models import AbstractModel

logger = logging.getLogger(__name__)


# TODO: Add metadata object with info like score on each model, train time on each model, etc.
class BaggedEnsembleModel(AbstractModel):
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
        self.bagged_mode = None

        try:
            feature_metadata = self.model_base.feature_metadata
        except:
            feature_metadata = None

        eval_metric = kwargs.pop('eval_metric', self.model_base.eval_metric)
        stopping_metric = kwargs.pop('stopping_metric', self.model_base.stopping_metric)

        super().__init__(problem_type=self.model_base.problem_type, eval_metric=eval_metric, stopping_metric=stopping_metric, feature_metadata=feature_metadata, **kwargs)

    def _set_default_params(self):
        default_params = {'save_bag_folds': True}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()

    def is_valid(self):
        return self.is_fit() and (self._n_repeats == self._n_repeats_finished)

    def can_infer(self):
        return self.is_fit() and self.params.get('save_bag_folds', True)

    def is_stratified(self):
        if self.problem_type == REGRESSION or self.problem_type == SOFTCLASS:
            return False
        else:
            return True

    def is_fit(self):
        return len(self.models) != 0

    # TODO: This assumes bagged ensemble has a complete k_fold and no partial k_fold models, this is likely fine but will act incorrectly if called when only a partial k_fold has been completed
    #  Solving this is memory intensive, requires all oof_pred_probas from all n_repeats, so its probably not worth it.
    @property
    def oof_pred_proba(self):
        # TODO: Require is_valid == True (add option param to ignore is_valid)
        return self._oof_pred_proba_func(self._oof_pred_proba, self._oof_pred_model_repeats)

    @staticmethod
    def _oof_pred_proba_func(oof_pred_proba, oof_pred_model_repeats):
        oof_pred_model_repeats_without_0 = np.where(oof_pred_model_repeats == 0, 1, oof_pred_model_repeats)
        if oof_pred_proba.ndim == 2:
            oof_pred_model_repeats_without_0 = oof_pred_model_repeats_without_0[:, None]
        return oof_pred_proba / oof_pred_model_repeats_without_0

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

    def _fit(self, X_train, y_train, k_fold=5, k_fold_start=0, k_fold_end=None, n_repeats=1, n_repeat_start=0, time_limit=None, **kwargs):
        if k_fold < 1:
            k_fold = 1
        if k_fold_end is None:
            k_fold_end = k_fold

        if self._oof_pred_proba is None and (k_fold_start != 0 or n_repeat_start != 0):
            self._load_oof()
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
        fold_start = n_repeat_start * k_fold + k_fold_start
        fold_end = (n_repeats - 1) * k_fold + k_fold_end
        time_start = time.time()

        model_base = self._get_model_base()
        if self.features is not None:
            model_base.features = self.features
        model_base.feature_metadata = self.feature_metadata  # TODO: Don't pass this here

        if self.model_base is not None:
            self.save_model_base(self.model_base)
            self.model_base = None

        if k_fold == 1:
            if self._n_repeats != 0:
                raise ValueError(f'n_repeats must equal 0 when fitting a single model with k_fold < 2, values: ({self._n_repeats}, {k_fold})')
            model_base.set_contexts(path_context=self.path + model_base.name + os.path.sep)
            time_start_fit = time.time()
            model_base.fit(X_train=X_train, y_train=y_train, time_limit=time_limit, **kwargs)
            model_base.fit_time = time.time() - time_start_fit
            model_base.predict_time = None
            self._oof_pred_proba = model_base.predict_proba(X=X_train)  # TODO: Cheater value, will be overfit to valid set
            self._oof_pred_model_repeats = np.ones(shape=len(X_train), dtype=np.uint8)
            self._n_repeats = 1
            self._n_repeats_finished = 1
            self._k_per_n_repeat = [1]
            self.bagged_mode = False
            model_base.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
            if not self.params.get('save_bag_folds', True):
                model_base.model = None
            if self.low_memory:
                self.save_child(model_base, verbose=False)
                self.models = [model_base.name]
            else:
                self.models = [model_base]
            self._add_child_times_to_bag(model=model_base)
            return

        # TODO: Preprocess data here instead of repeatedly
        kfolds = generate_kfold(X=X_train, y=y_train, n_splits=k_fold, stratified=self.is_stratified(), random_state=self._random_state, n_repeats=n_repeats)

        oof_pred_proba, oof_pred_model_repeats = self._construct_empty_oof(X=X_train, y=y_train)

        models = []
        folds_to_fit = fold_end - fold_start
        for j in range(n_repeat_start, n_repeats):  # For each n_repeat
            cur_repeat_count = j - n_repeat_start
            fold_start_n_repeat = fold_start + cur_repeat_count * k_fold
            fold_end_n_repeat = min(fold_start_n_repeat + k_fold, fold_end)
            # TODO: Consider moving model fit inner for loop to a function to simply this code
            for i in range(fold_start_n_repeat, fold_end_n_repeat):  # For each fold
                folds_finished = i - fold_start
                folds_left = fold_end - i
                fold = kfolds[i]
                time_elapsed = time.time() - time_start
                if time_limit is not None:
                    time_left = time_limit - time_elapsed
                    required_time_per_fold = time_left / folds_left
                    time_limit_fold = required_time_per_fold * 0.8
                    if folds_finished > 0:
                        expected_time_required = time_elapsed * folds_to_fit / folds_finished
                        expected_remaining_time_required = expected_time_required * folds_left / folds_to_fit
                        if expected_remaining_time_required > time_left:
                            raise TimeLimitExceeded
                    if time_left <= 0:
                        raise TimeLimitExceeded
                else:
                    time_limit_fold = None

                time_start_fold = time.time()
                train_index, val_index = fold
                X_train_fold, X_val_fold = X_train.iloc[train_index, :], X_train.iloc[val_index, :]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
                fold_model = copy.deepcopy(model_base)
                fold_model.name = f'{fold_model.name}_fold_{i}'
                fold_model.set_contexts(self.path + fold_model.name + os.path.sep)
                fold_model.fit(X_train=X_train_fold, y_train=y_train_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **kwargs)
                time_train_end_fold = time.time()
                if time_limit is not None:  # Check to avoid unnecessarily predicting and saving a model when an Exception is going to be raised later
                    if i != (fold_end - 1):
                        time_elapsed = time.time() - time_start
                        time_left = time_limit - time_elapsed
                        expected_time_required = time_elapsed * folds_to_fit / (folds_finished + 1)
                        expected_remaining_time_required = expected_time_required * (folds_left - 1) / folds_to_fit
                        if expected_remaining_time_required > time_left:
                            raise TimeLimitExceeded
                pred_proba = fold_model.predict_proba(X_val_fold)
                time_predict_end_fold = time.time()
                fold_model.fit_time = time_train_end_fold - time_start_fold
                fold_model.predict_time = time_predict_end_fold - time_train_end_fold
                fold_model.val_score = fold_model.score_with_y_pred_proba(y=y_val_fold, y_pred_proba=pred_proba)
                fold_model.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
                if not self.params.get('save_bag_folds', True):
                    fold_model.model = None
                if self.low_memory:
                    self.save_child(fold_model, verbose=False)
                    models.append(fold_model.name)
                else:
                    models.append(fold_model)
                oof_pred_proba[val_index] += pred_proba
                oof_pred_model_repeats[val_index] += 1
                self._add_child_times_to_bag(model=fold_model)
            if (fold_end_n_repeat != fold_end) or (k_fold == k_fold_end):
                self._k_per_n_repeat.append(k_fold)
        self.models += models

        self.bagged_mode = True

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

    def score_with_oof(self, y):
        self._load_oof()
        valid_indices = self._oof_pred_model_repeats > 0
        y = y[valid_indices]
        y_pred_proba = self.oof_pred_proba[valid_indices]

        return self.score_with_y_pred_proba(y=y, y_pred_proba=y_pred_proba)

    # TODO: Augment to generate OOF after shuffling each column in X (Batching), this is the fastest way.
    # TODO: v0.1 Reduce logging clutter during OOF importance calculation (Currently logs separately for each child)
    # Generates OOF predictions from pre-trained bagged models, assuming X and y are in the same row order as used in .fit(X, y)
    def compute_feature_importance(self, X, y, features=None, is_oof=True, time_limit=None, silent=False, **kwargs) -> pd.DataFrame:
        if features is None:
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
                if not self.bagged_mode:
                    raise AssertionError('Model trained with no validation data cannot get feature importances on training data, please specify new test data to compute feature importances (model=%s)' % self.name)
                kfolds = generate_kfold(X=X, y=y, n_splits=k, stratified=self.is_stratified(), random_state=self._random_state, n_repeats=n_repeat + 1)
                cur_kfolds = kfolds[n_repeat * k:(n_repeat+1) * k]
            else:
                cur_kfolds = [(None, list(range(len(X))))]*k
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
        init_args = self._get_init_args()
        init_args['hyperparameters']['save_bag_folds'] = True  # refit full models must save folds
        model_base_name_orig = init_args['model_base'].name
        init_args['model_base'] = self.convert_to_refitfull_template_child()
        model_base_name_new = init_args['model_base'].name
        if model_base_name_orig in init_args['name'] and model_base_name_orig != model_base_name_new:
            init_args['name'] = init_args['name'].replace(model_base_name_orig, model_base_name_new, 1)
        else:
            init_args['name'] = init_args['name'] + '_FULL'

        model_full_template = self.__class__(**init_args)
        return model_full_template

    def convert_to_refitfull_template_child(self):
        compressed_params = self._get_compressed_params()
        child_compressed = copy.deepcopy(self._get_model_base())
        child_compressed.feature_metadata = self.feature_metadata  # TODO: Don't pass this here
        child_compressed.params = compressed_params
        child_compressed.name = child_compressed.name + REFIT_FULL_SUFFIX
        child_compressed.set_contexts(self.path_root + child_compressed.name + os.path.sep)
        return child_compressed

    def _get_init_args(self):
        init_args = dict(
            model_base=self._get_model_base(),
            random_state=self._random_state,
        )
        init_args.update(super()._get_init_args())
        init_args.pop('problem_type')
        init_args.pop('feature_metadata')
        return init_args

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
            bagged_mode=self.bagged_mode,
            max_memory_size=max_memory_size,  # Memory used when all children are loaded into memory at once.
            min_memory_size=min_memory_size,  # Memory used when only the largest child is loaded into memory.
            child_hyperparameters=self._get_model_base().params,
            child_hyperparameters_fit = self._get_compressed_params_trained(),
            child_ag_args_fit = self._get_model_base().params_aux,
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
        else:
            oof_pred_proba = np.zeros(shape=len(X), dtype=np.float32)
        oof_pred_model_repeats = np.zeros(shape=len(X), dtype=np.uint8)
        return oof_pred_proba, oof_pred_model_repeats

    def _preprocess_fit_resources(self, **kwargs):
        """Pass along to child models to avoid altering up-front"""
        return kwargs

    # TODO: Currently double disk usage, saving model in HPO and also saving model in bag
    def _hyperparameter_tune(self, X_train, y_train, k_fold, scheduler_options, preprocess_kwargs=None, **kwargs):
        if len(self.models) != 0:
            raise ValueError('self.models must be empty to call hyperparameter_tune, value: %s' % self.models)

        self.model_base.feature_metadata = self.feature_metadata  # TODO: Move this

        # TODO: Preprocess data here instead of repeatedly
        if preprocess_kwargs is None:
            preprocess_kwargs = dict()
        X_train = self.preprocess(X=X_train, preprocess=False, fit=True, **preprocess_kwargs)
        kfolds = generate_kfold(X=X_train, y=y_train, n_splits=k_fold, stratified=self.is_stratified(), random_state=self._random_state, n_repeats=1)

        train_index, test_index = kfolds[0]
        X_train_fold, X_val_fold = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        orig_time = scheduler_options[1]['time_out']
        scheduler_options[1]['time_out'] = orig_time * 0.8  # TODO: Scheduler doesn't early stop on final model, this is a safety net. Scheduler should be updated to early stop
        hpo_models, hpo_model_performances, hpo_results = self.model_base.hyperparameter_tune(X_train=X_train_fold, y_train=y_train_fold, X_val=X_val_fold, y_val=y_val_fold, scheduler_options=scheduler_options, **kwargs)
        scheduler_options[1]['time_out'] = orig_time

        bags = {}
        bags_performance = {}
        for i, (model_name, model_path) in enumerate(hpo_models.items()):
            child: AbstractModel = self._child_type.load(path=model_path)
            y_pred_proba = child.predict_proba(X_val_fold)

            # TODO: Create new Ensemble Here
            bag = copy.deepcopy(self)
            bag.name = bag.name + os.path.sep + str(i)
            bag.set_contexts(self.path_root + bag.name + os.path.sep)

            oof_pred_proba, oof_pred_model_repeats = self._construct_empty_oof(X=X_train, y=y_train)
            oof_pred_proba[test_index] += y_pred_proba
            oof_pred_model_repeats[test_index] += 1

            bag.model_base = None
            child.set_contexts(bag.path + child.name + os.path.sep)
            bag.save_model_base(child.convert_to_template())

            bag._k = k_fold
            bag._k_fold_end = 1
            bag._n_repeats = 1
            bag._oof_pred_proba = oof_pred_proba
            bag._oof_pred_model_repeats = oof_pred_model_repeats
            child.name = child.name + '_fold_0'
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

            bag.save()
            bags[bag.name] = bag.path
            bags_performance[bag.name] = bag.val_score

        # TODO: hpo_results likely not correct because no renames
        return bags, bags_performance, hpo_results
