import copy
import logging
import os
import time
from collections import Counter
from statistics import mean

import numpy as np
import pandas as pd

from ..abstract.abstract_model import AbstractModel
from ...constants import MULTICLASS, REGRESSION, SOFTCLASS, REFIT_FULL_SUFFIX
from ...utils import generate_kfold
from ....utils.exceptions import TimeLimitExceeded
from ....utils.loaders import load_pkl
from ....utils.savers import save_pkl

logger = logging.getLogger(__name__)


# TODO: Add metadata object with info like score on each model, train time on each model, etc.
class BaggedEnsembleModel(AbstractModel):
    def __init__(self, path: str, name: str, model_base: AbstractModel, hyperparameters=None, objective_func=None, stopping_metric=None, random_state=0, debug=0):
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
            feature_types_metadata = self.model_base.feature_types_metadata
        except:
            feature_types_metadata = None

        if objective_func is None:
            objective_func = self.model_base.objective_func
        if stopping_metric is None:
            stopping_metric = self.model_base.stopping_metric

        super().__init__(path=path, name=name, problem_type=self.model_base.problem_type, objective_func=objective_func, stopping_metric=stopping_metric, feature_types_metadata=feature_types_metadata, hyperparameters=hyperparameters, debug=debug)

    def is_valid(self):
        return self.is_fit() and (self._n_repeats == self._n_repeats_finished)

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
        oof_pred_model_repeats_without_0 = np.where(self._oof_pred_model_repeats == 0, 1, self._oof_pred_model_repeats)
        if self._oof_pred_proba.ndim == 2:
            oof_pred_model_repeats_without_0 = oof_pred_model_repeats_without_0[:, None]
        return self._oof_pred_proba / oof_pred_model_repeats_without_0

    def preprocess(self, X, model=None):
        if model is None:
            if not self.models:
                return X
            model = self.models[0]
        model = self.load_child(model)
        return model.preprocess(X)

    def fit(self, X, y, k_fold=5, k_fold_start=0, k_fold_end=None, n_repeats=1, n_repeat_start=0, time_limit=None, **kwargs):
        if k_fold < 1:
            k_fold = 1
        if k_fold_end is None:
            k_fold_end = k_fold

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
        model_base.feature_types_metadata = self.feature_types_metadata  # TODO: Don't pass this here

        if self.model_base is not None:
            self.save_model_base(self.model_base)
            self.model_base = None

        if k_fold == 1:
            if self._n_repeats != 0:
                raise ValueError(f'n_repeats must equal 0 when fitting a single model with k_fold < 2, values: ({self._n_repeats}, {k_fold})')
            model_base.set_contexts(path_context=self.path + model_base.name + os.path.sep)
            time_start_fit = time.time()
            model_base.fit(X_train=X, Y_train=y, time_limit=time_limit, **kwargs)
            model_base.fit_time = time.time() - time_start_fit
            model_base.predict_time = None
            self._oof_pred_proba = model_base.predict_proba(X=X)  # TODO: Cheater value, will be overfit to valid set
            self._oof_pred_model_repeats = np.ones(shape=len(X))
            self._n_repeats = 1
            self._n_repeats_finished = 1
            self._k_per_n_repeat = [1]
            self.bagged_mode = False
            if self.low_memory:
                self.save_child(model_base, verbose=False)
                self.models = [model_base.name]
            else:
                self.models = [model_base]
            self._add_child_times_to_bag(model=model_base)
            return

        # TODO: Preprocess data here instead of repeatedly
        kfolds = generate_kfold(X=X, y=y, n_splits=k_fold, stratified=self.is_stratified(), random_state=self._random_state, n_repeats=n_repeats)

        if self.problem_type == MULTICLASS:
            oof_pred_proba = np.zeros(shape=(len(X), len(y.unique())))
        elif self.problem_type == SOFTCLASS:
            oof_pred_proba = np.zeros(shape=y.shape)
        else:
            oof_pred_proba = np.zeros(shape=len(X))
        oof_pred_model_repeats = np.zeros(shape=len(X))

        models = []
        folds_to_fit = fold_end - fold_start
        for j in range(n_repeat_start, n_repeats):  # For each n_repeat
            cur_repeat_count = j - n_repeat_start
            fold_start_n_repeat = fold_start + cur_repeat_count * k_fold
            fold_end_n_repeat = min(fold_start_n_repeat + k_fold, fold_end)
            is_training_from_start = fold_end_n_repeat - fold_start_n_repeat == k_fold
            if is_training_from_start:
                self._k_per_n_repeat.append(k_fold)
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
                train_index, test_index = fold
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                fold_model = copy.deepcopy(model_base)
                fold_model.name = f'{fold_model.name}_fold_{i}'
                fold_model.set_contexts(self.path + fold_model.name + os.path.sep)
                fold_model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, time_limit=time_limit_fold, **kwargs)
                time_train_end_fold = time.time()
                if time_limit is not None:  # Check to avoid unnecessarily predicting and saving a model when an Exception is going to be raised later
                    if i != (fold_end - 1):
                        time_elapsed = time.time() - time_start
                        time_left = time_limit - time_elapsed
                        expected_time_required = time_elapsed * folds_to_fit / (folds_finished + 1)
                        expected_remaining_time_required = expected_time_required * (folds_left - 1) / folds_to_fit
                        if expected_remaining_time_required > time_left:
                            raise TimeLimitExceeded
                pred_proba = fold_model.predict_proba(X_test)
                time_predict_end_fold = time.time()
                fold_model.fit_time = time_train_end_fold - time_start_fold
                fold_model.predict_time = time_predict_end_fold - time_train_end_fold
                fold_model.val_score = fold_model.score_with_y_pred_proba(y=y_test, y_pred_proba=pred_proba)
                if self.low_memory:
                    self.save_child(fold_model, verbose=False)
                    models.append(fold_model.name)
                else:
                    models.append(fold_model)
                oof_pred_proba[test_index] += pred_proba
                oof_pred_model_repeats[test_index] += 1
                self._add_child_times_to_bag(model=fold_model)
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

    # FIXME: Defective if model does not apply same preprocessing in all bags!
    #  No model currently violates this rule, but in future it could happen
    def predict_proba(self, X, preprocess=True):
        model = self.load_child(self.models[0])
        if preprocess:
            X = self.preprocess(X, model=model)

        pred_proba = model.predict_proba(X=X, preprocess=False)
        for model in self.models[1:]:
            model = self.load_child(model)
            pred_proba += model.predict_proba(X=X, preprocess=False)
        pred_proba = pred_proba / len(self.models)

        return pred_proba

    def score_with_oof(self, y):
        valid_indices = self._oof_pred_model_repeats > 0
        y = y[valid_indices]
        y_pred_proba = self.oof_pred_proba[valid_indices]

        return self.score_with_y_pred_proba(y=y, y_pred_proba=y_pred_proba)

    # TODO: Augment to generate OOF after shuffling each column in X (Batching), this is the fastest way.
    # Generates OOF predictions from pre-trained bagged models, assuming X and y are in the same row order as used in .fit(X, y)
    def compute_feature_importance(self, X, y, features_to_use=None, preprocess=True, is_oof=True, silent=False, **kwargs):
        feature_importance_fold_list = []
        fold_weights = []
        # TODO: Preprocess data here instead of repeatedly
        model_index = 0
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
                feature_importance_fold = model.compute_feature_importance(X=X.iloc[test_index, :], y=y.iloc[test_index], features_to_use=features_to_use, preprocess=preprocess, silent=silent)
                feature_importance_fold_list.append(feature_importance_fold)
                fold_weights.append(len(test_index))
            model_index += k

        weight_total = sum(fold_weights)
        fold_weights = [weight/weight_total for weight in fold_weights]

        for i, result in enumerate(feature_importance_fold_list):
            feature_importance_fold_list[i] = feature_importance_fold_list[i] * fold_weights[i]

        feature_importance = pd.concat(feature_importance_fold_list, axis=1, sort=True).sum(1).sort_values(ascending=False)

        # TODO: Consider utilizing z scores and stddev to make threshold decisions
        # stddev = pd.concat(feature_importance_fold_list, axis=1, sort=True).std(1).sort_values(ascending=False)
        # feature_importance_df = pd.DataFrame(index=feature_importance.index)
        # feature_importance_df['importance'] = feature_importance
        # feature_importance_df['stddev'] = stddev
        # feature_importance_df['z'] = feature_importance_df['importance'] / feature_importance_df['stddev']

        return feature_importance

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
    def convert_to_refitfull_template(self):
        compressed_params = self._get_compressed_params()
        model_compressed = copy.deepcopy(self._get_model_base())
        model_compressed.feature_types_metadata = self.feature_types_metadata  # TODO: Don't pass this here
        model_compressed.params = compressed_params
        model_compressed.name = model_compressed.name + REFIT_FULL_SUFFIX
        model_compressed.set_contexts(self.path + model_compressed.name + os.path.sep)
        return model_compressed

    def _get_compressed_params(self):
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
    def load(cls, path, file_prefix="", reset_paths=True, low_memory=True, verbose=True):
        path = path + file_prefix
        load_path = path + cls.model_file_name
        obj = load_pkl.load(path=load_path, verbose=verbose)
        if reset_paths:
            obj.set_contexts(path)
        if not low_memory:
            obj.persist_child_models(reset_paths=reset_paths)
        return obj

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

    def save(self, file_prefix="", directory=None, return_filename=False, verbose=True, save_children=False):
        if directory is None:
            directory = self.path
        directory = directory + file_prefix

        if save_children:
            model_names = []
            for child in self.models:
                child = self.load_child(child)
                child.set_contexts(self.path + child.name + os.path.sep)
                child.save(verbose=False)
                model_names.append(child.name)
            self.models = model_names

        file_name = directory + self.model_file_name

        save_pkl.save(path=file_name, object=self, verbose=verbose)
        if return_filename:
            return file_name

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
        bagged_info = dict(
            child_type=self._child_type.__name__,
            num_child_models=len(self.models),
            child_model_names=self._get_model_names(),
            _n_repeats=self._n_repeats,
            # _n_repeats_finished=self._n_repeats_finished,  # commented out because these are too technical
            # _k_fold_end=self._k_fold_end,
            # _k=self._k,
            _k_per_n_repeat=self._k_per_n_repeat,
            _random_state=self._random_state,
            low_memory=self.low_memory,
            bagged_mode=self.bagged_mode,
        )
        info['bagged_info'] = bagged_info
        children_info = self._get_child_info()
        info['children_info'] = children_info

        return info

    def _get_child_info(self):
        child_info_dict = dict()
        for model in self.models:
            if isinstance(model, str):
                child_path = self.create_contexts(self.path + model + os.path.sep)
                child_info_dict[model] = self._child_type.load_info(child_path)
            else:
                child_info_dict[model.name] = model.get_info()
        return child_info_dict
