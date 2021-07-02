import copy
import logging
import os
import time
from abc import abstractmethod

from numpy import ndarray
from pandas import DataFrame, Series

from ...utils.exceptions import TimeLimitExceeded

logger = logging.getLogger(__name__)


class AbstractFoldFittingStrategy(object):

    @abstractmethod
    def schedule_fold_model_fit(self, model_base, fold_ctx, kwargs):
        """
        Schedule fold model training. By design this part is supposed to be 'lazy' evaluator no actual training is performed here.
        Distributed fitters will handle jobs scheduling here.
        """
        pass

    @abstractmethod
    def after_all_folds_scheduled(self):
        """
        Method is called when all the folds are scheduled.
        Local fitters will perform training here.
        Distributed fitters will handle job handles and results retrieval here.
        """
        pass


class SequentialLocalFoldFittingStrategy(AbstractFoldFittingStrategy):
    """
    This strategy fits the folds locally in a sequence.
    """

    def __init__(self, bagged_ensemble_model, X: DataFrame, y: Series, sample_weight, time_limit: float, time_start: float,
                 models: list, oof_pred_proba: ndarray, oof_pred_model_repeats: ndarray, save_folds: bool):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.time_limit = time_limit
        self.time_start = time_start
        self.models = models
        self.oof_pred_proba = oof_pred_proba
        self.oof_pred_model_repeats = oof_pred_model_repeats
        self.bagged_ensemble_model = bagged_ensemble_model
        self.jobs = []
        self.save_folds = save_folds

    def schedule_fold_model_fit(self, model_base, fold_ctx, kwargs):
        self.jobs.append([model_base, fold_ctx, kwargs])

    def after_all_folds_scheduled(self):
        for job in self.jobs:
            self._fit_fold_model(*job)

    def _fit_fold_model(self, model_base, fold_ctx, kwargs):
        time_start_fold = time.time()
        time_limit_fold = self._get_fold_time_limit(fold_ctx)
        fold_model = self._fit(model_base, time_start_fold, time_limit_fold, fold_ctx, kwargs)
        fold_model, pred_proba = self._predict_oof(fold_model, fold_ctx)
        self._update_bagged_ensemble(fold_model, pred_proba, fold_ctx)

    def _get_fold_time_limit(self, fold_ctx):
        _, folds_finished, folds_left, folds_to_fit, _, _ = self._get_fold_properties(fold_ctx)
        time_elapsed = time.time() - self.time_start
        if self.time_limit is not None:
            time_left = self.time_limit - time_elapsed
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
        return time_limit_fold

    def _update_bagged_ensemble(self, fold_model, pred_proba, fold_ctx):
        _, val_index = fold_ctx['fold']
        model_to_append = fold_model
        if not self.save_folds:
            fold_model.model = None
        if self.bagged_ensemble_model.low_memory:
            self.bagged_ensemble_model.save_child(fold_model, verbose=False)
            model_to_append = fold_model.name
        self.models.append(model_to_append)
        self.oof_pred_proba[val_index] += pred_proba
        self.oof_pred_model_repeats[val_index] += 1
        self.bagged_ensemble_model._add_child_times_to_bag(model=fold_model)

    def _fit(self, model_base, time_start_fold, time_limit_fold, fold_ctx, kwargs):
        fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix = self._get_fold_properties(fold_ctx)
        train_index, val_index = fold
        X_fold, X_val_fold = self.X.iloc[train_index, :], self.X.iloc[val_index, :]
        y_fold, y_val_fold = self.y.iloc[train_index], self.y.iloc[val_index]
        fold_model = copy.deepcopy(model_base)
        fold_model.name = f'{fold_model.name}{model_name_suffix}'
        fold_model.set_contexts(self.bagged_ensemble_model.path + fold_model.name + os.path.sep)
        kwargs_fold = kwargs.copy()
        if self.sample_weight is not None:
            kwargs_fold['sample_weight'] = self.sample_weight[train_index]
            kwargs_fold['sample_weight_val'] = self.sample_weight[val_index]
        fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **kwargs_fold)
        fold_model.fit_time = time.time() - time_start_fold
        return fold_model

    def _predict_oof(self, fold_model, fold_ctx):
        time_train_end_fold = time.time()
        fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix = self._get_fold_properties(fold_ctx)
        _, val_index = fold
        X_val_fold = self.X.iloc[val_index, :]
        y_val_fold = self.y.iloc[val_index]
        if self.time_limit is not None:  # Check to avoid unnecessarily predicting and saving a model when an Exception is going to be raised later
            if not is_last_fold:
                time_elapsed = time.time() - self.time_start
                time_left = self.time_limit - time_elapsed
                expected_time_required = time_elapsed * folds_to_fit / (folds_finished + 1)
                expected_remaining_time_required = expected_time_required * (folds_left - 1) / folds_to_fit
                if expected_remaining_time_required > time_left:
                    raise TimeLimitExceeded
        pred_proba = fold_model.predict_proba(X_val_fold)
        fold_model.predict_time = time.time() - time_train_end_fold
        fold_model.val_score = fold_model.score_with_y_pred_proba(y=y_val_fold, y_pred_proba=pred_proba)
        fold_model.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
        if not self.bagged_ensemble_model.params.get('save_bag_folds', True):
            fold_model.model = None
        return fold_model, pred_proba

    @staticmethod
    def _get_fold_properties(fold_ctx):
        fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix = [
            fold_ctx[f] for f in ['fold', 'folds_finished', 'folds_left', 'folds_to_fit', 'is_last_fold', 'model_name_suffix']
        ]
        return fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix
