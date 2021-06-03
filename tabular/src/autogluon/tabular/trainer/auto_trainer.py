import logging
import warnings

import pandas as pd

from autogluon.core.utils import generate_train_test_split

from .abstract_trainer import AbstractTrainer
from .model_presets.presets import get_preset_models
from .model_presets.presets_distill import get_preset_models_distillation

logger = logging.getLogger(__name__)


# This Trainer handles model training details
class AutoTrainer(AbstractTrainer):
    def construct_model_templates(self, hyperparameters, **kwargs):
        path = kwargs.pop('path', self.path)
        problem_type = kwargs.pop('problem_type', self.problem_type)
        eval_metric = kwargs.pop('eval_metric', self.eval_metric)
        quantile_levels = kwargs.pop('quantile_levels', self.quantile_levels)
        invalid_model_names = kwargs.pop('invalid_model_names', self._get_banned_model_names())
        silent = kwargs.pop('silent', self.verbosity < 3)

        return get_preset_models(path=path, problem_type=problem_type, eval_metric=eval_metric,
                                 quantile_levels=quantile_levels,
                                 hyperparameters=hyperparameters, invalid_model_names=invalid_model_names,
                                 silent=silent, **kwargs)

    def fit(self, X, y, hyperparameters, X_val=None, y_val=None, X_unlabeled=None, feature_prune=False, holdout_frac=0.1, num_stack_levels=0, core_kwargs: dict = None, time_limit=None, **kwargs):
        for key in kwargs:
            logger.warning(f'Warning: Unknown argument passed to `AutoTrainer.fit()`. Argument: {key}')

        if self.bagged_mode:
            if (y_val is not None) or (X_val is not None):
                # TODO: User could be intending to blend instead. Add support for blend stacking.
                #  This error message is necessary because when calculating out-of-fold predictions for user, we want to return them in the form given in train_data,
                #  but if we merge train and val here, it becomes very confusing from a users perspective, especially because we reset index, making it impossible to match
                #  the original train_data to the out-of-fold predictions from `predictor.get_oof_pred_proba()`.
                raise AssertionError('X_val, y_val is not None, but bagged mode was specified. If calling from `TabularPredictor.fit()`, `tuning_data` must be None.\n'
                                     'Bagged mode does not use tuning data / validation data. Instead, all data (`train_data` and `tuning_data`) should be combined and specified as `train_data`.\n'
                                     'Bagging/Stacking with a held-out validation set (blend stacking) is not yet supported.')
            X_val = None
            y_val = None
        else:
            if (y_val is None) or (X_val is None):
                X, X_val, y, y_val = generate_train_test_split(X, y, problem_type=self.problem_type, test_size=holdout_frac, random_state=self.random_state)
                logger.log(20, f'Automatically generating train/validation split with holdout_frac={holdout_frac}, Train Rows: {len(X)}, Val Rows: {len(X_val)}')

        self._train_multi_and_ensemble(X, y, X_val, y_val, X_unlabeled=X_unlabeled, hyperparameters=hyperparameters,
                                       feature_prune=feature_prune,
                                       num_stack_levels=num_stack_levels, time_limit=time_limit, core_kwargs=core_kwargs)

    def construct_model_templates_distillation(self, hyperparameters, **kwargs):
        path = kwargs.pop('path', self.path)
        problem_type = kwargs.pop('problem_type', self.problem_type)
        eval_metric = kwargs.pop('eval_metric', self.eval_metric)
        invalid_model_names = kwargs.pop('invalid_model_names', self._get_banned_model_names())
        silent = kwargs.pop('silent', self.verbosity < 3)

        # TODO: QUANTILE VERSION?

        return get_preset_models_distillation(
            path=path,
            problem_type=problem_type,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            invalid_model_names=invalid_model_names,
            silent=silent,
            **kwargs,
        )
