import logging
import pandas as pd

from .abstract_trainer import AbstractTrainer
from .model_presets.presets import get_preset_models
from ..utils import generate_train_test_split

logger = logging.getLogger(__name__)


# This Trainer handles model training details
class AutoTrainer(AbstractTrainer):
    def __init__(self, path, problem_type, scheduler_options=None, objective_func=None, stopping_metric=None, num_classes=None,
                 low_memory=False, feature_types_metadata=None, kfolds=0, n_repeats=1, stack_ensemble_levels=0, time_limit=None, save_data=False, save_bagged_folds=True, random_seed=0, verbosity=2):
        super().__init__(path=path, problem_type=problem_type, scheduler_options=scheduler_options,
                         objective_func=objective_func, stopping_metric=stopping_metric, num_classes=num_classes, low_memory=low_memory,
                         feature_types_metadata=feature_types_metadata, kfolds=kfolds, n_repeats=n_repeats,
                         stack_ensemble_levels=stack_ensemble_levels, time_limit=time_limit,
                         save_data=save_data, save_bagged_folds=save_bagged_folds, random_seed=random_seed, verbosity=verbosity)

    def get_models(self, hyperparameters, hyperparameter_tune=False, **kwargs):
        return get_preset_models(path=self.path, problem_type=self.problem_type, objective_func=self.objective_func, stopping_metric=self.stopping_metric,
                                 num_classes=self.num_classes, hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune)

    def train(self, X_train, y_train, X_test=None, y_test=None, hyperparameter_tune=True, feature_prune=False, holdout_frac=0.1, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = hyperparameters
        models = self.get_models(hyperparameters, hyperparameter_tune=hyperparameter_tune)
        if self.bagged_mode:
            if (y_test is not None) and (X_test is not None):
                # TODO: User could be intending to blend instead. Perhaps switch from OOF preds to X_test preds while still bagging? Doubt a user would want this.
                logger.debug('Warning: Training AutoGluon in Bagged Mode but X_test is specified, concatenating X_train and X_test for cross-validation')
                X_train = pd.concat([X_train, X_test], ignore_index=True)
                y_train = pd.concat([y_train, y_test], ignore_index=True)
            X_test = None
            y_test = None
        else:
            if (y_test is None) or (X_test is None):
                X_train, X_test, y_train, y_test = generate_train_test_split(X_train, y_train, problem_type=self.problem_type, test_size=holdout_frac, random_state=self.random_seed)
        self.train_multi_and_ensemble(X_train, y_train, X_test, y_test, models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
