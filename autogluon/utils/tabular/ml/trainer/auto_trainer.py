import logging
import pandas as pd

from .abstract_trainer import AbstractTrainer
from .model_presets.presets import get_preset_models
from ..utils import generate_train_test_split

logger = logging.getLogger(__name__)


# This Trainer handles model training details
class AutoTrainer(AbstractTrainer):
    def get_models(self, hyperparameters, hyperparameter_tune=False, level='default', extra_ag_args_fit=None, **kwargs):
        return get_preset_models(path=self.path, problem_type=self.problem_type, eval_metric=self.eval_metric, stopping_metric=self.stopping_metric,
                                 num_classes=self.num_classes, hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune, level=level, extra_ag_args_fit=extra_ag_args_fit)

    # TODO: rename to .fit for 0.1
    def train(self, X_train, y_train, X_val=None, y_val=None, hyperparameter_tune=False, feature_prune=False, holdout_frac=0.1, hyperparameters=None, ag_args_fit=None, excluded_model_types=None, time_limit=None, **kwargs):
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = self._process_hyperparameters(hyperparameters=hyperparameters, ag_args_fit=ag_args_fit, excluded_model_types=excluded_model_types)
        if self.bagged_mode:
            if (y_val is not None) and (X_val is not None):
                # TODO: User could be intending to blend instead. Perhaps switch from OOF preds to X_val preds while still bagging? Doubt a user would want this.
                logger.warning('Warning: Training AutoGluon in Bagged Mode but X_val is specified, concatenating X_train and X_val for cross-validation')
                X_train = pd.concat([X_train, X_val], ignore_index=True)
                y_train = pd.concat([y_train, y_val], ignore_index=True)
            X_val = None
            y_val = None
        else:
            if (y_val is None) or (X_val is None):
                X_train, X_val, y_train, y_val = generate_train_test_split(X_train, y_train, problem_type=self.problem_type, test_size=holdout_frac, random_state=self.random_seed)
        self._train_multi_and_ensemble(X_train, y_train, X_val, y_val, hyperparameters=self.hyperparameters, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, time_limit=time_limit)
