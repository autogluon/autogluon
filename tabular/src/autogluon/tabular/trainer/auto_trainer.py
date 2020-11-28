import logging
import pandas as pd

from .abstract_trainer import AbstractTrainer
from .model_presets.presets import get_preset_models
from ..utils import generate_train_test_split

logger = logging.getLogger(__name__)


# This Trainer handles model training details
class AutoTrainer(AbstractTrainer):
    def get_models(self, hyperparameters, **kwargs):
        path = kwargs.pop('path', self.path)
        problem_type = kwargs.pop('problem_type', self.problem_type)
        eval_metric = kwargs.pop('eval_metric', self.eval_metric)
        stopping_metric = kwargs.pop('stopping_metric', self.stopping_metric)
        num_classes = kwargs.pop('num_classes', self.num_classes)
        invalid_model_names = kwargs.pop('invalid_model_names', self.get_model_names())
        return get_preset_models(path=path, problem_type=problem_type, eval_metric=eval_metric, stopping_metric=stopping_metric,
                                 num_classes=num_classes, hyperparameters=hyperparameters, invalid_model_names=invalid_model_names, **kwargs)

    # TODO: rename to .fit for 0.1
    def train(self, X_train, y_train, X_val=None, y_val=None, X_unlabeled=None, hyperparameter_tune=False, feature_prune=False, holdout_frac=0.1, stack_ensemble_levels=0, hyperparameters=None, ag_args=None, ag_args_fit=None, ag_args_ensemble=None, excluded_model_types=None, time_limit=None, **kwargs):
        for key in kwargs:
            logger.warning(f'Warning: Unknown argument passed to `AutoTrainer.train()`. Argument: {key}')

        if hyperparameters is None:
            hyperparameters = {}
        # TODO: v0.1 self.hyperparmeters is not consistent in repeated calls (re-uses ag_args_fit but not ag_args or ag_args_ensemble)
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

        core_kwargs = {'extra_ag_args': ag_args, 'extra_ag_args_ensemble': ag_args_ensemble}
        self._train_multi_and_ensemble(X_train, y_train, X_val, y_val, X_unlabeled=X_unlabeled, hyperparameters=self.hyperparameters,
                                       hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                                       stack_ensemble_levels=stack_ensemble_levels, time_limit=time_limit, core_kwargs=core_kwargs)
