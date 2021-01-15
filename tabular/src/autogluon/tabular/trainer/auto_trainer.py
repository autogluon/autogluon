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
        num_classes = kwargs.pop('num_classes', self.num_classes)
        invalid_model_names = kwargs.pop('invalid_model_names', self.get_model_names())
        feature_metadata = kwargs.pop('feature_metadata', self.feature_metadata)
        return get_preset_models(path=path, problem_type=problem_type, eval_metric=eval_metric,
                                 num_classes=num_classes, hyperparameters=hyperparameters, invalid_model_names=invalid_model_names, feature_metadata=feature_metadata, **kwargs)

    # TODO: v0.1 rename to .fit
    # TODO: v0.1 rename stack_ensemble_levels -> num_stack_levels
    def train(self, X_train, y_train, hyperparameters, X_val=None, y_val=None, X_unlabeled=None, hyperparameter_tune_kwargs=None, feature_prune=False, holdout_frac=0.1, stack_ensemble_levels=0, core_kwargs: dict = None, time_limit=None, **kwargs):
        for key in kwargs:
            logger.warning(f'Warning: Unknown argument passed to `AutoTrainer.train()`. Argument: {key}')

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
                logger.log(20, f'Automatically generating train/validation split with holdout_frac={holdout_frac}, Train Rows: {len(X_train)}, Val Rows: {len(X_val)}')

        self._train_multi_and_ensemble(X_train, y_train, X_val, y_val, X_unlabeled=X_unlabeled, hyperparameters=hyperparameters,
                                       hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, feature_prune=feature_prune,
                                       stack_ensemble_levels=stack_ensemble_levels, time_limit=time_limit, core_kwargs=core_kwargs)
