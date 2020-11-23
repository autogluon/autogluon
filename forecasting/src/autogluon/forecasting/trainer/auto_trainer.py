import logging
import pandas as pd

from .abstract_trainer import AbstractTrainer
from .model_presets.presets import get_preset_models

logger = logging.getLogger(__name__)

__all__ = ['AutoTrainer', 'AbstractTrainer']


class AutoTrainer(AbstractTrainer):
    def get_models(self, hyperparameters, **kwargs):
        path = kwargs.pop('path', self.path)
        eval_metric = kwargs.pop('eval_metric', self.eval_metric)
        return get_preset_models(path=path,
                                 eval_metric=eval_metric,
                                 hyperparameters=hyperparameters,
                                 freq=self.freq,
                                 prediction_length=self.prediction_length)

    # TODO: rename to .fit for 0.1
    def train(self, train_data, test_data=None, hyperparameter_tune=False, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {}
        self._train_multi(train_data,
                          test_data=test_data,
                          hyperparameters=hyperparameters,
                          hyperparameter_tune=hyperparameter_tune)
