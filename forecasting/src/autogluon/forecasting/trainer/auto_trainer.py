import logging

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
                                 prediction_length=self.prediction_length,
                                 quantiles=self.quantiles)

    def train(self, train_data, val_data=None, hyperparameter_tune=False, hyperparameters=None):
        self._train_multi(train_data,
                          val_data=val_data,
                          hyperparameters=hyperparameters,
                          hyperparameter_tune=hyperparameter_tune)
