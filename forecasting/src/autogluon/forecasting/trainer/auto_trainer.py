import logging

from .abstract_trainer import AbstractTrainer
from .model_presets.presets import get_preset_models

logger = logging.getLogger(__name__)


class AutoTrainer(AbstractTrainer):
    def get_models(self, hyperparameters, **kwargs):
        path = kwargs.pop('path', self.path)
        eval_metric = kwargs.pop('eval_metric', self.eval_metric)
        hyperparameter_tune = kwargs.get('hyperparameter_tune', False)
        return get_preset_models(path=path,
                                 eval_metric=eval_metric,
                                 prediction_length=self.prediction_length,
                                 freq=self.freq,
                                 hyperparameters=hyperparameters,
                                 hyperparameter_tune=hyperparameter_tune,
                                 quantiles=self.quantiles,
                                 use_feat_static_cat=self.use_feat_static_cat,
                                 use_feat_static_real=self.use_feat_static_real,
                                 cardinality=self.cardinality)

    def train(self, train_data, val_data=None, hyperparameter_tune=False, hyperparameters=None, time_limit=None):
        self._train_multi(train_data,
                          val_data=val_data,
                          hyperparameters=hyperparameters,
                          hyperparameter_tune=hyperparameter_tune,
                          time_limit=time_limit)
