import logging
import time

from .abstract_learner import AbstractLearner
from ..trainer.auto_trainer import AutoTrainer

logger = logging.getLogger(__name__)


class DefaultLearner(AbstractLearner):

    def __init__(self, trainer_type=AutoTrainer, **kwargs):
        super().__init__(**kwargs)
        self.trainer_type = trainer_type

    def _fit(self, train_data, freq, prediction_length, val_data=None, scheduler_options=None, hyperparameter_tune=False,
            hyperparameters=None, time_limit=None, **kwargs):
        self._time_limit = time_limit
        time_start = time.time()
        if time_limit:
            logger.log(20, f'Beginning AutoGluon training ... Time limit = {time_limit}s')
        else:
            logger.log(20, 'Beginning AutoGluon training ...')
        logger.log(20, f'AutoGluon will save models to {self.path}')

        trainer = self.trainer_type(
            path=self.model_context,
            freq=freq,
            prediction_length=prediction_length,
            eval_metric=self.eval_metric,
            scheduler_options=scheduler_options,
            **kwargs
        )

        self.trainer_path = trainer.path
        if self.eval_metric is None:
            self.eval_metric = trainer.eval_metric

        self.save()
        trainer.train(train_data=train_data,
                      val_data=val_data,
                      hyperparameter_tune=hyperparameter_tune,
                      hyperparameters=hyperparameters)
        self.save_trainer(trainer=trainer)
        time_end = time.time()
        self._time_fit_training = time_end - time_start
        logger.log(20, f'AutoGluon training complete, total runtime = {round(self._time_fit_training, 2)}s ...')

    def get_info(self, include_model_info=False, **kwargs):
        learner_info = super().get_info(include_model_info=include_model_info)
        trainer = self.load_trainer()
        trainer_info = trainer.get_info(include_model_info=include_model_info)
        learner_info.update({
            'time_fit_training': self._time_fit_training,
            'time_limit': self._time_limit,
        })

        learner_info.update(trainer_info)
        return learner_info