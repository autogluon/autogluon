from autogluon.task.base.base_task import BaseTask
from .dataset import TimeSeriesDataset

from ...utils.ml.learner.abstract_learner import AbstractLearner

from .predictor import ForecastingPredictor

__all__ = ['Forecasting']


class Forecasting(BaseTask):

    @staticmethod
    def Dataset(*args, **kwargs):
        return TimeSeriesDataset(*args, **kwargs)

    @staticmethod
    def fit(hyperparameters, train_ds, test_ds, metric=None, hyperparameter_tune=False):
        learner = AbstractLearner()
        learner.fit(hyperparameters=hyperparameters,
                    hyperparameter_tune=hyperparameter_tune,
                    train_ds=train_ds,
                    test_ds=test_ds,
                    metric=metric)
        predictor = ForecastingPredictor(learner)
        return predictor






