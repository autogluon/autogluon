from ..base.base_task import BaseTask
from .dataset import ForecastingDataset
import logging

import mxnet as mx

import copy

from ...core import *
from ...scheduler.resource import get_cpu_count, get_gpu_count
from ..base import BaseTask, compile_scheduler_options
from ...utils import update_params
from .predictor import ForecastingPredictor
from ...utils.forecasting.ml.learner.abstract_learner import AbstractLearner

from .predictor import ForecastingPredictor


class Forecasting(BaseTask):

    @staticmethod
    def Dataset(*args, **kwargs):
        return ForecastingDataset(*args, **kwargs)

    @staticmethod
    def fit(hyperparameters, train_ds, test_ds, hyperparameter_tune=False):
        learner = AbstractLearner()
        learner.fit(hyperparameters, hyperparameter_tune, train_ds, test_ds)
        predictor = ForecastingPredictor(learner)
        return predictor






