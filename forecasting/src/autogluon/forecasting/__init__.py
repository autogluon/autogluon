import logging

from .task.forecasting import Forecasting
from .task.forecasting import ForecastingPredictorV1
from .predictor.predictor import ForecastingPredictor
from autogluon.core.dataset import TabularDataset

logging.basicConfig(format='%(message)s')  # just print message in logs