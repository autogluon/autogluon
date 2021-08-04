from .version import __version__
import logging

from .predictor.predictor import ForecastingPredictor
from autogluon.core.dataset import TabularDataset

logging.basicConfig(format='%(message)s')  # just print message in logs
