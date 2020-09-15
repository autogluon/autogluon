import logging
logging.basicConfig(format='%(message)s') # just print message in logs

from autogluon.core.task.base import BaseTask
from .text_classification import TextClassification
from .text_prediction import TextPrediction
from autogluon.tabular.task.tabular_prediction import TabularPrediction
from . import text_classification
