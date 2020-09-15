import logging
logging.basicConfig(format='%(message)s') # just print message in logs

from autogluon.core.task.base import BaseTask
from .image_classification import ImageClassification
from .object_detection import ObjectDetection, Detector
from autogluon.text.task.text_classification import TextClassification
from autogluon.text.task.text_prediction import TextPrediction
from autogluon.tabular.task.tabular_prediction import TabularPrediction
from . import image_classification, object_detection
from autogluon.text.task import text_classification
from autogluon.tabular import tabular_prediction
