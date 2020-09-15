import logging
logging.basicConfig(format='%(message)s') # just print message in logs

from autogluon.core.task.base import BaseTask
from .image_classification import ImageClassification
from .object_detection import ObjectDetection, Detector
from .text_classification import TextClassification
from .text_prediction import TextPrediction
from autogluon.tabular.task.tabular_prediction import TabularPrediction
from . import image_classification, object_detection, text_classification
from autogluon.tabular import tabular_prediction
