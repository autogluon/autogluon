import logging
logging.basicConfig(format='%(message)s') # just print message in logs

from .base import BaseTask
from .image_classification import ImageClassification
from .object_detection import ObjectDetection, Detector
from .text_classification import TextClassification
from .text_prediction import TextPrediction
from .tabular_prediction import TabularPrediction
from . import image_classification, object_detection, text_classification, tabular_prediction
