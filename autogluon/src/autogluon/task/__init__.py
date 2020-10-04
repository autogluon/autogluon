import logging
logging.basicConfig(format='%(message)s') # just print message in logs

from .image_classification import ImageClassification
from .object_detection import ObjectDetection, Detector
from .text_classification import TextClassification
from .text_prediction import TextPrediction
from . import image_classification, object_detection, text_classification
