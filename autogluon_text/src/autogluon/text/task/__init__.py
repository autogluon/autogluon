import logging
logging.basicConfig(format='%(message)s')  # just print message in logs

from .text_classification import TextClassification
from .text_prediction import TextPrediction
from . import text_classification
