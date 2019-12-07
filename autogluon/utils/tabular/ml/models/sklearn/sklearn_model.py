import logging
from ..abstract.abstract_model import AbstractModel
from ...utils import convert_categorical_to_int

logger = logging.getLogger(__name__)

class SKLearnModel(AbstractModel):
    def preprocess(self, X):
        X = convert_categorical_to_int(X)
        return super().preprocess(X)
