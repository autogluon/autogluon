from ..abstract.abstract_model import AbstractModel
from ...utils import convert_categorical_to_int


class SKLearnModel(AbstractModel):
    def preprocess(self, X):
        X = convert_categorical_to_int(X)
        return super().preprocess(X)
