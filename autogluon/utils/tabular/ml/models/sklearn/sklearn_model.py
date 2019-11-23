
from autogluon.utils.tabular.ml.models.abstract.abstract_model import AbstractModel
from autogluon.utils.tabular.ml.utils import convert_categorical_to_int


class SKLearnModel(AbstractModel):
    def preprocess(self, X):
        X = convert_categorical_to_int(X)
        return super().preprocess(X)
