
from tabular.ml.models.abstract_model import AbstractModel
from tabular.ml.utils import convert_categorical_to_int


class SKLearnModel(AbstractModel):
    def __init__(self, path, name, model, problem_type, objective_func, debug=0):
        super().__init__(path=path, name=name, model=model, problem_type=problem_type, objective_func=objective_func, debug=debug)

    def preprocess(self, X):
        X = convert_categorical_to_int(X)
        return super().preprocess(X)
