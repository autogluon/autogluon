from scipy.stats import randint as sp_randint
from scipy.stats import uniform

from tabular.ml.tuning.hyperparameters.abstract_spaces import AbstractSpaces


class LGBMSklearnSpaces(AbstractSpaces):
    def __init__(self, problem_type, objective_func, num_classes=None):
        super().__init__(problem_type=problem_type, objective_func=objective_func, num_classes=num_classes)

    def get_binary_baseline(self):
        spaces = {
            "colsample_bytree": uniform(),
            "num_leaves": sp_randint(16, 96),
        }
        return spaces

    def get_multiclass_baseline(self):
        return self.get_binary_baseline()

    def get_regression_baseline(self):
        spaces = {
            "colsample_bytree": uniform(),
            "num_leaves": sp_randint(16, 96),
        }
        return spaces
