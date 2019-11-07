from scipy.stats import randint as sp_randint
from scipy.stats import uniform

from tabular.ml.tuning.hyperparameters.abstract_spaces import AbstractSpaces


class RFSpaces(AbstractSpaces):
    def __init__(self, problem_type, objective_func, num_classes=None):
        super().__init__(problem_type=problem_type, objective_func=objective_func, num_classes=num_classes)

    def get_binary_baseline(self):
        spaces = {
            'n_estimators': [300],
            "max_depth": sp_randint(4, 32),
            "max_features": uniform(),
            "min_samples_split": sp_randint(2, 11),
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
            'n_jobs': [-1]
        }
        return spaces

    def get_multiclass_baseline(self):
        return self.get_binary_baseline()

    def get_regression_baseline(self):
        spaces = {
            'n_estimators': [300],
            "max_depth": sp_randint(4, 32),
            "max_features": uniform(),
            "min_samples_split": sp_randint(2, 11),
            "criterion": ['mae'],  # TODO: Criterion should be defined by objective func
            'n_jobs': [-1]
        }
        return spaces
