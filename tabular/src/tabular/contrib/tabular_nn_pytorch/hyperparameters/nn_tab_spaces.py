from skopt.space import Integer, Real
from tabular.ml.models.abstract.abstract_spaces import AbstractSpaces


class NNTabularSpaces(AbstractSpaces):
    def __init__(self, problem_type, objective_func, num_classes=None):
        super().__init__(problem_type=problem_type, objective_func=objective_func, num_classes=num_classes)

    def get_binary_baseline(self):
        spaces = [
            [
                Real(0.0, 0.5, name='nn.tabular.dropout'),
                Real(1e-4, 1e1, name='nn.tabular.lr'),
                Integer(1, 30, name='nn.tabular.epochs'),
            ]
        ]
        return spaces

    def get_multiclass_baseline(self):
        return self.get_binary_baseline()

    def get_regression_baseline(self):
        return self.get_binary_baseline()
