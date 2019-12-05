from ...constants import BINARY, MULTICLASS, REGRESSION


# TODO: Hyperparam tuning in here?
# TODO: Add metadata input: # classes, # rows, # features, etc., can use to get training data for global hyperparam optimizer
class AbstractSpaces:
    def __init__(self, problem_type, objective_func, num_classes=None):
        self.problem_type = problem_type
        self.objective_func = objective_func
        self.num_classes = num_classes

    def get_hyperparam_spaces_baseline(self):
        if self.problem_type == BINARY:
            return self.get_binary_baseline()
        elif self.problem_type == MULTICLASS:
            return self.get_multiclass_baseline()
        elif self.problem_type == REGRESSION:
            return self.get_regression_baseline()
        else:
            return self.get_binary_baseline()

    def get_binary_baseline(self):
        raise NotImplementedError

    def get_multiclass_baseline(self):
        raise NotImplementedError

    def get_regression_baseline(self):
        raise NotImplementedError
