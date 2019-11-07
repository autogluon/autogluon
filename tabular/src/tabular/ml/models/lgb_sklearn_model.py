
from tabular.ml.models.abstract_model import AbstractModel
from tabular.ml.tuning.hyperparameters.lgbm_sklearn_spaces import LGBMSklearnSpaces


class LGBSKLearnModel(AbstractModel):
    def __init__(self, path, name, model, problem_type, objective_func, debug=0):
        super().__init__(path=path, name=name, model=model, problem_type=problem_type, objective_func=objective_func, debug=debug)

    def hyperparameter_tune(self, X, y, spaces=None):
        if spaces is None:
            spaces = LGBMSklearnSpaces(problem_type=self.problem_type, objective_func=self.objective_func, num_classes=None).get_hyperparam_spaces_baseline()
        return super().hyperparameter_tune(X=X, y=y, spaces=spaces)
