
from tabular.ml.models.sklearn_model import SKLearnModel
from tabular.ml.tuning.hyperparameters.rf_spaces import RFSpaces


# TODO: Pass in num_classes?
class RFModel(SKLearnModel):
    def __init__(self, path, name, model, problem_type, objective_func, debug=0):
        super().__init__(path=path, name=name, model=model, problem_type=problem_type, objective_func=objective_func, debug=debug)

    def preprocess(self, X):
        X = super().preprocess(X)
        X = X.fillna(0)
        return X

    def hyperparameter_tune(self, X, y, spaces=None):
        if spaces is None:
            spaces = RFSpaces(problem_type=self.problem_type, objective_func=self.objective_func, num_classes=None).get_hyperparam_spaces_baseline()
        return super().hyperparameter_tune(X=X, y=y, spaces=spaces)
