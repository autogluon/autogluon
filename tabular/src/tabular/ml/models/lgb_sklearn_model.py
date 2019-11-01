
from tabular.ml.models.abstract_model import AbstractModel


class LGBSKLearnModel(AbstractModel):
    def __init__(self, path, name, model, problem_type, objective_func, debug=0):
        super().__init__(path=path, name=name, model=model, problem_type=problem_type, objective_func=objective_func, debug=debug)
