from typing import Dict, List

from .constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS

__all__ = ["problem_type_info"]


# Note to developers: This is a free-form class. If you need additional parameters, add them.
class ProblemType:
    """
    Simple class that holds information on what a problem type is capable of doing.

    Parameters
    ----------
    can_predict : bool
        Whether models for this problem type have the ability to predict via `model.predict(...)`.
    can_predict_proba : bool
        Whether models for this problem type have the ability to predict probabilities via `model.predict_proba(...)`.
    is_classification : bool
        Whether this is considered a classification problem type.
        For example:
            `binary`, `multiclass`, and `softclass` are considered classification problem types.
            `regression` and `quantile` are not considered classification problem types.
    """

    def __init__(self, can_predict: bool, can_predict_proba: bool, is_classification: bool):
        self.can_predict = can_predict
        self.can_predict_proba = can_predict_proba
        self.is_classification = is_classification


class ProblemTypeInfo:
    """Class that stores all problem_type information, and can vend this information via the provided methods."""

    def __init__(self, problem_type_dict: Dict[str, ProblemType]):
        self.problem_type_dict = problem_type_dict

    def list_problem_types(self):
        return [self.problem_type_dict.keys()]

    def can_predict(self, problem_type: str) -> bool:
        return self._get_problem_type(problem_type).can_predict

    def can_predict_proba(self, problem_type: str) -> bool:
        return self._get_problem_type(problem_type).can_predict_proba

    def is_classification(self, problem_type: str) -> bool:
        return self._get_problem_type(problem_type).is_classification

    def _get_problem_type(self, problem_type: str) -> ProblemType:
        return self.problem_type_dict[problem_type]

    def list_classification(self) -> List[str]:
        return [name for name, problem_type in self.problem_type_dict.items() if problem_type.is_classification]


problem_type_info = ProblemTypeInfo(
    problem_type_dict={
        BINARY: ProblemType(
            can_predict=True,
            can_predict_proba=True,
            is_classification=True,
        ),
        MULTICLASS: ProblemType(
            can_predict=True,
            can_predict_proba=True,
            is_classification=True,
        ),
        SOFTCLASS: ProblemType(
            can_predict=True,
            can_predict_proba=True,
            is_classification=True,
        ),
        REGRESSION: ProblemType(
            can_predict=True,
            can_predict_proba=False,
            is_classification=False,
        ),
        QUANTILE: ProblemType(
            can_predict=True,
            can_predict_proba=False,
            is_classification=False,
        ),
    }
)
