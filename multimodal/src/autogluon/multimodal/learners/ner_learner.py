from typing import Optional
from .default_learner import DefaultLearner


class NERLearner(DefaultLearner):
    def __init__(
        self,
        problem_type: str,
        label: Optional[str] = None,
        # column_types: Optional[dict] = None,  # this is now inferred in learner
        # query: Optional[Union[str, List[str]]] = None,
        # response: Optional[Union[str, List[str]]] = None,
        # match_label: Optional[Union[int, str]] = None,
        # pipeline: Optional[str] = None,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,  # this is now inferred in learner
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,
        classes: Optional[list] = None,
        enable_progress_bar: Optional[bool] = True,
    ):
        super().__init__(
            problem_type=problem_type,
            label=label,
            presets=presets,
            eval_metric=eval_metric,
            path=path,
            verbosity=verbosity,
            num_classes=num_classes,
            classes=classes,
            enable_progress_bar=enable_progress_bar,
        )
