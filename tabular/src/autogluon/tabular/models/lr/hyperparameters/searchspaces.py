from autogluon.common.space import Categorical, Real


def get_default_searchspace(problem_type, num_classes=None):
    spaces = {
        "C": Real(lower=0.1, upper=1e3, default=1),
        "proc.skew_threshold": Categorical(0.99, None),
        "penalty": Categorical("L2", "L1"),
    }
    return spaces
