"""Default hyperparameter search spaces used in EBM model"""

from autogluon.common import space
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION

def get_default_searchspace(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_searchspace_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_searchspace_multiclass_baseline(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_searchspace_regression_baseline()
    else:
        return get_searchspace_binary_baseline()


def get_base_searchspace():
    base_params = {
        "max_leaves": space.Int(2, 3, default=2),
        "smoothing_rounds": space.Int(0, 1000, default=200),
        "learning_rate": space.Real(0.0025, 0.2, default=0.02, log=True),
        "interactions": space.Categorical(
            0,
            "0.5x",
            "1x",
            "1.5x",
            "2x",
            "2.5x",
            "3x",
            "3.5x",
            "4x",
            "4.5x",
            "5x",
            "6x",
            "7x",
            "8x",
            "9x",
            "10x",
            "15x",
            "20x",
            "25x",
        ),
        "interaction_smoothing_rounds": space.Int(0, 200, default=90),
        "min_hessian": space.Real(1e-10, 1e-2, default=1e-4, log=True),
        "min_samples_leaf": space.Int(2, 20, default=4),
        "gain_scale": space.Real(0.5, 5.0, default=5.0, log=True),
        "min_cat_samples": space.Int(5, 20, default=10),
        "cat_smooth": space.Real(5.0, 100.0, default=10.0, log=True),
        "missing": space.Categorical("separate", "low", "high", "gain"),
    }
    return base_params


def get_searchspace_multiclass_baseline(num_classes):
    params = get_base_searchspace()
    baseline_params = {}
    params.update(baseline_params)
    return params


def get_searchspace_binary_baseline():
    params = get_base_searchspace()
    baseline_params = {}
    params.update(baseline_params)
    return params


def get_searchspace_regression_baseline():
    params = get_base_searchspace()
    baseline_params = {}
    params.update(baseline_params)
    return params
