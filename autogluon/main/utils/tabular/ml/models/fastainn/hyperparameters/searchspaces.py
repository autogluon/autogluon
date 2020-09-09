from .....ml.constants import BINARY, MULTICLASS, REGRESSION
from .......core import Categorical, Real, Int


def get_default_searchspace(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_searchspace_binary().copy()
    elif problem_type == MULTICLASS:
        return get_searchspace_multiclass(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_searchspace_regression().copy()
    else:
        return get_searchspace_binary().copy()


def get_searchspace_binary():
    spaces = {
        # See docs: https://docs.fast.ai/tabular.models.html
        'layers': Categorical(None, [200, 100], [200], [500], [1000], [500, 200], [50, 25], [1000, 500], [200, 100, 50], [500, 200, 100], [1000, 500, 200]),
        'emb_drop': Real(0.0, 0.5, default=0.1),
        'ps': Real(0.0, 0.5, default=0.1),
        'bs': Categorical(256, 64, 128, 512, 1024, 2048, 4096),
        'lr': Real(5e-5, 1e-1, default=1e-2, log=True),
        'epochs': Int(lower=5, upper=30, default=30),
        'early.stopping.min_delta': 0.0001,
        'early.stopping.patience': 20,
        'smoothing': Real(0.0, 0.3, default=0.0, log=True),
    }
    return spaces


def get_searchspace_multiclass(num_classes):
    return get_searchspace_binary()


def get_searchspace_regression():
    return get_searchspace_binary()
