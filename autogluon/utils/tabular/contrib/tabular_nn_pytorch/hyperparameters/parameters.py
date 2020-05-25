from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION


def get_param_baseline(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline()
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    else:
        return get_param_binary_baseline()


def get_param_multiclass_baseline():
    params = {
        'dropout': 0.1,  # linear layers dropout
        'bs': 256,  # batch size

        # maximum learning rate for one cycle policy https://docs.fast.ai/train.html#fit_one_cycle
        # One-cycle policy paper: https://arxiv.org/abs/1803.09820
        'lr': 1e-2,
        'epochs': 30,  # number of epochs
        'metric': 'accuracy',

        # Early stopping settings. See more details here: https://docs.fast.ai/callbacks.tracker.html#EarlyStoppingCallback
        'early.stopping.min_delta': 0.001,
        'early.stopping.patience': 7,

        # If > 0, then use LabelSmoothingCrossEntropy loss function for binary/multi-class classification;
        # otherwise use default loss function for this type of problem
        'smoothing': 0.0,
    }
    return params


def get_param_binary_baseline():
    params = get_param_multiclass_baseline()
    return params


def get_param_regression_baseline():
    params = {
        'dropout': 0.1,  # linear layers dropout
        'bs': 256,  # batch size

        # maximum learning rate for one cycle policy https://docs.fast.ai/train.html#fit_one_cycle
        # One-cycle policy paper: https://arxiv.org/abs/1803.09820
        'lr': 1e-2,
        'epochs': 30,  # number of epochs
        'metric': 'root_mean_squared_error',

        # Early stopping settings. See more details here: https://docs.fast.ai/callbacks.tracker.html#EarlyStoppingCallback
        'early.stopping.min_delta': 0.001,
        'early.stopping.patience': 7,

        # If > 0, then use LabelSmoothingCrossEntropy loss function for binary/multi-class classification;
        # otherwise use default loss function for this type of problem
        'smoothing': 0.0,
    }
    return params
