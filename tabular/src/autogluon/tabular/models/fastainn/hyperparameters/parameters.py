from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE

# TODO this method is generalizable and potentially should be moved out into framework
def get_param_baseline(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline()
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    elif problem_type == QUANTILE:
        return get_param_quantile_baseline()
    else:
        return get_param_binary_baseline()


def get_param_multiclass_baseline():
    # TODO: explore/add other hyperparameters like weight decay, use of batch-norm, activation-function choice, etc.
    params = {
        # See docs: https://docs.fast.ai/tabular.models.html
        'layers': None,  # layers configuration; None - use model's heuristics
        'emb_drop': 0.1,  # embedding layers dropout
        'ps': [0.1],  # linear layers dropout
        'bs': 256,  # batch size

        # maximum learning rate for one cycle policy https://docs.fast.ai/train.html#fit_one_cycle
        # One-cycle policy paper: https://arxiv.org/abs/1803.09820
        'lr': 1e-2,
        'epochs': 30,  # maximum number of epochs

        # Early stopping settings. See more details here: https://docs.fast.ai/callbacks.tracker.html#EarlyStoppingCallback
        'early.stopping.min_delta': 0.0001,
        'early.stopping.patience': 20,

        # If > 0, then use LabelSmoothingCrossEntropy loss function for binary/multi-class classification;
        # otherwise use default loss function for this type of problem
        'smoothing': 0.0,
    }
    return params


def get_param_binary_baseline():
    return get_param_multiclass_baseline()


def get_param_regression_baseline():
    return get_param_multiclass_baseline()


def get_param_quantile_baseline():
    return get_param_multiclass_baseline()
