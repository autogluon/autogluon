"""Default (fixed) hyperparameter values used in Tabular Neural Network models.
A value of None typically indicates an adaptive value for the hyperparameter will be chosen based on the data.
"""

from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION


def get_fixed_params(framework):
    """Parameters that currently cannot be searched during HPO"""
    fixed_params = {
        # 'seed_value': 0,  # random seed for reproducibility (set = None to ignore)
    }
    pytorch_fixed_params = {
        "num_epochs": 1000,  # maximum number of epochs (passes over full dataset) for training NN
        "epochs_wo_improve": None,  # we terminate training if validation performance hasn't improved in the last 'epochs_wo_improve' # of epochs
    }
    return merge_framework_params(framework=framework, shared_params=fixed_params, pytorch_params=pytorch_fixed_params)


def get_hyper_params(framework):
    """Parameters that currently can be tuned during HPO"""
    hyper_params = {
        ## Hyperparameters for neural net architecture:
        "activation": "relu",  # Activation function
        # Options: ['relu', 'softrelu', 'tanh', 'softsign'], options for pytorch: ['relu', 'elu', 'tanh']
        "embedding_size_factor": 1.0,  # scaling factor to adjust size of embedding layers (float > 0)
        # Options: range[0.01 - 100] on log-scale
        "embed_exponent": 0.56,  # exponent used to determine size of embedding layers based on # categories.
        "max_embedding_dim": 100,  # maximum size of embedding layer for a single categorical feature (int > 0).
        ## Regression-specific hyperparameters:
        "y_range": None,  # Tuple specifying whether Y is constrained to (min_y, max_y). Can be = (-np.inf, np.inf).
        # If None, inferred based on training labels. Note: MUST be None for classification tasks!
        "y_range_extend": 0.05,  # Only used to extend size of inferred y_range when y_range = None.
        ## Hyperparameters for neural net training:
        "dropout_prob": 0.1,  # dropout probability, = 0 turns off Dropout.
        # Options: range(0.0, 0.5)
        "optimizer": "adam",  # Which optimizer to use for training
        # Options include: ['adam','sgd']
        "learning_rate": 3e-4,  # learning rate used for NN training (float > 0)
        "weight_decay": 1e-6,  # weight decay regularizer (float > 0)
        ## Hyperparameters for data processing:
        "proc.embed_min_categories": 4,  # apply embedding layer to categorical features with at least this many levels. Features with fewer levels are one-hot encoded. Choose big value to avoid use of Embedding layers
        # Options: [3,4,10, 100, 1000]
        "proc.impute_strategy": "median",  # strategy argument of sklearn.SimpleImputer() used to impute missing numeric values
        # Options: ['median', 'mean', 'most_frequent']
        "proc.max_category_levels": 100,  # maximum number of allowed levels per categorical feature
        # Options: [10, 100, 200, 300, 400, 500, 1000, 10000]
        "proc.skew_threshold": 0.99,  # numerical features whose absolute skewness is greater than this receive special power-transform preprocessing. Choose big value to avoid using power-transforms
        # Options: [0.2, 0.3, 0.5, 0.8, 0.9, 0.99, 0.999, 1.0, 10.0, 100.0]
        "use_ngram_features": False,  # If False, will drop automatically generated ngram features from language features. This results in worse model quality but far faster inference and training times.
        # Options: [True, False]
    }
    pytorch_hyper_params = {
        "num_layers": 4,  # number of layers
        # Options: [2, 3, 4, 5]
        "hidden_size": 128,  # number of hidden units in each layer
        # Options: [128, 256, 512]
        "max_batch_size": 512,  # maximum batch-size, actual batch size may be slightly smaller.
        "use_batchnorm": False,  # whether or not to utilize batch normalization
        # Options: [True, False]
        "loss_function": "auto",  # Pytorch loss function minimized during training
        # Example options for regression: nn.MSELoss(), nn.L1Loss()
    }
    return merge_framework_params(framework=framework, shared_params=hyper_params, pytorch_params=pytorch_hyper_params)


def get_quantile_hyper_params(framework):
    """Parameters that currently can be searched during HPO"""
    hyper_params = get_hyper_params(framework)
    new_hyper_params = {
        "gamma": 5.0,  # margin loss weight which helps ensure noncrossing quantile estimates
        # Options: range(0.1, 10.0)
        "alpha": 0.01,  # used for smoothing huber pinball loss
    }
    hyper_params.update(new_hyper_params)
    return hyper_params


# Note: params for original NNTabularModel were:
# weight_decay=0.01, dropout_prob = 0.1, batch_size = 2048, lr = 1e-2, epochs=30, layers= [200, 100] (semi-equivalent to our layers = [100],numeric_embed_dim=200)
def get_default_param(problem_type, framework, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary(framework)
    elif problem_type == MULTICLASS:
        return get_param_multiclass(framework=framework, num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_param_regression(framework)
    elif problem_type == QUANTILE:
        return get_param_quantile(framework)
    else:
        return get_param_binary(framework)


def get_param_binary(framework):
    params = get_fixed_params(framework)
    params.update(get_hyper_params(framework))
    return params


def get_param_multiclass(framework, num_classes):
    return get_param_binary(framework)  # Use same hyperparameters as for binary classification for now.


def get_param_regression(framework):
    return get_param_binary(framework)  # Use same hyperparameters as for binary classification for now.


def get_param_quantile(framework):
    if framework != "pytorch":
        raise ValueError("Only pytorch tabular neural network is currently supported for quantile regression.")
    params = get_fixed_params(framework)
    params.update(get_quantile_hyper_params(framework))
    return params


def merge_framework_params(framework, shared_params, pytorch_params):
    if framework == "pytorch":
        shared_params.update(pytorch_params)
    else:
        raise ValueError("framework must be 'pytorch'")
    return shared_params
