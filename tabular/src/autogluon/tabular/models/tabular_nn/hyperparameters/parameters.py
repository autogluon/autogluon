""" Default (fixed) hyperparameter values used in Neural network model """

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE


def get_fixed_params():
    """ Parameters that currently cannot be searched during HPO """
    fixed_params = {
        'num_epochs': 500,  # maximum number of epochs for training NN
        'epochs_wo_improve': 20,  # we terminate training if validation performance hasn't improved in the last 'epochs_wo_improve' # of epochs
        # TODO: Epochs could take a very long time, we may want smarter logic than simply # of epochs without improvement (slope, difference in score, etc.)
        'seed_value': None,  # random seed for reproducibility (set = None to ignore)
    }
    return fixed_params


def get_hyper_params():
    """ Parameters that currently can be tuned during HPO """
    hyper_params = {
        ## Hyperparameters for neural net architecture:
        'network_type': 'widedeep',  # Type of neural net used to produce predictions
        # Options: ['widedeep', 'feedforward']
        'layers': None,  # List of widths (num_units) for each hidden layer (Note: only specifies hidden layers. These numbers are not absolute, they will also be scaled based on number of training examples and problem type)
        # Options: List of lists that are manually created
        'numeric_embed_dim': None,  # Size of joint embedding for all numeric+one-hot features.
        # Options: integer values between 10-10000
        'activation': 'relu',  # Activation function
        # Options: ['relu', 'softrelu', 'tanh', 'softsign']
        'max_layer_width': 2056,  # maximum number of hidden units in network layer (integer > 0)
        # Does not need to be searched by default
        'embedding_size_factor': 1.0,  # scaling factor to adjust size of embedding layers (float > 0)
        # Options: range[0.01 - 100] on log-scale
        'embed_exponent': 0.56,  # exponent used to determine size of embedding layers based on # categories.
        'max_embedding_dim': 100,  # maximum size of embedding layer for a single categorical feature (int > 0).
        ## Regression-specific hyperparameters:
        'y_range': None,  # Tuple specifying whether (min_y, max_y). Can be = (-np.inf, np.inf).
        # If None, inferred based on training labels. Note: MUST be None for classification tasks!
        'y_range_extend': 0.05,  # Only used to extend size of inferred y_range when y_range = None.
        ## Hyperparameters for neural net training:
        'use_batchnorm': True,  # whether or not to utilize Batch-normalization
        # Options: [True, False]
        'dropout_prob': 0.1,  # dropout probability, = 0 turns off Dropout.
        # Options: range(0.0, 0.5)
        'batch_size': 512,  # batch-size used for NN training
        # Options: [32, 64, 128. 256, 512, 1024, 2048]
        'loss_function': None,  # MXNet loss function minimized during training
        'optimizer': 'adam',  # MXNet optimizer to use.
        # Options include: ['adam','sgd']
        'learning_rate': 3e-4,  # learning rate used for NN training (float > 0)
        'weight_decay': 1e-6,  # weight decay regularizer (float > 0)
        'clip_gradient': 100.0,  # gradient clipping threshold (float > 0)
        'momentum': 0.9,  # momentum which is only used for SGD optimizer
        'lr_scheduler': None,  # If not None, string specifying what type of learning rate scheduler to use (may override learning_rate).
        # Options: [None, 'cosine', 'step', 'poly', 'constant']
        # Below are hyperparameters specific to the LR scheduler (only used if lr_scheduler != None). For more info, see: https://gluon-cv.mxnet.io/api/utils.html#gluoncv.utils.LRScheduler
        'base_lr': 3e-5,  # smallest LR (float > 0)
        'target_lr': 1.0,  # largest LR (float > 0)
        'lr_decay': 0.1,  # step factor used to decay LR (float in (0,1))
        'warmup_epochs': 10,  # number of epochs at beginning of training in which LR is linearly ramped up (float > 1).
        ## Hyperparameters for data processing:
        'proc.embed_min_categories': 4,  # apply embedding layer to categorical features with at least this many levels. Features with fewer levels are one-hot encoded. Choose big value to avoid use of Embedding layers
        # Options: [3,4,10, 100, 1000]
        'proc.impute_strategy': 'median',  # strategy argument of sklearn.SimpleImputer() used to impute missing numeric values
        # Options: ['median', 'mean', 'most_frequent']
        'proc.max_category_levels': 100,  # maximum number of allowed levels per categorical feature
        # Options: [10, 100, 200, 300, 400, 500, 1000, 10000]
        'proc.skew_threshold': 0.99,  # numerical features whose absolute skewness is greater than this receive special power-transform preprocessing. Choose big value to avoid using power-transforms
        # Options: [0.2, 0.3, 0.5, 0.8, 1.0, 10.0, 100.0]
        'use_ngram_features': False,  # If False, will drop automatically generated ngram features from language features. This results in worse model quality but far faster inference and training times.
        # Options: [True, False]
    }
    return hyper_params


def get_quantile_fixed_params():
    """ Parameters that currently cannot be searched during HPO """
    params = get_fixed_params()
    new_params = {
        'num_updates': 20000,  # maximum number of updates
        'updates_wo_improve': 1000,  # we terminate training if validation performance hasn't improved in the last 'updates_wo_improve' # of updates
        'max_batch_size': 512,  # max batch-size
    }
    params.update(new_params)
    return params


def get_quantile_hyper_params():
    """ Parameters that currently can be searched during HPO """
    hyper_params = get_hyper_params()
    new_hyper_params = {
        'num_layers': 2,  # number of layers
        # Options: [2, 3, 4, 5]
        'hidden_size': 64,  # hidden size
        # Options: [128, 256, 512]
        'gamma': 5.0,  # margin loss weight
        # Options: range(0.1, 10.0)
        'alpha': 0.01, # alpha for huber pinball loss
    }
    hyper_params.update(new_hyper_params)
    return hyper_params


# Note: params for original NNTabularModel were:
# weight_decay=0.01, dropout_prob = 0.1, batch_size = 2048, lr = 1e-2, epochs=30, layers= [200, 100] (semi-equivalent to our layers = [100],numeric_embed_dim=200)
def get_default_param(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary()
    elif problem_type == MULTICLASS:
        return get_param_multiclass(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_param_regression()
    elif problem_type == QUANTILE:
        return get_param_quantile()
    else:
        return get_param_binary()


def get_param_multiclass(num_classes):
    params = get_fixed_params()
    params.update(get_hyper_params())
    return params


def get_param_binary():
    params = get_fixed_params()
    params.update(get_hyper_params())
    return params


def get_param_regression():
    params = get_fixed_params()
    params.update(get_hyper_params())
    return params


def get_param_quantile():
    params = get_quantile_fixed_params()
    params.update(get_quantile_hyper_params())
    return params
