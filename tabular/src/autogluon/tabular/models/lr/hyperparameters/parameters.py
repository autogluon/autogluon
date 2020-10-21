import logging

from sklearn.linear_model import LogisticRegression, Ridge, Lasso

from ....constants import BINARY
from ....constants import REGRESSION

L1 = 'L1'
L2 = 'L2'
IGNORE = 'ignore'
ONLY = 'only'
INCLUDE = 'include'

logger = logging.getLogger(__name__)


def get_param_baseline():
    default_params = {
        'C': 1,
        'vectorizer_dict_size': 75000,  # size of TFIDF vectorizer dictionary; used only in text model
        'proc.ngram_range': (1, 5),  # range of n-grams for TFIDF vectorizer dictionary; used only in text model
        'proc.skew_threshold': 0.99,  # numerical features whose absolute skewness is greater than this receive special power-transform preprocessing. Choose big value to avoid using power-transforms
        'proc.impute_strategy': 'median',  # strategy argument of sklearn.SimpleImputer() used to impute missing numeric values
        'penalty': L2,  # regularization to use with regression models
        'handle_text': IGNORE, # how text should be handled: `ignore` - don't use NLP features; `only` - only use NLP features; `include` - use both regular and NLP features
    }
    return default_params


def get_model_params(problem_type: str, hyperparameters):
    penalty = hyperparameters.get('penalty', L2)
    handle_text = hyperparameters.get('handle_text', IGNORE)
    if problem_type == REGRESSION:
        if penalty == L2:
            model_class = Ridge
        elif penalty == L1:
            model_class = Lasso
        else:
            logger.warning('Unknown value for penalty {} - supported types are [l1, l2] - falling back to l2'.format(penalty))
            penalty = L2
            model_class = Ridge
    else:
        model_class = LogisticRegression

    return model_class, penalty, handle_text


def get_default_params(problem_type: str, penalty: str):
    # TODO: get seed from seeds provider
    if problem_type == REGRESSION:
        default_params = {'C': None, 'random_state': 0, 'fit_intercept': True}
        if penalty == L2:
            default_params['solver'] = 'auto'
    else:
        default_params = {'C': None, 'random_state': 0, 'solver': _get_solver(problem_type), 'n_jobs': -1, 'fit_intercept': True}
    model_params = list(default_params.keys())
    return model_params, default_params


def _get_solver(problem_type):
    if problem_type == BINARY:
        # TODO explore using liblinear for smaller datasets
        solver = 'lbfgs'
    else:
        solver = 'lbfgs'
    return solver
