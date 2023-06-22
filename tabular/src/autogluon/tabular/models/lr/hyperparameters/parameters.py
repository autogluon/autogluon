import logging

from autogluon.core.constants import BINARY

IGNORE = "ignore"
ONLY = "only"
INCLUDE = "include"

logger = logging.getLogger(__name__)

preprocess_params_set = {
    "vectorizer_dict_size",
    "proc.ngram_range",
    "proc.skew_threshold",
    "proc.impute_strategy",
    "penalty",
    "handle_text",
}


def get_param_baseline():
    default_params = {
        "C": 1,
        "vectorizer_dict_size": 75000,  # size of TFIDF vectorizer dictionary; used only in text model
        "proc.ngram_range": (1, 5),  # range of n-grams for TFIDF vectorizer dictionary; used only in text model
        "proc.skew_threshold": 0.99,  # numerical features whose absolute skewness is greater than this receive special power-transform preprocessing. Choose big value to avoid using power-transforms
        "proc.impute_strategy": "median",  # strategy argument of sklearn.SimpleImputer() used to impute missing numeric values
        "penalty": "L2",  # regularization to use with regression models
        "handle_text": IGNORE,  # how text should be handled: `ignore` - don't use NLP features; `only` - only use NLP features; `include` - use both regular and NLP features
    }
    return default_params


def _get_solver(problem_type):
    if problem_type == BINARY:
        # TODO explore using liblinear for smaller datasets
        solver = "lbfgs"
    else:
        solver = "lbfgs"
    return solver
