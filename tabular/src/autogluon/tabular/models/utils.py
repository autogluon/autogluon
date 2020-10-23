
import logging

from autogluon.core import Space, Categorical, List, NestedSpace

logger = logging.getLogger(__name__)


# Methods useful for all models:
def fixedvals_from_searchspaces(params):
    """Converts any search space hyperparams in params dict into fixed default values."""
    if any(isinstance(params[hyperparam], Space) for hyperparam in params):
        logger.warning("Attempting to fit model without HPO, but search space is provided. fit() will only consider default hyperparameter values from search space.")
        bad_keys = [hyperparam for hyperparam in params if isinstance(params[hyperparam], Space)][:]  # delete all keys which are of type autogluon Space
        params = params.copy()
        for hyperparam in bad_keys:
            params[hyperparam] = hp_default_value(params[hyperparam])
        return params
    else:
        return params


def hp_default_value(hp_value):
    """Extracts default fixed value from hyperparameter search space hp_value to use a fixed value instead of a search space."""
    if not isinstance(hp_value, Space):
        return hp_value
    if isinstance(hp_value, Categorical):
        return hp_value[0]
    elif isinstance(hp_value, List):
        return [z[0] for z in hp_value]
    elif isinstance(hp_value, NestedSpace):
        raise ValueError("Cannot extract default value from NestedSpace. Please specify fixed value instead of: %s" % str(hp_value))
    else:
        return hp_value.get_hp('dummy_name').default_value
