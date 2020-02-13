import logging

from ...constants import BINARY, MULTICLASS, REGRESSION
from ...models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from ....metrics import mean_squared_error, log_loss
from ...models.rf.rf_model import RFModel
from .presets import get_preset_models_regression, get_preset_models_softclass

logger = logging.getLogger(__name__)


def get_preset_models_distillation(path, problem_type, objective_func, stopping_metric=None, num_classes=None,
                      hyperparameters={'NN':{},'GBM':{}}, hyperparameter_tune=False, distill_level=1):
    if problem_type == MULTICLASS:
        # raise ValueError("Cannot distill multiclass prediction problem, you must specify problem_type=softclass instead.")
        models = get_preset_models_softclass(path=path, hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune)
        return models
    elif problem_type != REGRESSION:
        objective_func = mean_squared_error
        stopping_metric = None
        problem_type = REGRESSION
    name_suffix = '_d' + str(distill_level)
    models = get_preset_models_regression(path=path, problem_type=problem_type, objective_func=objective_func, stopping_metric=stopping_metric, hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune, name_suffix=name_suffix)
    return models
