import logging

from ...constants import BINARY, MULTICLASS, REGRESSION
from ...models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from ....metrics import mean_squared_error
from ...models.rf.rf_model import RFModel
from .presets import get_preset_models_regression

logger = logging.getLogger(__name__)


def get_preset_models_distillation(path, problem_type, objective_func, stopping_metric=None, num_classes=None,
                      hyperparameters={'NN':{},'GBM':{}}, hyperparameter_tune=False, distill_level=1):
    if problem_type == MULTICLASS:
        raise NotImplementedError
    elif problem_type != REGRESSION:
        objective_func = mean_squared_error
        stopping_metric = None
        problem_type = REGRESSION
    name_suffix = '_d' + str(distill_level)
    models = get_preset_models_regression(path=path, problem_type=problem_type, objective_func=objective_func, stopping_metric=stopping_metric, hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune, name_suffix=name_suffix)
    return models
