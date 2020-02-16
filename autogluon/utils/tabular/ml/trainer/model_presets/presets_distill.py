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
        name_suffix = '_d' + str(distill_level)
        models = get_preset_models_softclass(path=path, hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune, name_suffix=name_suffix)
        return models
    elif problem_type != REGRESSION:
        objective_func = mean_squared_error
        stopping_metric = None
        problem_type = REGRESSION
    _ = hyperparameters.pop('NN', None)
    name_suffix = '_d' + str(distill_level)
    models = get_preset_models_regression(path=path, problem_type=problem_type, objective_func=objective_func, stopping_metric=stopping_metric, hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune, name_suffix=name_suffix)
    nn_options = {'num_epochs': 500, 'dropout_prob': 0, 'weight_decay': 1e-7, 'epochs_wo_improve': 50, 'layers': [2048]*4, 'numeric_embed_dim': 2048, 'activation': 'softrelu', 'embedding_size_factor': 2.0}
    nn_model = TabularNeuralNetModel(path=path, name='NeuralNetRegressor'+name_suffix, problem_type=problem_type,
                              objective_func=objective_func, stopping_metric=stopping_metric, hyperparameters=nn_options.copy())
    return [nn_model] + models
