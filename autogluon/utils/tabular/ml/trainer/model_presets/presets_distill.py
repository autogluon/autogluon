import logging

from ...constants import BINARY, MULTICLASS, REGRESSION
from ...models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from ....metrics import mean_squared_error, log_loss
from ...models.rf.rf_model import RFModel
from .presets import get_preset_models, get_preset_models_softclass

logger = logging.getLogger(__name__)

# Higher values indicate higher priority, priority dictates the order models are trained (which matters if time runs out).
DEFAULT_DISTILL_PRIORITY = dict(
    GBM=100,
    NN=90,
    CAT=80,
    RF=70,
    custom=0,
)


def get_preset_models_distillation(path, problem_type, eval_metric, hyperparameters, stopping_metric=None, num_classes=None,
                                   hyperparameter_tune=False, distill_level=0, name_suffix='_DSTL'):
    if problem_type == MULTICLASS:
        models = get_preset_models_softclass(path=path, num_classes=num_classes, hyperparameters=hyperparameters,
                                             hyperparameter_tune=hyperparameter_tune, name_suffix=name_suffix)
    elif problem_type == BINARY:  # convert to regression in distillation
        eval_metric = mean_squared_error
        stopping_metric = mean_squared_error
        # Constrain output-range of NN:
        nn_outputrange = {'y_range': (0.0,1.0), 'y_range_extend': 0.0}
        if 'NN' in hyperparameters:
            nn_hyperparameters = hyperparameters['NN']
        elif 'default' in hyperparameters and 'NN' in hyperparameters['default']:
            nn_hyperparameters = hyperparameters['default']['NN']
        else:
            nn_hyperparameters = None
        if isinstance(nn_hyperparameters, list):
            for i in range(len(nn_hyperparameters)):
                nn_hyperparameters[i].update(nn_outputrange)
        elif nn_hyperparameters is not None:
            nn_hyperparameters.update(nn_outputrange)
        if 'NN' in hyperparameters:
            hyperparameters['NN'] = nn_hyperparameters
        elif 'default' in hyperparameters and 'NN' in hyperparameters['default']:
            hyperparameters['default']['NN'] = nn_hyperparameters
        # Swap RF criterion for MSE:
        rf_newparams = {'criterion': 'mse', 'AG_args': {'name_suffix': 'MSE'}}
        if 'RF' in hyperparameters:
            rf_hyperparameters = hyperparameters['RF']
        elif 'default' in hyperparameters and 'RF' in hyperparameters['default']:
            rf_hyperparameters = hyperparameters['default']['RF']
        else:
            rf_hyperparameters = None
        if isinstance(rf_hyperparameters, list):
            for i in range(len(rf_hyperparameters)):
                rf_hyperparameters[i].update(rf_newparams)
            rf_hyperparameters = [j for n, j in enumerate(rf_hyperparameters) if j not in rf_hyperparameters[(n+1):]]  # Remove duplicates which may arise after overwriting criterion
        elif rf_hyperparameters is not None:
            rf_hyperparameters.update(rf_newparams)
        if 'RF' in hyperparameters:
            hyperparameters['RF'] = rf_hyperparameters
        elif 'default' in hyperparameters and 'RF' in hyperparameters['default']:
            hyperparameters['default']['RF'] = rf_hyperparameters

    if problem_type == REGRESSION or problem_type == BINARY:
        models = get_preset_models(path=path, problem_type=REGRESSION, eval_metric=eval_metric, stopping_metric=stopping_metric,
                                   hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune, name_suffix=name_suffix, default_priorities=DEFAULT_DISTILL_PRIORITY)

    if problem_type in [MULTICLASS, BINARY]:
        for model in models:
            model.normalize_pred_probas = True
            model.name = model.name.replace('Regressor', 'Classifier')  # conceal from user that model may actually be a regressor.
            model.name = model.name.replace('SoftClassifier', 'Classifier')

    logger.log(20, f"Distilling with each of these student models: {[model.name for model in models]}")
    return models
