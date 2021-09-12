import logging

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import mean_squared_error
from autogluon.core.trainer.utils import process_hyperparameters

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


def get_preset_models_distillation(path, problem_type, eval_metric, hyperparameters,
                                   level=1, name_suffix='_DSTL', invalid_model_names: list = None, **kwargs):
    hyperparameters = process_hyperparameters(hyperparameters)
    level_key = level if level in hyperparameters.keys() else 'default'
    if level_key not in hyperparameters.keys() and level_key == 'default':
        hyperparameters = {'default': hyperparameters}
    hyperparameters = hyperparameters[level_key]
    if problem_type == BINARY:  # convert to regression in distillation
        eval_metric = mean_squared_error
        # Constrain output-range of NN:
        nn_outputrange = {'y_range': (0.0,1.0), 'y_range_extend': 0.0}
        if 'NN' in hyperparameters:
            nn_hyperparameters = hyperparameters['NN']
        else:
            nn_hyperparameters = None
        if isinstance(nn_hyperparameters, list):
            for i in range(len(nn_hyperparameters)):
                nn_hyperparameters[i].update(nn_outputrange)
        elif nn_hyperparameters is not None:
            nn_hyperparameters.update(nn_outputrange)
        if 'NN' in hyperparameters:
            hyperparameters['NN'] = nn_hyperparameters
        # Swap RF criterion for MSE:
        rf_newparams = {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE'}}
        if 'RF' in hyperparameters:
            rf_hyperparameters = hyperparameters['RF']
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

    if problem_type == MULTICLASS:
        models, model_args_fit = get_preset_models_softclass(path=path, hyperparameters=hyperparameters, level=level,
                                                             name_suffix=name_suffix, invalid_model_names=invalid_model_names, **kwargs)
    else:  # BINARY or REGRESSION
        models, model_args_fit = get_preset_models(path=path, problem_type=REGRESSION, eval_metric=eval_metric, hyperparameters=hyperparameters, level=level,
                                                   name_suffix=name_suffix, default_priorities=DEFAULT_DISTILL_PRIORITY, invalid_model_names=invalid_model_names, **kwargs)

    if problem_type in [MULTICLASS, BINARY]:
        for model in models:
            model.normalize_pred_probas = True

    logger.log(20, f"Distilling with each of these student models: {[model.name for model in models]}")
    return models, model_args_fit
