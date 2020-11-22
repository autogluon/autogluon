from ...models.gluonts_model.mqcnn.mqcnn_model import MQCNNModel
from ...models.abstract.abstract_model import AbstractModel
import copy
from collections import defaultdict
from ...constants import AG_ARGS, AG_ARGS_FIT
import inspect
import logging

logger = logging.getLogger(__name__)


MODEL_TYPES = dict(
    MQCNN=MQCNNModel
)

DEFAULT_MODEL_NAMES = {
    MQCNNModel: "MQCNN"
}

DEFAULT_MODEL_PRIORITY = dict(
    MQCNN=50
)

DEFAULT_CUSTOM_MODEL_PRIORITY = 0


def get_preset_models(path, eval_metric, hyperparameters, freq, prediction_length, hyperparameter_tune=False,
                    level="default", name_suffix='', invalid_model_names: list = None, default_priorities=None):

    preset_models = []
    for model, model_hps in hyperparameters.items():
        model_type = MODEL_TYPES[model]
        preset_models.append(model_type(path=path, freq=freq, prediction_length=prediction_length, eval_metric=eval_metric, hyperparameters=model_hps))
    return preset_models
    # TODO:Try to follow tabular's code
    # if default_priorities is None:
    #     default_priorities = copy.deepcopy(DEFAULT_MODEL_PRIORITY)
    #
    # if level in hyperparameters.keys():
    #     level_key = level
    # else:
    #     level_key = 'default'
    # print(hyperparameters)
    # hp_level = hyperparameters[level_key]
    # priority_dict = defaultdict(list)
    # for model_type in hp_level:
    #     for model in hp_level[model_type]:
    #         model = copy.deepcopy(model)
    #         if AG_ARGS not in model:
    #             model[AG_ARGS] = dict()
    #         if 'model_type' not in model[AG_ARGS]:
    #             model[AG_ARGS]['model_type'] = model_type
    #         model_priority = model[AG_ARGS].get('priority', default_priorities.get(model_type, DEFAULT_CUSTOM_MODEL_PRIORITY))
    #         # Check if model is valid
    #         if hyperparameter_tune and model[AG_ARGS].get('disable_in_hpo', False):
    #             continue  # Not valid
    #         priority_dict[model_priority].append(model)
    #
    # model_names_set = set()
    # model_priority_list = [model for priority in sorted(priority_dict.keys(), reverse=True) for model in priority_dict[priority]]
    # models = []
    # for model in model_priority_list:
    #     model_type = model[AG_ARGS]['model_type']
    #     if not inspect.isclass(model_type):
    #         model_type = MODEL_TYPES[model_type]
    #     elif not issubclass(model_type, AbstractModel):
    #         logger.warning(f'Warning: Custom model type {model_type} does not inherit from {AbstractModel}. This may lead to instability. Consider wrapping {model_type} with an implementation of {AbstractModel}!')
    #     else:
    #         logger.log(20, f'Custom Model Type Detected: {model_type}')
    #     name_orig = model[AG_ARGS].get('name', None)
    #     if name_orig is None:
    #         name_main = model[AG_ARGS].get('name_main', DEFAULT_MODEL_NAMES.get(model_type, model_type.__name__))
    #         name_prefix = model[AG_ARGS].get('name_prefix', '')
    #         name_suff = model[AG_ARGS].get('name_suffix', '')
    #         name_orig = name_prefix + name_main + name_suff
    #
    #     name_orig = name_orig + name_suffix
    #     name = name_orig
    #     num_increment = 2
    #     while name in model_names_set:  # Ensure name is unique
    #         name = f'{name_orig}_{num_increment}'
    #         num_increment += 1
    #     model_names_set.add(name)
    #     model_params = copy.deepcopy(model)
    #     model_params.pop(AG_ARGS)
    #
    #     model_init = model_type(path=path, name=name, eval_metric=eval_metric,
    #                             hyperparameters=model_params, freq=freq, prediction_length=prediction_length)
    #     models.append(model_init)
    #
    # return models
