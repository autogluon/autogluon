import copy
import inspect
import logging
from collections import defaultdict

from autogluon.core.metrics import soft_log_loss, mean_squared_error
from autogluon.core.constants import AG_ARGS, AG_ARGS_FIT, AG_ARGS_ENSEMBLE, BINARY, MULTICLASS, REGRESSION, SOFTCLASS
from autogluon.core.models import AbstractModel, GreedyWeightedEnsembleModel, StackerEnsembleModel

from .presets_custom import get_preset_custom
from ..utils import process_hyperparameters
from ...models import LGBModel, CatBoostModel, XGBoostModel, RFModel, XTModel, KNNModel, LinearModel,\
    TabularNeuralNetModel, NNFastAiTabularModel, FastTextModel, TextPredictionV1Model
from ...models.tab_transformer.tab_transformer_model import TabTransformerModel

logger = logging.getLogger(__name__)

# Higher values indicate higher priority, priority dictates the order models are trained for a given level.
DEFAULT_MODEL_PRIORITY = dict(
    RF=100,
    XT=90,
    KNN=80,
    GBM=70,
    CAT=60,
    XGB=55,
    NN=50,
    FASTAI=45,
    LR=40,
    FASTTEXT=0,
    TEXT_NN_V1=0,
    TRANSF=0,
    custom=0,
)

# Problem type specific model priority overrides (will update default values in DEFAULT_MODEL_PRIORITY)
PROBLEM_TYPE_MODEL_PRIORITY = {
    MULTICLASS: dict(
        NN=120,
        FASTAI=115,
        KNN=110,
    ),
}

DEFAULT_SOFTCLASS_PRIORITY = dict(
    GBM=100,
    NN=90,
    RF=80,
    CAT=60,
    custom=0,
)

DEFAULT_CUSTOM_MODEL_PRIORITY = 0

MODEL_TYPES = dict(
    RF=RFModel,
    XT=XTModel,
    KNN=KNNModel,
    GBM=LGBModel,
    CAT=CatBoostModel,
    XGB=XGBoostModel,
    NN=TabularNeuralNetModel,
    LR=LinearModel,
    FASTAI=NNFastAiTabularModel,
    TRANSF=TabTransformerModel,
    TEXT_NN_V1=TextPredictionV1Model,
    FASTTEXT=FastTextModel,
    ENS_WEIGHTED=GreedyWeightedEnsembleModel,
)

DEFAULT_MODEL_NAMES = {
    RFModel: 'RandomForest',
    XTModel: 'ExtraTrees',
    KNNModel: 'KNeighbors',
    LGBModel: 'LightGBM',
    CatBoostModel: 'CatBoost',
    XGBoostModel: 'XGBoost',
    TabularNeuralNetModel: 'NeuralNetMXNet',
    LinearModel: 'LinearModel',
    NNFastAiTabularModel: 'NeuralNetFastAI',
    TabTransformerModel: 'Transformer',
    TextPredictionV1Model: 'TextNeuralNetV1',
    FastTextModel: 'FastText',
    GreedyWeightedEnsembleModel: 'WeightedEnsemble',
}


VALID_AG_ARGS_KEYS = {
    'name',
    'name_main',
    'name_prefix',
    'name_suffix',
    'name_bag_suffix',
    'model_type',
    'priority',
    'problem_types',
    'disable_in_hpo',
    'valid_stacker',
    'valid_base',
    'hyperparameter_tune_kwargs',
}


# DONE: Add levels, including 'default'
# DONE: Add lists
# DONE: Add custom which can append to lists
# DONE: Add special optional AG args for things like name prefix, name suffix, name, etc.
# DONE: Move creation of stack ensemble internally into this function? Requires passing base models in as well.
# DONE: Add special optional AG args for training order
# DONE: Add special optional AG args for base models
# TODO: Consider making hyperparameters arg in fit() accept lists, concatenate hyperparameter sets together.
# TODO: Consider adding special optional AG args for #cores,#gpus,num_early_stopping_iterations,etc.
# DONE: Consider adding special optional AG args for max train time, max memory size, etc.
# TODO: Consider adding special optional AG args for use_original_features,features_to_use,etc.
# TODO: Consider adding optional AG args to dynamically disable models such as valid_num_classes_range, valid_row_count_range, valid_feature_count_range, etc.
# TODO: Args such as max_repeats, num_folds
# DONE: Add banned_model_types arg
# TODO: Add option to update hyperparameters with only added keys, so disabling CatBoost would just be {'CAT': []}, which keeps the other models as is.
# TODO: special optional AG arg for only training model if eval_metric in list / not in list. Useful for F1 and 'is_unbalanced' arg in LGBM.
def get_preset_models(path, problem_type, eval_metric, hyperparameters, feature_metadata=None, num_classes=None,
                      level: int = 0, ensemble_type=StackerEnsembleModel, ensemble_kwargs: dict = None, ag_args_fit=None, ag_args=None, ag_args_ensemble=None,
                      name_suffix: str = None, default_priorities=None, invalid_model_names: list = None, excluded_model_types: list = None,
                      hyperparameter_preprocess_func=None, hyperparameter_preprocess_kwargs=None, silent=True):
    hyperparameters = process_hyperparameters(hyperparameters)
    if hyperparameter_preprocess_func is not None:
        if hyperparameter_preprocess_kwargs is None:
            hyperparameter_preprocess_kwargs = dict()
        hyperparameters = hyperparameter_preprocess_func(hyperparameters, **hyperparameter_preprocess_kwargs)
    if problem_type not in [BINARY, MULTICLASS, REGRESSION, SOFTCLASS]:
        raise NotImplementedError
    invalid_name_set = set()
    if invalid_model_names is not None:
        invalid_name_set.update(invalid_model_names)
    invalid_type_set = set()
    if excluded_model_types is not None:
        logger.log(20, f'Excluded Model Types: {excluded_model_types}')
        invalid_type_set.update(excluded_model_types)
    if default_priorities is None:
        default_priorities = copy.deepcopy(DEFAULT_MODEL_PRIORITY)
        if problem_type in PROBLEM_TYPE_MODEL_PRIORITY:
            default_priorities.update(PROBLEM_TYPE_MODEL_PRIORITY[problem_type])

    level_key = level if level in hyperparameters.keys() else 'default'
    if level_key not in hyperparameters.keys() and level_key == 'default':
        hyperparameters = {'default': hyperparameters}
    hp_level = hyperparameters[level_key]
    model_cfg_priority_dict = defaultdict(list)
    for model_type in hp_level:
        models_of_type = hp_level[model_type]
        if not isinstance(models_of_type, list):
            models_of_type = [models_of_type]
        model_cfgs_to_process = []
        for model_cfg in models_of_type:
            if model_type in invalid_type_set:
                logger.log(20, f"\tFound '{model_type}' model in hyperparameters, but '{model_type}' is present in `excluded_model_types` and will be removed.")
                continue  # Don't include excluded models
            if isinstance(model_cfg, str):
                model_cfgs_to_process += get_preset_custom(name=model_cfg, problem_type=problem_type, num_classes=num_classes)
            else:
                model_cfgs_to_process.append(model_cfg)
        for model_cfg in model_cfgs_to_process:
            model_cfg = clean_model_cfg(model_cfg=model_cfg, model_type=model_type, ag_args=ag_args, ag_args_ensemble=ag_args_ensemble, ag_args_fit=ag_args_fit)
            model_cfg[AG_ARGS]['priority'] = model_cfg[AG_ARGS].get('priority', default_priorities.get(model_type, DEFAULT_CUSTOM_MODEL_PRIORITY))
            model_priority = model_cfg[AG_ARGS]['priority']
            # Check if model_cfg is valid
            is_valid = is_model_cfg_valid(model_cfg, level=level, problem_type=problem_type)
            if AG_ARGS_FIT in model_cfg and not model_cfg[AG_ARGS_FIT]:
                model_cfg.pop(AG_ARGS_FIT)
            if is_valid:
                model_cfg_priority_dict[model_priority].append(model_cfg)

    model_cfg_priority_list = [model for priority in sorted(model_cfg_priority_dict.keys(), reverse=True) for model in model_cfg_priority_dict[priority]]

    if not silent:
        logger.log(20, 'Model configs that will be trained (in order):')
    models = []
    model_args_fit = {}
    for model_cfg in model_cfg_priority_list:
        model = model_factory(model_cfg, path=path, problem_type=problem_type, eval_metric=eval_metric,
                              num_classes=num_classes, name_suffix=name_suffix, ensemble_type=ensemble_type, ensemble_kwargs=ensemble_kwargs,
                              invalid_name_set=invalid_name_set, level=level, feature_metadata=feature_metadata)
        invalid_name_set.add(model.name)
        if 'hyperparameter_tune_kwargs' in model_cfg[AG_ARGS]:
            model_args_fit[model.name] = {'hyperparameter_tune_kwargs': model_cfg[AG_ARGS]['hyperparameter_tune_kwargs']}
        if not silent:
            logger.log(20, f'\t{model.name}: \t{model_cfg}')
        models.append(model)
    return models, model_args_fit


def clean_model_cfg(model_cfg: dict, model_type=None, ag_args=None, ag_args_ensemble=None, ag_args_fit=None):
    model_cfg = copy.deepcopy(model_cfg)
    if AG_ARGS not in model_cfg:
        model_cfg[AG_ARGS] = dict()
    if 'model_type' not in model_cfg[AG_ARGS]:
        model_cfg[AG_ARGS]['model_type'] = model_type
    if model_cfg[AG_ARGS]['model_type'] is None:
        raise AssertionError(f'model_type was not specified for model! Model: {model_cfg}')
    model_type = model_cfg[AG_ARGS]['model_type']
    if not inspect.isclass(model_type):
        model_type = MODEL_TYPES[model_type]
    elif not issubclass(model_type, AbstractModel):
        logger.warning(f'Warning: Custom model type {model_type} does not inherit from {AbstractModel}. This may lead to instability. Consider wrapping {model_type} with an implementation of {AbstractModel}!')
    else:
        logger.log(20, f'Custom Model Type Detected: {model_type}')
    model_cfg[AG_ARGS]['model_type'] = model_type
    model_type_real = model_cfg[AG_ARGS]['model_type']
    if not inspect.isclass(model_type_real):
        model_type_real = MODEL_TYPES[model_type_real]
    default_ag_args = model_type_real._get_default_ag_args()
    if ag_args is not None:
        model_extra_ag_args = ag_args.copy()
        model_extra_ag_args.update(model_cfg[AG_ARGS])
        model_cfg[AG_ARGS] = model_extra_ag_args
    if ag_args_ensemble is not None:
        model_extra_ag_args_ensemble = ag_args_ensemble.copy()
        model_extra_ag_args_ensemble.update(model_cfg.get(AG_ARGS_ENSEMBLE, dict()))
        model_cfg[AG_ARGS_ENSEMBLE] = model_extra_ag_args_ensemble
    if ag_args_fit is not None:
        if AG_ARGS_FIT not in model_cfg:
            model_cfg[AG_ARGS_FIT] = dict()
        model_extra_ag_args_fit = ag_args_fit.copy()
        model_extra_ag_args_fit.update(model_cfg[AG_ARGS_FIT])
        model_cfg[AG_ARGS_FIT] = model_extra_ag_args_fit
    if default_ag_args is not None:
        default_ag_args.update(model_cfg[AG_ARGS])
        model_cfg[AG_ARGS] = default_ag_args
    return model_cfg


# Check if model is valid
def is_model_cfg_valid(model_cfg, level=0, problem_type=None):
    is_valid = True
    for key in model_cfg.get(AG_ARGS, {}):
        if key not in VALID_AG_ARGS_KEYS:
            logger.warning(f'WARNING: Unknown ag_args key: {key}')
    if AG_ARGS not in model_cfg:
        is_valid = False  # AG_ARGS is required
    elif model_cfg[AG_ARGS].get('model_type', None) is None:
        is_valid = False  # model_type is required
    elif model_cfg[AG_ARGS].get('hyperparameter_tune_kwargs', None) and model_cfg[AG_ARGS].get('disable_in_hpo', False):
        is_valid = False
    elif not model_cfg[AG_ARGS].get('valid_stacker', True) and level > 0:
        is_valid = False  # Not valid as a stacker model
    elif not model_cfg[AG_ARGS].get('valid_base', True) and level == 0:
        is_valid = False  # Not valid as a base model
    elif problem_type is not None and problem_type not in model_cfg[AG_ARGS].get('problem_types', [problem_type]):
        is_valid = False  # Not valid for this problem_type
    return is_valid


def model_factory(
        model, path, problem_type, eval_metric, num_classes=None,
        name_suffix=None, ensemble_type=StackerEnsembleModel, ensemble_kwargs=None,
        invalid_name_set=None, level=0, feature_metadata=None,
):
    if invalid_name_set is None:
        invalid_name_set = set()
    model_type = model[AG_ARGS]['model_type']
    if not inspect.isclass(model_type):
        model_type = MODEL_TYPES[model_type]
    name_orig = model[AG_ARGS].get('name', None)
    if name_orig is None:
        name_main = model[AG_ARGS].get('name_main', DEFAULT_MODEL_NAMES.get(model_type, model_type.__name__))
        name_prefix = model[AG_ARGS].get('name_prefix', '')
        name_suff = model[AG_ARGS].get('name_suffix', '')
        name_orig = name_prefix + name_main + name_suff
    if name_suffix is not None:
        name_orig = name_orig + name_suffix
    name = name_orig
    name_stacker = None
    num_increment = 2
    if ensemble_kwargs is None:
        while name in invalid_name_set:  # Ensure name is unique
            name = f'{name_orig}_{num_increment}'
            num_increment += 1
    else:
        name_bag_suffix = model[AG_ARGS].get('name_bag_suffix', '_BAG')
        name_stacker = f'{name}{name_bag_suffix}_L{level}'
        while name_stacker in invalid_name_set:  # Ensure name is unique
            name = f'{name_orig}_{num_increment}'
            name_stacker = f'{name}{name_bag_suffix}_L{level}'
            num_increment += 1
    model_params = copy.deepcopy(model)
    model_params.pop(AG_ARGS, None)
    model_params.pop(AG_ARGS_ENSEMBLE, None)
    model_init = model_type(path=path, name=name, problem_type=problem_type, eval_metric=eval_metric, num_classes=num_classes, hyperparameters=model_params, feature_metadata=feature_metadata)

    if ensemble_kwargs is not None:
        ensemble_kwargs_model = copy.deepcopy(ensemble_kwargs)
        extra_ensemble_hyperparameters = copy.deepcopy(model.get(AG_ARGS_ENSEMBLE, dict()))
        ensemble_kwargs_model['hyperparameters'] = ensemble_kwargs_model.get('hyperparameters', {})
        if ensemble_kwargs_model['hyperparameters'] is None:
            ensemble_kwargs_model['hyperparameters'] = {}
        ensemble_kwargs_model['hyperparameters'].update(extra_ensemble_hyperparameters)
        model_init = ensemble_type(path=path, name=name_stacker, model_base=model_init, num_classes=num_classes, **ensemble_kwargs_model)

    return model_init


# TODO: v0.1 cleanup and avoid hardcoded logic with model names
def get_preset_models_softclass(path, hyperparameters, feature_metadata, num_classes=None, name_suffix='', ag_args=None, invalid_model_names: list = None):
    model_types_standard = ['GBM', 'NN', 'CAT']
    hyperparameters = copy.deepcopy(hyperparameters)
    hyperparameters_standard = copy.deepcopy(hyperparameters)
    hyperparameters_rf = copy.deepcopy(hyperparameters)
    default_level_key = 'default'
    if default_level_key in hyperparameters:
        hyperparameters_standard[default_level_key] = {key: hyperparameters_standard[default_level_key][key] for key in hyperparameters_standard[default_level_key] if key in model_types_standard}
        hyperparameters_rf[default_level_key] = {key: hyperparameters_rf[default_level_key][key] for key in hyperparameters_rf[default_level_key] if key == 'RF'}
    else:
        hyperparameters_standard = {key: hyperparameters_standard[key] for key in hyperparameters_standard if key in model_types_standard}
        hyperparameters_rf = {key: hyperparameters_rf[key] for key in hyperparameters_rf if key == 'RF'}
        # TODO: add support for per-stack level hyperparameters
    models, model_args_fit = get_preset_models(path=path, problem_type=SOFTCLASS, feature_metadata=feature_metadata, eval_metric=soft_log_loss, ag_args_fit={'stopping_metric': soft_log_loss},
                               hyperparameters=hyperparameters_standard, num_classes=num_classes,
                               ag_args=ag_args, name_suffix=name_suffix, default_priorities=DEFAULT_SOFTCLASS_PRIORITY, invalid_model_names=invalid_model_names)
    if invalid_model_names is None:
        invalid_model_names = []
    invalid_model_names = invalid_model_names + [model.name for model in models]
    # Swap RF criterion for MSE:
    rf_models = []
    if len(hyperparameters_rf) > 0:
        rf_newparams = {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE'}}
        if 'RF' in hyperparameters_rf:
            rf_params = hyperparameters_rf['RF']
        elif 'default' in hyperparameters_rf and 'RF' in hyperparameters_rf['default']:
            rf_params = hyperparameters_rf['default']['RF']
        else:
            rf_params = None
        if isinstance(rf_params, list):
            for i in range(len(rf_params)):
                rf_params[i].update(rf_newparams)
            rf_params = [j for n, j in enumerate(rf_params) if j not in rf_params[(n+1):]]  # Remove duplicates which may arise after overwriting criterion
        elif rf_params is not None:
            rf_params.update(rf_newparams)
        if 'RF' in hyperparameters_rf:
            hyperparameters_rf['RF'] = rf_params
        elif 'default' in hyperparameters_rf and 'RF' in hyperparameters_rf['default']:
            hyperparameters_rf['default']['RF'] = rf_params
        rf_models, model_args_fit_rf = get_preset_models(path=path, problem_type=REGRESSION, feature_metadata=feature_metadata, eval_metric=mean_squared_error,
                                      hyperparameters=hyperparameters_rf,
                                      ag_args=ag_args, name_suffix=name_suffix, default_priorities=DEFAULT_SOFTCLASS_PRIORITY, invalid_model_names=invalid_model_names)
        model_args_fit.update(model_args_fit_rf)
    models_cat = [model for model in models if isinstance(model, CatBoostModel)]
    models_noncat = [model for model in models if not isinstance(model, CatBoostModel)]
    models = models_noncat + rf_models + models_cat
    if len(models) == 0:
        raise ValueError("At least one of the following model-types must be present in hyperparameters: ['GBM','CAT','NN','RF'], "
                         "These are the only supported models for softclass prediction problems. "
                         "Softclass problems are also not yet supported for fit() with per-stack level hyperparameters.")
    for model in models:
        model.normalize_pred_probas = True

    return models, model_args_fit
