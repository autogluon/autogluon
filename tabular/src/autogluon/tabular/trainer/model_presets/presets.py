import copy
import inspect
import logging
import warnings
from collections import defaultdict

from sklearn.linear_model import LogisticRegression, LinearRegression

from autogluon.core.metrics import soft_log_loss, mean_squared_error
from autogluon.core.constants import AG_ARGS, AG_ARGS_FIT, AG_ARGS_ENSEMBLE, BINARY, MULTICLASS,\
    REGRESSION, SOFTCLASS, PROBLEM_TYPES_CLASSIFICATION
from ...models.abstract.abstract_model import AbstractModel
from ...models.fastainn.tabular_nn_fastai import NNFastAiTabularModel
from ...models.lgb.lgb_model import LGBModel
from ...models.lr.lr_model import LinearModel
from ...models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from ...models.rf.rf_model import RFModel
from ...models.knn.knn_model import KNNModel
from ...models.catboost.catboost_model import CatBoostModel
from ...models.xgboost.xgboost_model import XGBoostModel
from ...models.xt.xt_model import XTModel
from ...models.tab_transformer.tab_transformer_model import TabTransformerModel
from ...models.fasttext.fasttext_model import FastTextModel
from ...models.text_prediction.text_prediction_v1_model import TextPredictionV1Model
from ...models.ensemble.stacker_ensemble_model import StackerEnsembleModel
from ...models.ensemble.greedy_weighted_ensemble_model import GreedyWeightedEnsembleModel

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
def get_preset_models(path, problem_type, eval_metric, hyperparameters, stopping_metric=None, feature_metadata=None, num_classes=None, hyperparameter_tune_kwargs=None,
                      level: int = 0, ensemble_type=StackerEnsembleModel, ensemble_kwargs: dict = None, ag_args_fit=None, ag_args=None, ag_args_ensemble=None,
                      name_suffix: str = None, default_priorities=None, invalid_model_names: list = None, hyperparameter_preprocess_func=None, hyperparameter_preprocess_kwargs=None):
    if hyperparameter_preprocess_func is not None:
        if hyperparameter_preprocess_kwargs is None:
            hyperparameter_preprocess_kwargs = dict()
        hyperparameters = hyperparameter_preprocess_func(hyperparameters, **hyperparameter_preprocess_kwargs)
    if problem_type not in [BINARY, MULTICLASS, REGRESSION, SOFTCLASS]:
        raise NotImplementedError
    if default_priorities is None:
        default_priorities = copy.deepcopy(DEFAULT_MODEL_PRIORITY)
        if problem_type in PROBLEM_TYPE_MODEL_PRIORITY:
            default_priorities.update(PROBLEM_TYPE_MODEL_PRIORITY[problem_type])

    if level in hyperparameters.keys():
        level_key = level
    else:
        level_key = 'default'
    hp_level = hyperparameters[level_key]
    priority_dict = defaultdict(list)
    for model_type in hp_level:
        for model in hp_level[model_type]:
            model = copy.deepcopy(model)
            if AG_ARGS not in model:
                model[AG_ARGS] = dict()
            if 'model_type' not in model[AG_ARGS]:
                model[AG_ARGS]['model_type'] = model_type
            model_type_real = model[AG_ARGS]['model_type']
            if not inspect.isclass(model_type_real):
                model_type_real = MODEL_TYPES[model_type_real]
            default_ag_args = model_type_real._get_default_ag_args()
            if ag_args is not None:
                model_extra_ag_args = ag_args.copy()
                model_extra_ag_args.update(model[AG_ARGS])
                model[AG_ARGS] = model_extra_ag_args
            if ag_args_ensemble is not None:
                model_extra_ag_args_ensemble = ag_args_ensemble.copy()
                model_extra_ag_args_ensemble.update(model.get(AG_ARGS_ENSEMBLE, dict()))
                model[AG_ARGS_ENSEMBLE] = model_extra_ag_args_ensemble
            if ag_args_fit is not None:
                if AG_ARGS_FIT not in model:
                    model[AG_ARGS_FIT] = dict()
                model_extra_ag_args_fit = ag_args_fit.copy()
                model_extra_ag_args_fit.update(model[AG_ARGS_FIT])
                model[AG_ARGS_FIT] = model_extra_ag_args_fit
            if default_ag_args is not None:
                default_ag_args.update(model[AG_ARGS])
                model[AG_ARGS] = default_ag_args
            model_priority = model[AG_ARGS].get('priority', default_priorities.get(model_type, DEFAULT_CUSTOM_MODEL_PRIORITY))
            # Check if model is valid
            if hyperparameter_tune_kwargs and model[AG_ARGS].get('disable_in_hpo', False):
                continue  # Not valid
            elif not model[AG_ARGS].get('valid_stacker', True) and level > 0:
                continue  # Not valid as a stacker model
            elif not model[AG_ARGS].get('valid_base', True) and level == 0:
                continue  # Not valid as a base model
            priority_dict[model_priority].append(model)
    model_priority_list = [model for priority in sorted(priority_dict.keys(), reverse=True) for model in priority_dict[priority]]
    model_names_set = set()
    if invalid_model_names is not None:
        model_names_set.update(invalid_model_names)
    models = []
    for model in model_priority_list:
        model_type = model[AG_ARGS]['model_type']
        if not inspect.isclass(model_type):
            model_type = MODEL_TYPES[model_type]
        elif not issubclass(model_type, AbstractModel):
            logger.warning(f'Warning: Custom model type {model_type} does not inherit from {AbstractModel}. This may lead to instability. Consider wrapping {model_type} with an implementation of {AbstractModel}!')
        else:
            logger.log(20, f'Custom Model Type Detected: {model_type}')
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
            while name in model_names_set:  # Ensure name is unique
                name = f'{name_orig}_{num_increment}'
                num_increment += 1
            model_names_set.add(name)
        else:
            name_bag_suffix = model[AG_ARGS].get('name_bag_suffix', '_BAG')
            name_stacker = f'{name}{name_bag_suffix}_L{level}'
            while name_stacker in model_names_set:  # Ensure name is unique
                name = f'{name_orig}_{num_increment}'
                name_stacker = f'{name}{name_bag_suffix}_L{level}'
                num_increment += 1
            model_names_set.add(name_stacker)
        model_params = copy.deepcopy(model)
        model_params.pop(AG_ARGS, None)
        model_params.pop(AG_ARGS_ENSEMBLE, None)
        model_init = model_type(path=path, name=name, problem_type=problem_type, eval_metric=eval_metric, stopping_metric=stopping_metric, num_classes=num_classes, hyperparameters=model_params, feature_metadata=feature_metadata)

        if ensemble_kwargs is not None:
            ensemble_kwargs_model = copy.deepcopy(ensemble_kwargs)
            extra_ensemble_hyperparameters = copy.deepcopy(model.get(AG_ARGS_ENSEMBLE, dict()))
            ensemble_kwargs_model['hyperparameters'] = ensemble_kwargs_model.get('hyperparameters', {})
            if ensemble_kwargs_model['hyperparameters'] is None:
                ensemble_kwargs_model['hyperparameters'] = {}
            ensemble_kwargs_model['hyperparameters'].update(extra_ensemble_hyperparameters)
            model_init = ensemble_type(path=path, name=name_stacker, model_base=model_init, num_classes=num_classes, **ensemble_kwargs_model)

        models.append(model_init)

    return models


# TODO: v0.1 cleanup and avoid hardcoded logic with model names
def get_preset_models_softclass(path, hyperparameters, feature_metadata, num_classes=None, hyperparameter_tune_kwargs=None, name_suffix='', ag_args=None, invalid_model_names: list = None):
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
    models = get_preset_models(path=path, problem_type=SOFTCLASS, feature_metadata=feature_metadata, eval_metric=soft_log_loss, stopping_metric=soft_log_loss,
                               hyperparameters=hyperparameters_standard, num_classes=num_classes, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
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
        rf_models = get_preset_models(path=path, problem_type=REGRESSION, feature_metadata=feature_metadata, eval_metric=mean_squared_error,
                                      hyperparameters=hyperparameters_rf, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                                      ag_args=ag_args, name_suffix=name_suffix, default_priorities=DEFAULT_SOFTCLASS_PRIORITY, invalid_model_names=invalid_model_names)
    models_cat = [model for model in models if isinstance(model, CatBoostModel)]
    models_noncat = [model for model in models if not isinstance(model, CatBoostModel)]
    models = models_noncat + rf_models + models_cat
    if len(models) == 0:
        raise ValueError("At least one of the following model-types must be present in hyperparameters: ['GBM','CAT','NN','RF'], "
                         "These are the only supported models for softclass prediction problems. "
                         "Softclass problems are also not yet supported for fit() with per-stack level hyperparameters.")
    for model in models:
        model.normalize_pred_probas = True

    return models
