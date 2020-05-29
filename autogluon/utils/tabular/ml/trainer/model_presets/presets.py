import copy
import logging
from collections import defaultdict

from sklearn.linear_model import LogisticRegression, LinearRegression

from ...constants import AG_ARGS, AG_ARGS_FIT, BINARY, MULTICLASS, REGRESSION, SOFTCLASS, PROBLEM_TYPES_CLASSIFICATION
from ...models.abstract.abstract_model import AbstractModel
from ...models.lgb.lgb_model import LGBModel
from ...models.lr.lr_model import LinearModel
from ...models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from ...models.rf.rf_model import RFModel
from ...models.knn.knn_model import KNNModel
from ...models.catboost.catboost_model import CatboostModel
from ...models.xt.xt_model import XTModel
from ....metrics import soft_log_loss

logger = logging.getLogger(__name__)

# Higher values indicate higher priority, priority dictates the order models are trained for a given level.
DEFAULT_MODEL_PRIORITY = dict(
    RF=100,
    XT=90,
    KNN=80,
    GBM=70,
    CAT=60,
    NN=50,
    LR=40,
    custom=0,
)

MODEL_TYPES = dict(
    RF=RFModel,
    XT=XTModel,
    KNN=KNNModel,
    GBM=LGBModel,
    CAT=CatboostModel,
    NN=TabularNeuralNetModel,
    LR=LinearModel,
)

DEFAULT_MODEL_NAMES = {
    RFModel: 'RandomForest',
    XTModel: 'ExtraTrees',
    KNNModel: 'KNeighbors',
    LGBModel: 'LightGBM',
    CatboostModel: 'Catboost',
    TabularNeuralNetModel: 'NeuralNet',
    LinearModel: 'LinearModel',
}


def _dd_classifier():
    return 'Classifier'


def _dd_regressor():
    return 'Regressor'


DEFAULT_MODEL_TYPE_SUFFIX = dict(
    classifier=defaultdict(_dd_classifier),
    regressor=defaultdict(_dd_regressor),
)
DEFAULT_MODEL_TYPE_SUFFIX['classifier'].update({LinearModel: ''})
DEFAULT_MODEL_TYPE_SUFFIX['regressor'].update({LinearModel: ''})


# DONE: Add levels, including 'default'
# DONE: Add lists
# DONE: Add custom which can append to lists
# DONE: Add special optional AG args for things like name prefix, name suffix, name, etc.
# TODO: Move creation of stack ensemble internally into this function? Requires passing base models in as well.
# DONE: Add special optional AG args for training order
# TODO: Add special optional AG args for base models
# TODO: Consider making hyperparameters arg in fit() accept lists, concatenate hyperparameter sets together.
# TODO: Consider adding special optional AG args for #cores,#gpus,num_early_stopping_iterations,etc.
# TODO: Consider adding special optional AG args for max train time, max memory size, etc.
# TODO: Consider adding special optional AG args for use_original_features,features_to_use,etc.
# TODO: Consider adding optional AG args to dynamically disable models such as valid_num_classes_range, valid_row_count_range, valid_feature_count_range, etc.
# TODO: Args such as max_repeats, num_folds
# TODO: Add banned_model_types arg
# TODO: Add option to update hyperparameters with only added keys, so disabling CatBoost would just be {'CAT': []}, which keeps the other models as is.
# TODO: special optional AG arg for only training model if eval_metric in list / not in list. Useful for F1 and 'is_unbalanced' arg in LGBM.
def get_preset_models(path, problem_type, objective_func, hyperparameters, stopping_metric=None, num_classes=None, hyperparameter_tune=False, level='default', extra_ag_args_fit=None, name_suffix=''):
    if problem_type not in [BINARY, MULTICLASS, REGRESSION]:
        raise NotImplementedError

    if level in hyperparameters.keys():
        level_key = level
    else:
        level_key = 'default'
    hp_level = hyperparameters[level_key]
    priority_dict = defaultdict(list)
    for model_type in hp_level:
        for model in hp_level[model_type]:
            model = copy.deepcopy(model)
            try:
                model_priority = model[AG_ARGS]['priority']
            except:
                model_priority = DEFAULT_MODEL_PRIORITY[model_type]
            if AG_ARGS not in model:
                model[AG_ARGS] = dict()
            if 'model_type' not in model[AG_ARGS]:
                model[AG_ARGS]['model_type'] = model_type
            # Check if model is valid
            if hyperparameter_tune and model[AG_ARGS].get('disable_in_hpo', False):
                continue  # Not valid
            priority_dict[model_priority].append(model)
    model_priority_list = [model for priority in sorted(priority_dict.keys(), reverse=True) for model in priority_dict[priority]]
    model_names_set = set()
    models = []
    for model in model_priority_list:
        model_type = model[AG_ARGS]['model_type']
        if not isinstance(model_type, AbstractModel):
            model_type = MODEL_TYPES[model_type]
        name_orig = model[AG_ARGS].get('name', None)
        if name_orig is None:
            name_main = model[AG_ARGS].get('name_main', DEFAULT_MODEL_NAMES[model_type])
            name_prefix = model[AG_ARGS].get('name_prefix', '')
            name_type_suffix = model[AG_ARGS].get('name_type_suffix', None)
            if name_type_suffix is None:
                suffix_key = 'classifier' if problem_type in PROBLEM_TYPES_CLASSIFICATION else 'regressor'
                name_type_suffix = DEFAULT_MODEL_TYPE_SUFFIX[suffix_key][model_type]
            name_suff = model[AG_ARGS].get('name_suffix', '')
            name_orig = name_prefix + name_main + name_type_suffix + name_suff
        name = name_orig
        num_increment = 2
        while name in model_names_set:  # Ensure name is unique
            name = f'{name_orig}_{num_increment}'
            num_increment += 1
        model_names_set.add(name)
        model_params = copy.deepcopy(model)
        model_params.pop(AG_ARGS)
        if extra_ag_args_fit is not None:
            if AG_ARGS_FIT not in model_params:
                model_params[AG_ARGS_FIT] = {}
            model_params[AG_ARGS_FIT].update(extra_ag_args_fit.copy())  # TODO: Consider case of overwriting user specified extra args.
        model_init = model_type(path=path, name=name, problem_type=problem_type, objective_func=objective_func, stopping_metric=stopping_metric, num_classes=num_classes, hyperparameters=model_params)
        models.append(model_init)

    for model in models:
        model.rename(model.name + name_suffix)

    return models


def get_preset_stacker_model(path, problem_type, objective_func, num_classes=None,
                             hyperparameters={'NN': {}, 'GBM': {}}, hyperparameter_tune=False):
    # TODO: Expand options to RF and NN
    if problem_type == REGRESSION:
        model = RFModel(path=path, name='LinearRegression', model=LinearRegression(),
                        problem_type=problem_type, objective_func=objective_func)
    else:
        model = RFModel(path=path, name='LogisticRegression', model=LogisticRegression(
            solver='liblinear', multi_class='auto', max_iter=500,  # n_jobs=-1  # TODO: HP set to hide warnings, but we should find optimal HP for this
        ), problem_type=problem_type, objective_func=objective_func)
    return model


def get_preset_models_softclass(path, hyperparameters={}, hyperparameter_tune=False, name_suffix=''):
    # print("Neural Net is currently the only model supported for multi-class distillation.")
    models = []
    # TODO: only NN supported for now. add other models. We use a big NN for distillation to ensure it has high capacity to approximate ensemble:
    nn_options = {'num_epochs': 500, 'dropout_prob': 0, 'weight_decay': 1e-7, 'epochs_wo_improve': 50, 'layers': [2048]*2 + [512], 'numeric_embed_dim': 2048, 'activation': 'softrelu', 'embedding_size_factor': 2.0}
    models.append(
        TabularNeuralNetModel(path=path, name='NeuralNetSoftClassifier', problem_type=SOFTCLASS,
                              objective_func=soft_log_loss, stopping_metric=soft_log_loss, hyperparameters=nn_options.copy())
    )
    rf_options = dict(criterion='mse')
    models.append(
        RFModel(path=path, name='RandomForestRegressorMSE', problem_type=REGRESSION,
                objective_func=soft_log_loss, hyperparameters=rf_options),
    )

    for model in models:
        model.rename(model.name + name_suffix)

    return models
