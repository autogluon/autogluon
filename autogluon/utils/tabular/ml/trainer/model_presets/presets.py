import logging
import mxnet as mx
from sklearn.linear_model import LogisticRegression, LinearRegression

from ...constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS
from ...models.lgb.lgb_model import LGBModel
from ...models.lgb.hyperparameters.parameters import get_param_baseline_custom
from ...models.lr.lr_model import LinearModel
from ...models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from ...models.rf.rf_model import RFModel
from ...models.knn.knn_model import KNNModel
from ...models.catboost.catboost_model import CatboostModel
from .presets_rf import rf_classifiers, xt_classifiers, rf_regressors, xt_regressors
from ....metrics import soft_log_loss, mean_squared_error

logger = logging.getLogger(__name__)


def get_preset_models(path, problem_type, objective_func, stopping_metric=None, num_classes=None,
                      hyperparameters={'NN': {}, 'GBM': {}}, hyperparameter_tune=False, name_suffix=''):
    if problem_type in [BINARY, MULTICLASS]:
        return get_preset_models_classification(path=path, problem_type=problem_type,
                                                objective_func=objective_func, stopping_metric=stopping_metric, num_classes=num_classes,
                                                hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune, name_suffix=name_suffix)
    elif problem_type == REGRESSION:
        return get_preset_models_regression(path=path, problem_type=problem_type,
                                            objective_func=objective_func, stopping_metric=stopping_metric, hyperparameters=hyperparameters,
                                            hyperparameter_tune=hyperparameter_tune, name_suffix=name_suffix)
    else:
        raise NotImplementedError


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


def get_preset_models_classification(path, problem_type, objective_func, stopping_metric=None, num_classes=None,
                                     hyperparameters={'NN': {}, 'GBM': {}, 'custom': {}}, hyperparameter_tune=False, name_suffix=''):
    # TODO: define models based on additional keys in hyperparameters

    models = []
    lr_options = hyperparameters.get('LR', None)
    gbm_options = hyperparameters.get('GBM', None)
    nn_options = hyperparameters.get('NN', None)
    cat_options = hyperparameters.get('CAT', None)
    rf_options = hyperparameters.get('RF', None)
    xt_options = hyperparameters.get('XT', None)
    knn_options = hyperparameters.get('KNN', None)
    custom_options = hyperparameters.get('custom', None)
    if rf_options is not None:
        models += rf_classifiers(hyperparameters=rf_options, path=path, problem_type=problem_type, objective_func=objective_func, num_classes=num_classes)
    if xt_options is not None:
        models += xt_classifiers(hyperparameters=xt_options, path=path, problem_type=problem_type, objective_func=objective_func, num_classes=num_classes)
    if knn_options is not None:
        # TODO: Combine uniform and distance into one model when doing HPO
        knn_unif_params = knn_options.copy()
        knn_unif_params['weights'] = 'uniform'
        models.append(
            KNNModel(path=path, name='KNeighborsClassifierUnif', problem_type=problem_type,
                     objective_func=objective_func, hyperparameters=knn_unif_params),
        )
        knn_dist_params = knn_options.copy()
        knn_dist_params['weights'] = 'distance'
        models.append(
            KNNModel(path=path, name='KNeighborsClassifierDist', problem_type=problem_type,
                     objective_func=objective_func, hyperparameters=knn_dist_params),
        )
    if gbm_options is not None:
        models.append(
            LGBModel(path=path, name='LightGBMClassifier', problem_type=problem_type,
                     objective_func=objective_func, stopping_metric=stopping_metric, num_classes=num_classes, hyperparameters=gbm_options.copy())
        )
    if cat_options is not None:
        models.append(
            CatboostModel(path=path, name='CatboostClassifier', problem_type=problem_type,
                          objective_func=objective_func, stopping_metric=stopping_metric, num_classes=num_classes, hyperparameters=cat_options.copy()),
        )
    if nn_options is not None:
        models.append(
            TabularNeuralNetModel(path=path, name='NeuralNetClassifier', problem_type=problem_type,
                                  objective_func=objective_func, stopping_metric=stopping_metric, hyperparameters=nn_options.copy()),
        )
    if lr_options is not None:
        _add_models(
            models, lr_options, 'LinearModel',
            lambda name_prefix, lr_option: LinearModel(path=path, name=name_prefix, problem_type=problem_type, objective_func=objective_func, hyperparameters=lr_option.copy())
        )
    if (not hyperparameter_tune) and (custom_options is not None):
        # Consider additional models with custom pre-specified hyperparameter settings:
        if 'GBM' in custom_options:
            models += [LGBModel(path=path, name='LightGBMClassifierCustom', problem_type=problem_type, objective_func=objective_func, stopping_metric=stopping_metric,
                         num_classes=num_classes, hyperparameters=get_param_baseline_custom(problem_type, num_classes=num_classes))
                ]
        # SKLearnModel(path=path, name='DummyClassifier', model=DummyClassifier(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='GaussianNB', model=GaussianNB(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='DecisionTreeClassifier', model=DecisionTreeClassifier(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='LogisticRegression', model=LogisticRegression(n_jobs=-1), problem_type=problem_type, objective_func=objective_func)

    for model in models:
        model.rename(model.name + name_suffix)

    # TODO: Update name_suffix to only apply here so its not repeated code! Add .rename function to model
    return models


def get_preset_models_regression(path, problem_type, objective_func, stopping_metric=None, hyperparameters={'NN':{},'GBM':{},'custom':{}}, hyperparameter_tune=False, name_suffix=''):
    models = []
    lr_options = hyperparameters.get('LR', None)
    gbm_options = hyperparameters.get('GBM', None)
    nn_options = hyperparameters.get('NN', None)
    cat_options = hyperparameters.get('CAT', None)
    rf_options = hyperparameters.get('RF', None)
    xt_options = hyperparameters.get('XT', None)
    knn_options = hyperparameters.get('KNN', None)
    custom_options = hyperparameters.get('custom', None)
    if rf_options is not None:
        models += rf_regressors(hyperparameters=rf_options, path=path, problem_type=problem_type, objective_func=objective_func)
    if xt_options is not None:
        models += xt_regressors(hyperparameters=xt_options, path=path, problem_type=problem_type, objective_func=objective_func)
    if knn_options is not None:
        # TODO: Combine uniform and distance into one model when doing HPO
        knn_unif_params = knn_options.copy()
        knn_unif_params['weights'] = 'uniform'
        models.append(
            KNNModel(path=path, name='KNeighborsRegressorUnif', problem_type=problem_type,
                     objective_func=objective_func, hyperparameters=knn_unif_params),
        )
        knn_dist_params = knn_options.copy()
        knn_dist_params['weights'] = 'distance'
        models.append(
            KNNModel(path=path, name='KNeighborsRegressorDist', problem_type=problem_type,
                     objective_func=objective_func, hyperparameters=knn_dist_params),
        )
    if gbm_options is not None:
        models.append(
            LGBModel(path=path, name='LightGBMRegressor', problem_type=problem_type,
                     objective_func=objective_func, stopping_metric=stopping_metric, hyperparameters=gbm_options.copy())
        )
    if cat_options is not None:
        models.append(
            CatboostModel(path=path, name='CatboostRegressor', problem_type=problem_type,
                          objective_func=objective_func, stopping_metric=stopping_metric, hyperparameters=cat_options.copy()),
        )
    if nn_options is not None:
        models.append(
            TabularNeuralNetModel(path=path, name='NeuralNetRegressor', problem_type=problem_type,
                                  objective_func=objective_func, stopping_metric=stopping_metric, hyperparameters=nn_options.copy())
        )
    if (not hyperparameter_tune) and (custom_options is not None):
        if 'GBM' in custom_options:
            models += [LGBModel(path=path, name='LightGBMRegressorCustom', problem_type=problem_type, objective_func=objective_func, stopping_metric=stopping_metric, hyperparameters=get_param_baseline_custom(problem_type))]
        # SKLearnModel(path=path, name='DummyRegressor', model=DummyRegressor(), problem_type=problem_type, objective_func=objective_func),
    if lr_options is not None:
        _add_models(
            models, lr_options, 'LinearModel',
            lambda name_prefix, lr_option: LinearModel(path=path, name=name_prefix, problem_type=problem_type, objective_func=objective_func, hyperparameters=lr_option.copy())
        )

    for model in models:
        model.rename(model.name + name_suffix)

    return models


def get_preset_models_softclass(path, hyperparameters={}, hyperparameter_tune=False, name_suffix=''):
    # print("Neural Net is currently the only model supported for multi-class distillation.")
    models = []
    # TODO: only NN supported for now. add other models. We use a big NN for distillation to ensure it has high capacity to approximate ensemble:
    nn_options = {'num_epochs': 500, 'dropout_prob': 0, 'weight_decay': 1e-7, 'epochs_wo_improve': 50, 'layers': [2048]*2 + [512], 'numeric_embed_dim': 2048, 'activation': 'softrelu', 'embedding_size_factor': 2.0}
    models.append(
        TabularNeuralNetModel(path=path, name='NeuralNetSoftClassifier', problem_type=SOFTCLASS,
                              objective_func=soft_log_loss, stopping_metric=soft_log_loss, hyperparameters=nn_options.copy())
    )
    rf_options = {}
    models += rf_regressors(hyperparameters=rf_options, path=path, problem_type=REGRESSION, objective_func=soft_log_loss)
    for model in models:
        model.rename(model.name + name_suffix)

    return models


def _add_models(models, options, name_prefix, model_fn):
    if isinstance(options, list):
        for i, option in enumerate(options):
            name = f'{name_prefix}_{i}' if len(options) > 1 else name_prefix
            models.append(model_fn(name, option))
    else:
        models.append(model_fn(name_prefix, options))

