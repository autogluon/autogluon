import logging
from sklearn.linear_model import LogisticRegression, LinearRegression

from ...constants import BINARY, MULTICLASS, REGRESSION
from ...models.lgb.lgb_model import LGBModel
from ...models.lgb.hyperparameters.parameters import get_param_baseline_custom
from ...models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from ...models.rf.rf_model import RFModel
from ...models.knn.knn_model import KNNModel
from ...models.catboost.catboost_model import CatboostModel
from .presets_rf import rf_classifiers, xt_classifiers, rf_regressors, xt_regressors

logger = logging.getLogger(__name__)


def get_preset_models(path, problem_type, objective_func, num_classes=None,
                      hyperparameters={'NN':{},'GBM':{}}, hyperparameter_tune=False):
    if problem_type in [BINARY, MULTICLASS]:
        return get_preset_models_classification(path=path, problem_type=problem_type,
                    objective_func=objective_func, num_classes=num_classes,
                    hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune)
    elif problem_type == REGRESSION:
        return get_preset_models_regression(path=path, problem_type=problem_type,
                    objective_func=objective_func, hyperparameters=hyperparameters, hyperparameter_tune=hyperparameter_tune)
    else:
        raise NotImplementedError


def get_preset_stacker_model(path, problem_type, objective_func, num_classes=None,
                      hyperparameters={'NN':{},'GBM':{}}, hyperparameter_tune=False):
    # TODO: Expand options to RF and NN
    if problem_type == REGRESSION:
        model = RFModel(path=path, name='LinearRegression', model=LinearRegression(), 
                        problem_type=problem_type, objective_func=objective_func)
    else:
        model = RFModel(path=path, name='LogisticRegression', model=LogisticRegression(
            solver='liblinear', multi_class='auto', max_iter=500,  # n_jobs=-1  # TODO: HP set to hide warnings, but we should find optimal HP for this
        ), problem_type=problem_type, objective_func=objective_func)
    return model


def get_preset_models_classification(path, problem_type, objective_func, num_classes=None,
                                     hyperparameters={'NN':{},'GBM':{},'custom':{}}, hyperparameter_tune=False):
    # TODO: define models based on additional keys in hyperparameters

    models = []
    gbm_options = hyperparameters.get('GBM', None)
    nn_options = hyperparameters.get('NN', None)
    cat_options = hyperparameters.get('CAT', None)
    rf_options = hyperparameters.get('RF', None)
    xt_options = hyperparameters.get('XT', None)
    knn_options = hyperparameters.get('KNN', None)
    custom_options = hyperparameters.get('custom', None)
    if rf_options is not None:
        models += rf_classifiers(hyperparameters=rf_options, path=path, problem_type=problem_type, objective_func=objective_func)
    if xt_options is not None:
        models += xt_classifiers(hyperparameters=xt_options, path=path, problem_type=problem_type, objective_func=objective_func)
    if knn_options is not None:
        # TODO: Combine uniform and distance into one model when doing HPO
        knn_unif_params = {'weights': 'uniform', 'n_jobs': -1}
        knn_unif_params.update(knn_options.copy())
        models.append(
            KNNModel(path=path, name='KNeighborsClassifierUnif', problem_type=problem_type,
                     objective_func=objective_func, hyperparameters=knn_unif_params),
        )
        knn_dist_params = {'weights': 'distance', 'n_jobs': -1}
        knn_dist_params.update(knn_options.copy())
        models.append(
            KNNModel(path=path, name='KNeighborsClassifierDist', problem_type=problem_type,
                     objective_func=objective_func, hyperparameters=knn_dist_params),
        )
    if gbm_options is not None:
        models.append(
            LGBModel(path=path, name='LightGBMClassifier', problem_type=problem_type,
                     objective_func=objective_func, num_classes=num_classes, hyperparameters=gbm_options.copy())
        )
    if cat_options is not None:
        models.append(
            CatboostModel(path=path, name='CatboostClassifier', problem_type=problem_type,
                          objective_func=objective_func, hyperparameters=cat_options.copy()),
        )
    if nn_options is not None:
        models.append(
            TabularNeuralNetModel(path=path, name='NeuralNetClassifier', problem_type=problem_type,
                                  objective_func=objective_func, hyperparameters=nn_options.copy()),
        )
    if (not hyperparameter_tune) and (custom_options is not None):
        # Consider additional models with custom pre-specified hyperparameter settings:
        if 'GBM' in custom_options:
            models += [LGBModel(path=path, name='LightGBMClassifierCustom', problem_type=problem_type, objective_func=objective_func, 
                                num_classes=num_classes, hyperparameters=get_param_baseline_custom(problem_type, num_classes=num_classes))
                      ]
        # SKLearnModel(path=path, name='DummyClassifier', model=DummyClassifier(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='GaussianNB', model=GaussianNB(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='DecisionTreeClassifier', model=DecisionTreeClassifier(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='LogisticRegression', model=LogisticRegression(n_jobs=-1), problem_type=problem_type, objective_func=objective_func)

    return models


def get_preset_models_regression(path, problem_type, objective_func, hyperparameters={'NN':{},'GBM':{},'custom':{}}, hyperparameter_tune=False):
    models = []
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
        knn_unif_params = {'weights': 'uniform', 'n_jobs': -1}
        knn_unif_params.update(knn_options.copy())
        models.append(
            KNNModel(path=path, name='KNeighborsRegressorUnif', problem_type=problem_type,
                    objective_func=objective_func, hyperparameters=knn_unif_params),
        )
        knn_dist_params = {'weights': 'distance', 'n_jobs': -1}
        knn_dist_params.update(knn_options.copy())
        models.append(
            KNNModel(path=path, name='KNeighborsRegressorDist', problem_type=problem_type,
                     objective_func=objective_func, hyperparameters=knn_dist_params),
        )
    if gbm_options is not None:
        models.append(
            LGBModel(path=path, name='LightGBMRegressor', problem_type=problem_type,
                     objective_func=objective_func, hyperparameters=gbm_options.copy())
        )
    if cat_options is not None:
        models.append(
            CatboostModel(path=path, name='CatboostRegressor', problem_type=problem_type,
                          objective_func=objective_func, hyperparameters=cat_options.copy()),
        )
    if nn_options is not None:
        models.append(
            TabularNeuralNetModel(path=path, name='NeuralNetRegressor', problem_type=problem_type,
                                  objective_func=objective_func, hyperparameters=nn_options.copy())
        )
    if (not hyperparameter_tune) and (custom_options is not None):
        if 'GBM' in custom_options:
            models += [LGBModel(path=path, name='LightGBMRegressorCustom', problem_type=problem_type, objective_func=objective_func, hyperparameters=get_param_baseline_custom(problem_type))]
        # SKLearnModel(path=path, name='DummyRegressor', model=DummyRegressor(), problem_type=problem_type, objective_func=objective_func),

    return models
