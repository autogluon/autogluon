from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor

from ...constants import BINARY, MULTICLASS, REGRESSION
from ...models.lgb.lgb_model import LGBModel
from ...models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from ...models.rf.rf_model import RFModel
# from ...models.catboost.catboost_model import CatboostModel


def get_preset_models(path, problem_type, objective_func, num_classes=None,
                      hyperparameters={'NN':{},'GBM':{}}):
    if problem_type in [BINARY, MULTICLASS]:
        return get_preset_models_classification(path=path, problem_type=problem_type,
                    objective_func=objective_func, num_classes=num_classes,
                    hyperparameters=hyperparameters)
    elif problem_type == REGRESSION:
        return get_preset_models_regression(path=path, problem_type=problem_type,
                    objective_func=objective_func, hyperparameters=hyperparameters)
    else:
        raise NotImplementedError


def get_preset_models_classification(path, problem_type, objective_func, num_classes=None, 
                                     hyperparameters={'NN':{},'GBM':{}}):
    # TODO: define models based on additional keys in hyperparameters

    models = []
    gbm_options = hyperparameters.get('GBM', None)
    nn_options = hyperparameters.get('NN', None)
    cat_options = hyperparameters.get('CAT', None)
    rf_options = hyperparameters.get('RF', None)
    xt_options = hyperparameters.get('XT', None)
    if gbm_options is not None:
        models.append(
            LGBModel(path=path, name='LightGBMClassifier', problem_type=problem_type,
                     objective_func=objective_func, num_classes=num_classes, hyperparameters=gbm_options.copy())
        )
    if nn_options is not None:
        models.append(
            TabularNeuralNetModel(path=path, name='NeuralNetClassifier', problem_type=problem_type,
                                  objective_func=objective_func, hyperparameters=nn_options.copy()),
        )
    # if cat_options is not None:
    #     models.append(
    #         CatboostModel(path=path, name='CatboostClassifier', problem_type=problem_type,
    #                       objective_func=objective_func, hyperparameters=cat_options.copy()),
    #     )
    if rf_options is not None:
        params = {'n_estimators': 300, 'n_jobs': -1}
        params.update(rf_options.copy())  # TODO: Move into RFModel, currently ignores hyperparameters
        models.append(
            RFModel(path=path, name='RandomForestClassifier', model=RandomForestClassifier(**params), problem_type=problem_type,
                    objective_func=objective_func, hyperparameters=rf_options.copy()),
        )
    if xt_options is not None:
        params = {'n_estimators': 100, 'n_jobs': -1}
        params.update(xt_options.copy())  # TODO: Move into RFModel, currently ignores hyperparameters
        models.append(
            RFModel(path=path, name='ExtraTreesClassifier', model=ExtraTreesClassifier(**params), problem_type=problem_type,
                    objective_func=objective_func, hyperparameters=xt_options.copy()),
        )

    models += [
        # LGBModel(path=path, name='LGBMClassifierCustom', problem_type=problem_type, objective_func=objective_func, num_classes=num_classes, hyperparameters=lgb_get_param_baseline(problem_type, num_classes=num_classes)),
        # NNTabularModel(path=path, name='GrailNNTabularModel', params=nn_get_param_baseline(problem_type), problem_type=problem_type, objective_func=objective_func),  # OG fast.ai model. TODO: remove!

        # SKLearnModel(path=path, name='DummyClassifier', model=DummyClassifier(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='GaussianNB', model=GaussianNB(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='DecisionTreeClassifier', model=DecisionTreeClassifier(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='LogisticRegression', model=LogisticRegression(n_jobs=-1), problem_type=problem_type, objective_func=objective_func),
        # RFModel(path=path, name='LGBMClassifier', model=lgb.LGBMClassifier(n_jobs=-1), problem_type=problem_type, objective_func=objective_func),
        # LGBSKLearnModel(path=path, name='LGBMClassifierV2', model=lgb.LGBMClassifier(n_jobs=-1), problem_type=problem_type, objective_func=objective_func),
        # LGBModel(path=path, name='LGBMClassifierCustomDummyGPU', params=get_param_binary_baseline_dummy_gpu(), num_boost_round=num_boost_round, problem_type=problem_type, objective_func=objective_func),
    ]

    return models


def get_preset_models_regression(path, problem_type, objective_func, hyperparameters={'NN':{},'GBM':{}}):
    models = []
    gbm_options = hyperparameters.get('GBM', None)
    nn_options = hyperparameters.get('NN', None)
    cat_options = hyperparameters.get('CAT', None)
    rf_options = hyperparameters.get('RF', None)
    xt_options = hyperparameters.get('XT', None)
    if gbm_options is not None:
        models.append(
            LGBModel(path=path, name='LightGBMRegressor', problem_type=problem_type,
                     objective_func=objective_func, hyperparameters=gbm_options.copy())
        )
    if nn_options is not None:
        models.append(
            TabularNeuralNetModel(path=path, name='NeuralNetRegressor', problem_type=problem_type,
                                  objective_func=objective_func, hyperparameters=nn_options.copy())
        )
    # if cat_options is not None:
    #     models.append(
    #         CatboostModel(path=path, name='CatboostRegressor', problem_type=problem_type,
    #                       objective_func=objective_func, hyperparameters=cat_options.copy()),
    #     )
    if rf_options is not None:
        params = {'n_estimators': 300, 'n_jobs': -1}
        params.update(rf_options.copy())  # TODO: Move into RFModel, currently ignores hyperparameters
        models.append(
            RFModel(path=path, name='RandomForestRegressor', model=RandomForestRegressor(**params), problem_type=problem_type,
                    objective_func=objective_func, hyperparameters=rf_options.copy()),
        )
    if xt_options is not None:
        params = {'n_estimators': 100, 'n_jobs': -1}
        params.update(xt_options.copy())  # TODO: Move into RFModel, currently ignores hyperparameters
        models.append(
            RFModel(path=path, name='ExtraTreesRegressor', model=ExtraTreesRegressor(**params), problem_type=problem_type,
                    objective_func=objective_func, hyperparameters=xt_options.copy()),
        )
    models += [
        # Good GBDT
        # LGBModel(path=path, name='LGBMRegressorCustom', problem_type=problem_type, objective_func=objective_func, hyperparameters=lgb_get_param_baseline(problem_type)),
        # NNTabularModel(path=path, name='GrailNNTabularModel', params=nn_get_param_baseline(problem_type), problem_type=problem_type, objective_func=objective_func),

        # SKLearnModel(path=path, name='DummyRegressor', model=DummyRegressor(), problem_type=problem_type, objective_func=objective_func),
    ]

    return models
