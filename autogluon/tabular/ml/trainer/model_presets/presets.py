# TODO: move these files
from autogluon.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION, LANGUAGE_MODEL
from autogluon.tabular.ml.models.lgb.lgb_model import LGBModel

from autogluon.tabular.ml.models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from autogluon.tabular.contrib.nn_nlp_lm_model import NNNLPLanguageModel
from autogluon.tabular.contrib.tabular_nn_pytorch.hyperparameters.parameters import get_nlp_param_baseline # TODO: remove


def get_preset_models(path, problem_type, objective_func, num_classes=None,
                      hyperparameters={'NN':{},'GBM':{}}):
    if problem_type in [BINARY, MULTICLASS]:
        return get_preset_models_classification(path=path, problem_type=problem_type,
                    objective_func=objective_func, num_classes=num_classes,
                    hyperparameters=hyperparameters)
    elif problem_type == REGRESSION:
        return get_preset_models_regression(path=path, problem_type=problem_type,
                    objective_func=objective_func, hyperparameters=hyperparameters)
    elif problem_type == LANGUAGE_MODEL:
        return get_preset_models_language(path=path, hyperparameters=hyperparameters)
    else:
        raise NotImplementedError


def get_preset_models_classification(path, problem_type, objective_func, num_classes=None, 
                                     hyperparameters={'NN':{},'GBM':{}}):
    # TODO: define models based on additional keys in hyperparameters
    models = [
        # SKLearnModel(path=path, name='DummyClassifier', model=DummyClassifier(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='GaussianNB', model=GaussianNB(), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='DecisionTreeClassifier', model=DecisionTreeClassifier(), problem_type=problem_type, objective_func=objective_func),
        # RFModel(path=path, name='RandomForestClassifier', model=RandomForestClassifier(n_jobs=-1), problem_type=problem_type, objective_func=objective_func),
        # RFModel(path=path, name='RandomForestClassifierLarge', model=RandomForestClassifier(n_estimators=300, n_jobs=-1), problem_type=problem_type, objective_func=objective_func),
        # RFModel(path=path, name='RandomForestClassifierLargest', model=RandomForestClassifier(n_estimators=3000, n_jobs=-1), problem_type=problem_type, objective_func=objective_func),
        # RFModel(path=path, name='ExtraTreesClassifier', model=ExtraTreesClassifier(n_jobs=-1), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='LogisticRegression', model=LogisticRegression(n_jobs=-1), problem_type=problem_type, objective_func=objective_func),
        # RFModel(path=path, name='LGBMClassifier', model=lgb.LGBMClassifier(n_jobs=-1), problem_type=problem_type, objective_func=objective_func),
        # NNTabularModel(path=path, name='NNTabularModel', params=nn_get_param_baseline(problem_type), problem_type=problem_type, objective_func=objective_func) # OG fast.ai model. TODO: remove!
        # NNNLPClassificationModel(path=path, name='NNNLPClassificationModel-FWD', params=get_nlp_param_baseline(), problem_type=problem_type, objective_func=objective_func),
        # NNNLPClassificationModel(path=path, name='NNNLPClassificationModel-BWD', params=get_nlp_param_baseline(), problem_type=problem_type, objective_func=objective_func, train_backwards=True),
    ]
    gbm_options = hyperparameters.get('GBM', None)
    nn_options = hyperparameters.get('NN', None)
    if gbm_options is not None:
        models.append(
            LGBModel(path=path, name='GradientBoostClassifier', problem_type=problem_type, 
                objective_func=objective_func, num_classes=num_classes, hyperparameters=gbm_options.copy())
        )
    if nn_options is not None:
        models.append(
            TabularNeuralNetModel(path=path, name='NeuralNetClassifier', problem_type=problem_type, 
                                  objective_func=objective_func, hyperparameters=nn_options.copy()),
        )
    return models


def get_preset_models_language(path, hyperparameters={'NN':{},'GBM':{}}):
    models = [
        NNNLPLanguageModel(path=path, name='LanguageModel', params=get_nlp_param_baseline()),
    ]
    return models


def get_preset_models_regression(path, problem_type, objective_func, hyperparameters={'NN':{},'GBM':{}}):
    models = [
        # SKLearnModel(path=path, name='DummyRegressor', model=DummyRegressor(), problem_type=problem_type, objective_func=objective_func),
        # RFModel(path=path, name='RandomForestRegressor', model=RandomForestRegressor(n_jobs=-1), problem_type=problem_type, objective_func=objective_func),
        # SKLearnModel(path=path, name='LGBMRegressor', model=lgb.LGBMRegressor(n_jobs=-1, verbose=2, silent=False), problem_type=problem_type, objective_func=objective_func),
        # NNTabularModel(path=path, name='NNTabularModel', params=nn_get_param_baseline(problem_type), problem_type=problem_type, objective_func=objective_func),
    ]
    gbm_options = hyperparameters.get('GBM', None)
    nn_options = hyperparameters.get('NN', None)
    if gbm_options is not None: 
        models.append(
            LGBModel(path=path, name='GradientBoostRegressor', problem_type=problem_type,
                     objective_func=objective_func, hyperparameters=gbm_options.copy())
        )
    if nn_options is not None:
        models.append(
            TabularNeuralNetModel(path=path, name='NeuralNetRegressor', problem_type=problem_type,
                                  objective_func=objective_func, hyperparameters=nn_options.copy())
        )
    return models
