import os
import math
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import METRICS, get_metric, make_scorer
from autogluon.core.constants import BINARY, REGRESSION, MULTICLASS
from autogluon.tabular.trainer.model_presets.presets import MODEL_TYPES, DEFAULT_MODEL_NAMES
from autogluon.tabular.models import LGBModel, XGBoostModel, TabularNeuralNetTorchModel

MODELS = [name for name, model in MODEL_TYPES.items() if model._get_class_tags().get("supports_learning_curves", False)]
PROBLEM_TYPES = [BINARY, MULTICLASS, REGRESSION]

common_args = {"sample_size": 50, "delete_directory": False, "refit_full": False}

early_stop = 999999
long_run = 500
short_run = 50

model_iterations = {
    "GBM": long_run,
    "XGB": long_run,
    "NN_TORCH": short_run,
}

extended_run_hyperparams = {
    "GBM": {
        "ag.early_stop": early_stop,
        "num_boost_round": long_run,
    },
    "XGB": {
        "ag.early_stop": early_stop,
        "n_estimators": long_run,
    },
    "NN_TORCH": {
        "epochs_wo_improve": early_stop,
        "num_epochs": short_run,
    },
}


def get_one_model_problem():
    problem_type = PROBLEM_TYPES[0]
    model = MODELS[0]
    return [(problem_type, model)]


def get_all_models():
    problem_type = PROBLEM_TYPES[0]
    models = MODELS
    return [(problem_type, model) for model in models]


def get_all_problems():
    problem_types = PROBLEM_TYPES
    model = MODELS[0]
    return [(problem_type, model) for problem_type in problem_types]


def get_all_model_problems():
    return [(problem, model) for problem in PROBLEM_TYPES for model in MODELS]


def get_all_model_problem_metrics():
    return [(problem, model, metric) for model in MODELS for problem in PROBLEM_TYPES for metric in METRICS[problem]]


@pytest.mark.parametrize("problem_type, model", get_one_model_problem())
def test_off(problem_type, model, get_dataset_map, fit_helper):
    fit_args = dict(
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, **common_args)

    with pytest.raises(FileNotFoundError):
        predictor.learning_curves()


@pytest.mark.parametrize("problem_type, model", get_one_model_problem())
def test_flag_false(problem_type, model, get_dataset_map, fit_helper):
    fit_args = dict(
        learning_curves=False,
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, **common_args)

    with pytest.raises(FileNotFoundError):
        predictor.learning_curves()


@pytest.mark.parametrize("problem_type, model", get_all_model_problems())
def test_flag_true(problem_type, model, get_dataset_map, fit_helper):
    fit_args = dict(
        learning_curves=True,
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, **common_args)
    meta_data, model_data = predictor.learning_curves()

    model = DEFAULT_MODEL_NAMES[MODEL_TYPES[model]]
    model_metrics = model_data[model][0]
    metric_count, eval_set_count, iteration_count = np.array(model_data[model][1]).shape

    assert metric_count == 1
    assert eval_set_count == 2
    assert model_metrics[0] == predictor.eval_metric.name


@pytest.mark.parametrize("problem_type, model", get_all_problems())
def test_metrics(problem_type, model, get_dataset_map, fit_helper):
    metrics = [metric for metric in METRICS[problem_type]]

    fit_args = dict(
        learning_curves={
            "metrics": metrics,
        },
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, **common_args)
    meta_data, model_data = predictor.learning_curves()

    metrics = list(set([get_metric(metric, problem_type, "eval_metric").name for metric in metrics]))
    model = DEFAULT_MODEL_NAMES[MODEL_TYPES[model]]
    model_metrics = model_data[model][0]
    assert sorted(model_metrics) == sorted(metrics)


def custom_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100

@pytest.mark.parametrize("problem_type, model", get_one_model_problem())
def test_custom_metrics(problem_type, model, get_dataset_map, fit_helper):
    myaccuracy = make_scorer("myaccuracy", custom_metric, needs_class=True)

    fit_args = dict(
        learning_curves={
            "metrics": [myaccuracy, "accuracy"]
        },
        hyperparameters={model: {}},
    )

    args = common_args.copy()
    del args["sample_size"]

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, **args)
    meta_data, model_data = predictor.learning_curves()

    model = DEFAULT_MODEL_NAMES[MODEL_TYPES[model]]
    metrics = model_data[model][0]
    idx, myidx = metrics.index("accuracy"), metrics.index("myaccuracy")
    myaccuracy_scores = model_data[model][1][myidx]
    accuracy_scores = [[metric * 100 for metric in eval_set] for eval_set in model_data[model][1][idx]]
    assert myaccuracy_scores == accuracy_scores


@pytest.mark.parametrize("problem_type, model", get_all_problems())
@pytest.mark.parametrize("use_error", [True, False])
def test_metric_format(problem_type, model, use_error, get_dataset_map, fit_helper):
    metrics = {
        BINARY: "log_loss",
        MULTICLASS: "log_loss",
        REGRESSION: "mean_squared_error"
    }[problem_type]

    fit_args = dict(
        learning_curves={
            "metrics": metrics,
            "use_error": use_error,
        },
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, **common_args)
    meta_data, model_data = predictor.learning_curves()

    model = DEFAULT_MODEL_NAMES[MODEL_TYPES[model]]
    curve = model_data[model][1][0][0]

    if use_error:
        assert all(val >= 0 for val in curve)
    else:
        assert all(val <= 0 for val in curve)


@pytest.mark.parametrize("problem_type, model", get_all_models())
def test_with_test_data(problem_type, model, get_dataset_map, fit_helper):
    fit_args = dict(
        learning_curves=True,
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(
        dataset_name=dataset_name, 
        fit_args=fit_args, 
        use_test_data=True, 
        use_test_for_val=True, 
        **common_args
    )
    meta_data, model_data = predictor.learning_curves()
    model = DEFAULT_MODEL_NAMES[MODEL_TYPES[model]]

    metric_count, eval_set_count, iteration_count = np.array(model_data[model][1]).shape
    assert eval_set_count == 3

    # ensure test_data preprocessing is same as val_data preprocessing
    eval_sets = [[], [], []]

    for m in range(metric_count):
        for e in range(eval_set_count):
            eval_sets[e].append(model_data[model][1][m][e])
    
    _, val, test = eval_sets
    assert val == test


# def leaderboard_result(predictor: TabularPredictor, model: str):
#     df = predictor.leaderboard(score_format='error')
#     return list(df[df["model"] == model]["metric_error_val"])[0]

# # why is nn torch failing?
# @pytest.mark.parametrize("problem_type, model, metric", get_all_model_problem_metrics())
# def test_correctness(problem_type, model, metric, get_dataset_map, fit_helper):
#     metric = get_metric(metric, problem_type, "eval_metric")

#     init_args = {
#         "eval_metric": metric,
#         "verbosity": 4,
#     }

#     fit_args = dict(
#         learning_curves={
#             "metrics": metric,
#             "use_error": True,
#         },
#         hyperparameters={model: extended_run_hyperparams[model]},
#     )

#     args = common_args.copy()
#     del args["sample_size"]

#     dataset_name = get_dataset_map[problem_type]
#     predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, init_args=init_args, fit_args=fit_args, **args)

#     meta_data, model_data = predictor.learning_curves()
#     model_name = DEFAULT_MODEL_NAMES[MODEL_TYPES[model]]

#     def score(tabular_predictor):
#         model_type = tabular_predictor._trainer.get_model_attribute(model=model_name, attribute="type")
#         model_path = os.path.join(tabular_predictor.path, 'models', model_name)
#         model_obj = model_type.load(model_path)

#         X_val = tabular_predictor._trainer.load_X_val()
#         y_val = tabular_predictor._trainer.load_y_val()

#         result = model_obj.score(X=X_val, y=y_val, metric = metric)
#         return metric.convert_score_to_error(result)

#     # default eval_metric curve on validation dataset
#     curve = model_data[model_name][1][0][1]
#     assert len(curve) == model_iterations[model]

#     best = min(curve)
#     # predictor_result = leaderboard_result(predictor=predictor, model=model_name)
#     assert math.isclose(best, score(predictor), rel_tol=1e-02)

#     fit_args["learning_curves"] = False
#     clean_predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, init_args=init_args, fit_args=fit_args, **args)
#     # clean_predictor_result = leaderboard_result(predictor=clean_predictor, model=model_name)
#     assert math.isclose(best, score(clean_predictor), rel_tol=1e-02)


@pytest.mark.parametrize("learning_curve_supported_class", [LGBModel, XGBoostModel, TabularNeuralNetTorchModel])
def test_supported_class_tags(learning_curve_supported_class):
    assert learning_curve_supported_class._get_class_tags().get("supports_learning_curves", False)


@pytest.fixture()
def get_dataset_map():
    return {
        BINARY: "adult", 
        MULTICLASS: "covertype_small", 
        REGRESSION: "ames",
    }

