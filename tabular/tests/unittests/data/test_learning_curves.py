import math

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import METRICS, get_metric, make_scorer
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models import LGBModel, TabularNeuralNetTorchModel, XGBoostModel
from autogluon.tabular.testing import FitHelper
from autogluon.tabular.registry import ag_model_registry


def get_default_model_name(model: str) -> str:
    return ag_model_registry.key_to_cls(model).ag_name

model_key_to_cls_map = ag_model_registry.key_to_cls_map()
MODELS = [name for name, model in model_key_to_cls_map.items() if model._get_class_tags().get("supports_learning_curves", False)]
PROBLEM_TYPES = [BINARY, MULTICLASS, REGRESSION]

common_args = {"sample_size": 50, "delete_directory": False, "refit_full": False, "raise_on_model_failure": True}

early_stop = 999999
long_run = 5
short_run = 3

model_iterations = {
    "LightGBM": long_run,
    "XGBoost": long_run,
    "NeuralNetTorch": short_run,
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

metrics_to_test = {
    BINARY: ["roc_auc"],
    MULTICLASS: ["log_loss"],
    REGRESSION: ["root_mean_squared_error"],
}

for model in MODELS:
    if model not in extended_run_hyperparams:
        extended_run_hyperparams[model] = {}

    # TODO: Not sure this will be correct for additional models
    if get_default_model_name(model) not in extended_run_hyperparams:
        extended_run_hyperparams[get_default_model_name(model)] = long_run


def get_one_model_problem():
    problem_type = PROBLEM_TYPES[0]
    model = MODELS[0]
    return [(problem_type, model)]


def get_all_models():
    n = len(PROBLEM_TYPES)
    return [(PROBLEM_TYPES[i % n], model) for i, model in enumerate(MODELS)]


def get_all_problems():
    n = len(MODELS)
    return [(problem_type, MODELS[i % n]) for i, problem_type in enumerate(PROBLEM_TYPES)]


def get_all_model_problems():
    return [(problem, model) for problem in PROBLEM_TYPES for model in MODELS]


def get_all_model_problem_metrics():
    return [(problem, model, metric) for model in MODELS for problem in PROBLEM_TYPES for metric in METRICS[problem]]


# This is much faster to run than `get_all_model_problem_metrics`, but isn't fully comprehensive
# This makes tests run in 79s , vs 1200s with `get_all_model_problem_metrics`.
def get_subset_model_problem_metrics():
    output = [(problem, model, metric) for model in MODELS for problem in PROBLEM_TYPES for metric in metrics_to_test[problem]]
    return output


@pytest.mark.parametrize("problem_type, model", get_one_model_problem())
def test_off(problem_type, model, get_dataset_map):
    fit_args = dict(
        hyperparameters={
            model: extended_run_hyperparams[model],
        },
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, **common_args)

    _, data = predictor.learning_curves()
    assert data == {}


@pytest.mark.parametrize("problem_type, model", get_one_model_problem())
def test_flag_false(problem_type, model, get_dataset_map):
    fit_args = dict(
        learning_curves=False,
        hyperparameters={
            model: extended_run_hyperparams[model],
        },
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, **common_args)

    _, data = predictor.learning_curves()
    assert data == {}


@pytest.mark.parametrize("problem_type, model", get_all_model_problems())
def test_flag_true(problem_type, model, get_dataset_map):
    fit_args = dict(
        learning_curves=True,
        hyperparameters={
            model: extended_run_hyperparams[model],
        },
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, **common_args)

    model = get_default_model_name(model)
    _, model_data = predictor.learning_curves()
    _, model_metrics, data = model_data[model]
    metric_count, eval_set_count, _ = np.array(data).shape

    assert metric_count == 1
    assert eval_set_count == 2
    assert model_metrics[0] == predictor.eval_metric.name


@pytest.mark.parametrize("problem_type, model", get_all_problems())
def test_metrics(problem_type, model, get_dataset_map):
    # get all unique metrics
    metrics = list(set([metric.name for metric in METRICS[problem_type].values()]))
    metrics = [get_metric(name, problem_type) for name in metrics]

    fit_args = dict(
        learning_curves={
            "metrics": metrics,
        },
        hyperparameters={
            model: extended_run_hyperparams[model],
        },
    )

    # FIXME: This is needed due to a bug: https://github.com/autogluon/autogluon/issues/4423
    min_cls_count_train = 10

    dataset_name = get_dataset_map[problem_type]
    predictor = FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, min_cls_count_train=min_cls_count_train, **common_args)

    model = get_default_model_name(model)
    _, model_data = predictor.learning_curves()
    _, model_metrics, data = model_data[model]

    metrics = list(set([get_metric(metric, problem_type, "eval_metric").name for metric in metrics]))
    assert sorted(model_metrics) == sorted(metrics)


def custom_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100


@pytest.mark.parametrize("problem_type, model", get_one_model_problem())
def test_custom_metrics(problem_type, model, get_dataset_map):
    myaccuracy = make_scorer("myaccuracy", custom_metric, needs_class=True)

    fit_args = dict(
        learning_curves={"metrics": [myaccuracy, "accuracy"]},
        hyperparameters={
            model: extended_run_hyperparams[model],
        },
    )

    args = common_args.copy()
    del args["sample_size"]

    dataset_name = get_dataset_map[problem_type]
    predictor = FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, **args)

    model = get_default_model_name(model)
    _, model_data = predictor.learning_curves()
    _, model_metrics, data = model_data[model]

    idx, myidx = model_metrics.index("accuracy"), model_metrics.index("myaccuracy")
    myaccuracy_scores = data[myidx]
    accuracy_scores = [[metric * 100 for metric in eval_set] for eval_set in data[idx]]
    assert myaccuracy_scores == accuracy_scores


# TODO: can't the error = True tests these be checked at the same time as the
# correctness tests?
# @pytest.mark.parametrize("problem_type, model", get_all_problems())
@pytest.mark.parametrize("problem_type, model, metric", get_subset_model_problem_metrics())
@pytest.mark.parametrize("use_error", [True, False])
def test_metric_format(problem_type, model, metric, use_error, get_dataset_map):
    metric = get_metric(metric, problem_type, "eval_metric")

    init_args = {
        "eval_metric": metric,
        # "verbosity": 4,
    }

    fit_args = dict(
        learning_curves={
            "metrics": metric,
            "use_error": use_error,
        },
        hyperparameters={
            model: extended_run_hyperparams[model],
        },
    )

    args = common_args.copy()
    args["sample_size"] = 500

    # FIXME: Avoid needing this, using this to avoid bug: https://github.com/autogluon/autogluon/issues/4423
    args["min_cls_count_train"] = 10

    dataset_name = get_dataset_map[problem_type]
    predictor = FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, init_args=init_args, fit_args=fit_args, **args)

    model = get_default_model_name(model)
    _, model_data = predictor.learning_curves()
    _, _, data = model_data[model]

    curve = np.array(data[0][1])

    if not use_error:
        curve = np.array([metric.convert_score_to_error(val) for val in curve])

    assert np.all(curve >= 0)


@pytest.mark.parametrize("problem_type, model", get_all_models())
def test_with_test_data(problem_type, model, get_dataset_map, get_default_metrics):
    init_args = {
        "eval_metric": get_default_metrics[problem_type],
        "verbosity": 4,
    }

    fit_args = dict(
        learning_curves=True,
        hyperparameters={
            model: extended_run_hyperparams[model],
        },
    )

    common_args["sample_size"] = 1000

    dataset_name = get_dataset_map[problem_type]
    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name, init_args=init_args, fit_args=fit_args, use_test_data=True, use_test_for_val=True, **common_args
    )

    model = get_default_model_name(model)
    _, model_data = predictor.learning_curves()
    eval_sets, _, data = model_data[model]
    metric_count, eval_set_count, _ = np.array(data).shape

    assert eval_set_count == 3

    # ensure test_data preprocessing is same as val_data preprocessing
    eval_sets = [[], [], []]

    for m in range(metric_count):
        for e in range(eval_set_count):
            eval_sets[e].append(data[m][e])

    _, val, test = eval_sets
    assert val == test


# TODO: how should we limit test parameters here? 22.5 min is much too long, but curve correctness
# is a crucial aspect of learning curve generation that should be tested well
# takes 8.7 minutes for full test run (only correctness tests)
@pytest.mark.parametrize("problem_type, model, metric", get_subset_model_problem_metrics())
def test_correctness(problem_type, model, metric, get_dataset_map):
    metric = get_metric(metric, problem_type, "eval_metric")

    init_args = {
        "eval_metric": metric,
        "verbosity": 4,
    }

    fit_args = dict(
        learning_curves={
            "metrics": metric,
            "use_error": True,
        },
        hyperparameters={
            model: extended_run_hyperparams[model],
        },
    )

    args = common_args.copy()
    args["sample_size"] = 1000
    if metric.name == "roc_auc":
        args["sample_size"] = 10000

    dataset_name = get_dataset_map[problem_type]
    predictor = FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, init_args=init_args, fit_args=fit_args, **args)

    model = get_default_model_name(model)
    _, model_data = predictor.learning_curves()
    eval_sets, _, data = model_data[model]

    def error(tabular_predictor: TabularPredictor):
        df = tabular_predictor.leaderboard(score_format="error")
        return list(df[df["model"] == model]["metric_error_val"])[0]

    def equal(a, b):
        if metric.needs_proba or metric.needs_threshold or problem_type == "regression":
            tol = 1e-06 if metric.name != "roc_auc" else 1e-02
            return math.isclose(a, b, rel_tol=tol)
        return a == b

    val_index = eval_sets.index("val")
    curve = data[0][val_index]  # get default eval_metric curve on validation dataset
    best = min(curve)

    assert len(curve) == model_iterations[model]
    assert equal(best, error(predictor))

    fit_args["learning_curves"] = False
    clean_predictor = FitHelper.fit_and_validate_dataset(dataset_name=dataset_name, init_args=init_args, fit_args=fit_args, **args)

    assert equal(error(predictor), error(clean_predictor))


@pytest.mark.parametrize("learning_curve_supported_class", [LGBModel, XGBoostModel, TabularNeuralNetTorchModel])
def test_supported_class_tags(learning_curve_supported_class):
    assert learning_curve_supported_class._get_class_tags().get("supports_learning_curves", False)


@pytest.fixture()
def get_dataset_map():
    return {
        BINARY: "toy_binary_10",
        MULTICLASS: "toy_multiclass_30",
        REGRESSION: "toy_regression_10",
    }


@pytest.fixture()
def get_default_metrics():
    return {
        BINARY: "accuracy",
        MULTICLASS: "log_loss",
        REGRESSION: "rmse",
    }
