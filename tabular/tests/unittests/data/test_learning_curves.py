import pytest
from autogluon.core.metrics import METRICS, get_metric
from autogluon.core.constants import BINARY, REGRESSION, MULTICLASS
from autogluon.core.constants import DEFAULT_LEARNING_CURVE_METRICS, LEARNING_CURVE_SUPPORTED_MODELS, LEARNING_CURVE_SUPPORTED_PROBLEM_TYPES


def get_one_model_problem():
    problem_type = LEARNING_CURVE_SUPPORTED_PROBLEM_TYPES[0]
    model = LEARNING_CURVE_SUPPORTED_MODELS[0]
    return [(problem_type, model)]


def get_all_models():
    problem_type = LEARNING_CURVE_SUPPORTED_PROBLEM_TYPES[0]
    models = LEARNING_CURVE_SUPPORTED_MODELS
    return [(problem_type, model) for model in models]


def get_all_model_problems():
    problem_types = LEARNING_CURVE_SUPPORTED_PROBLEM_TYPES
    models = LEARNING_CURVE_SUPPORTED_MODELS
    return [(problem, model) for problem in problem_types for model in models]


@pytest.mark.parametrize("problem_type, model", get_one_model_problem())
def test_learning_curves_off(problem_type, model, get_dataset_map, fit_helper):
    fit_args = dict(
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, delete_directory=False)

    with pytest.raises(FileNotFoundError):
        predictor.learning_curves()


@pytest.mark.parametrize("problem_type, model", get_one_model_problem())
def test_learning_curves_false(problem_type, model, get_dataset_map, fit_helper):
    fit_args = dict(
        learning_curves=False,
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, delete_directory=False)

    with pytest.raises(FileNotFoundError):
        predictor.learning_curves()


@pytest.mark.parametrize("problem_type, model", get_all_model_problems())
def test_learning_curves_true(problem_type, model, get_dataset_map, fit_helper):
    import numpy as np
    fit_args = dict(
        learning_curves=True,
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, delete_directory=False)
    lc = predictor.learning_curves()

    model_metrics = lc[1][model][0]
    _, _, curve_count = np.array(lc[1][model][1]).shape
    
    assert curve_count == 2
    assert sorted(model_metrics) == sorted(DEFAULT_LEARNING_CURVE_METRICS[predictor.problem_type])


@pytest.mark.parametrize("problem_type, model", get_one_model_problem())
def test_learning_curves_custom_metrics(problem_type, model, get_dataset_map, fit_helper):
    metrics = [metric for metric in METRICS[problem_type]]

    fit_args = dict(
        learning_curves={
            "metrics": metrics,
        },
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, delete_directory=False)
    lc = predictor.learning_curves()

    metrics = list(set([get_metric(metric, problem_type, "eval_metric").name for metric in metrics]))
    model_metrics = lc[1][model][0]
    assert sorted(model_metrics) == sorted(metrics)


@pytest.mark.parametrize("problem_type, model", get_all_models())
def test_learning_curves_error(problem_type, model, get_dataset_map, fit_helper):
    metrics = {
        BINARY: "log_loss",
        MULTICLASS: "log_loss",
        REGRESSION: "mean_squared_error"
    }[problem_type]

    fit_args = dict(
        learning_curves={
            "metrics": metrics,
        },
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, delete_directory=False)
    lc = predictor.learning_curves()

    assert lc[1][model][1][0][0][0] >= 0

@pytest.mark.parametrize("problem_type, model", get_all_models())
def test_learning_curves_score(problem_type, model, get_dataset_map, fit_helper):
    metrics = {
        BINARY: "log_loss",
        MULTICLASS: "log_loss",
        REGRESSION: "mean_squared_error"
    }[problem_type]

    fit_args = dict(
        learning_curves={
            "metrics": metrics,
            "use_error": False,
        },
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, delete_directory=False)
    lc = predictor.learning_curves()

    assert lc[1][model][1][0][0][0] <= 0

@pytest.mark.parametrize("problem_type, model", get_all_models())
def test_learning_curves_with_test_data(problem_type, model, get_dataset_map, fit_helper):
    fit_args = dict(
        learning_curves=True,
        hyperparameters={model: {}},
    )

    dataset_name = get_dataset_map[problem_type]
    predictor = fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, use_test_data=True, use_test_for_val=True, delete_directory=False)
    lc = predictor.learning_curves()

    import numpy as np
    _, _, curve_count = np.array(lc[1][model][1]).shape
    assert curve_count == 3

    # ensure test_data preprocessing is same as val_data preprocessing
    metrics = lc[1][model][0]
    iterations = len(lc[1][model][1])
    curves = [
        { metric : [] for metric in metrics },
        { metric : [] for metric in metrics },
        { metric : [] for metric in metrics },
    ]

    for i in range(iterations):
        for j, metric in enumerate(metrics):
            for c in range(3):
                score = lc[1][model][1][i][j][c]
                curves[c][metric].append(score)
    
    _, val, test = curves
    assert val == test


@pytest.fixture()
def get_dataset_map():
    return {
        BINARY: "adult", 
        MULTICLASS: "covertype_small", 
        REGRESSION: "ames",
    }

