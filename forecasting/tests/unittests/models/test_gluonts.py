import random

import pandas as pd
import pytest
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor as GluonTSPredictor

import autogluon.core as ag
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.forecasting.models.gluonts import (
    DeepARModel,
    # AutoTabularModel,
    SimpleFeedForwardModel,
    MQCNNModel,
)
from autogluon.forecasting.models.gluonts.abstract_gluonts import AbstractGluonTSModel
from autogluon.forecasting.utils.metric_utils import AVAILABLE_METRICS

TESTABLE_MODELS = [
    # AutoTabularModel,  # TODO: enable tests when model is stabilized
    DeepARModel,
    MQCNNModel,
    SimpleFeedForwardModel,
]

DUMMY_DATASET = ListDataset(
    [
        {
            "target": [random.random() for _ in range(10)],
            "start": pd.Timestamp("2022-01-01 00:00:00"),  # noqa
            "item_id": 0,
        },
        {
            "target": [random.random() for _ in range(10)],
            "start": pd.Timestamp("2022-01-01 00:00:00"),  # noqa
            "item_id": 1,
        },
    ],
    freq="H",
)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_models_can_be_initialized(model_class, temp_model_path):
    model = model_class(path=temp_model_path, freq="H", prediction_length=24)
    assert isinstance(model, AbstractGluonTSModel)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [5, 10])
def test_when_fit_called_then_models_train_and_returned_predictor_inference_correct(
    model_class, prediction_length, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=prediction_length,
        hyperparameters={"epochs": 2},
    )

    assert not model.gts_predictor
    model.fit(train_data=DUMMY_DATASET)
    assert isinstance(model.gts_predictor, GluonTSPredictor)

    predictions = model.predict(DUMMY_DATASET)

    assert len(predictions) == len(DUMMY_DATASET)
    assert all(len(df) == prediction_length for _, df in predictions.items())
    assert all(df.index[0].hour for _, df in predictions.items())


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
@pytest.mark.parametrize("metric", [AVAILABLE_METRICS])
def test_when_fit_called_then_models_train_and_all_scores_can_be_computed(
    model_class, prediction_length, metric, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=prediction_length,
        hyperparameters={"epochs": 2},
    )

    model.fit(train_data=DUMMY_DATASET)
    score = model.score(DUMMY_DATASET)

    assert isinstance(score, float)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("time_limit", [10, None])
def test_given_time_limit_when_fit_called_then_models_train_correctly(
    model_class, time_limit, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=5,
        hyperparameters={"epochs": 2},
    )

    assert not model.gts_predictor
    model.fit(train_data=DUMMY_DATASET, time_limit=time_limit)
    assert isinstance(model.gts_predictor, GluonTSPredictor)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_no_freq_argument_when_fit_called_then_model_raises_value_error(
    model_class, temp_model_path
):
    model = model_class(path=temp_model_path)
    with pytest.raises(ValueError):
        model.fit(train_data=DUMMY_DATASET, time_limit=10)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_no_freq_argument_when_fit_called_with_freq_then_model_does_not_raise_error(
    model_class, temp_model_path
):
    model = model_class(path=temp_model_path)
    try:
        model.fit(train_data=DUMMY_DATASET, time_limit=2, freq="H")
    except ValueError:
        pytest.fail("unexpected ValueError raised in fit")


@pytest.mark.timeout(4)
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_low_time_limit_when_fit_called_then_model_training_does_not_exceed_time_limit(
    model_class, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=5,
        hyperparameters={"epochs": 20000},
    )

    assert not model.gts_predictor
    model.fit(train_data=DUMMY_DATASET, time_limit=2)
    assert isinstance(model.gts_predictor, GluonTSPredictor)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_hyperparameter_spaces_when_tune_called_then_tuning_output_correct(
    model_class, temp_model_path
):
    scheduler_options = scheduler_factory(hyperparameter_tune_kwargs="auto")

    model = model_class(
        path=temp_model_path,
        freq="H",
        hyperparameters={
            "epochs": ag.Int(3, 4),
            "ag_args_fit": {"quantile_levels": [0.1, 0.9]},
        },
    )

    _, _, results = model.hyperparameter_tune(
        scheduler_options=scheduler_options,
        time_limit=100,
        train_data=DUMMY_DATASET,
        val_data=DUMMY_DATASET,
    )

    assert len(results["config_history"]) == 2
    assert results["config_history"][0]["epochs"] == 3
    assert results["config_history"][1]["epochs"] == 4


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_hyperparameter_spaces_to_init_when_fit_called_then_error_is_raised(
    model_class, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        hyperparameters={
            "epochs": ag.Int(3, 4),
            "ag_args_fit": {"quantile_levels": [0.1, 0.9]},
        },
    )
    with pytest.raises(ValueError, match=".*hyperparameter_tune.*"):
        model.fit(
            train_data=DUMMY_DATASET,
        )


# TODO: test models can save correctly
# TODO: test other inherited functionality
# TODO: test model hyperparameters are passed correctly
