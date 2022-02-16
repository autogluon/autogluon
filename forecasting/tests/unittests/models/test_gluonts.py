import random
import tempfile

import pandas as pd
import pytest
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor as GluonTSPredictor

from autogluon.forecasting.models.gluonts import (
    DeepARModel, AutoTabularModel, SimpleFeedForwardModel, MQCNNModel
)
from autogluon.forecasting.models.gluonts.abstract_gluonts import AbstractGluonTSModel

TESTABLE_MODELS = [
    # AutoTabularModel,  # TODO: enable tests when model is stabilized
    DeepARModel,
    MQCNNModel,
    SimpleFeedForwardModel,
]

DUMMY_DATASET = ListDataset([
    {
        "target": [random.random() for _ in range(10)],
        "start": pd.Timestamp("2022-01-01 00:00:00"),
        "item_id": 0,
    },
    {
        "target": [random.random() for _ in range(10)],
        "start": pd.Timestamp("2022-01-01 00:00:00"),
        "item_id": 1,
    }
], freq="H")


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_models_initializable(model_class):
    with tempfile.TemporaryDirectory() as tp:
        model = model_class(tp, freq="H", prediction_length=24)
    assert isinstance(model, AbstractGluonTSModel)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [5, 10])
def test_when_fit_called_then_models_train_and_returned_predictor_inference_correct(
    model_class, prediction_length
):
    with tempfile.TemporaryDirectory() as tp:
        model = model_class(
            tp,
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
@pytest.mark.parametrize("time_limit", [10, None])
def test_given_time_limit_when_fit_called_then_models_train_correctly(
    model_class, time_limit
):
    with tempfile.TemporaryDirectory() as tp:
        model = model_class(
            tp,
            freq="H",
            prediction_length=5,
            hyperparameters={"epochs": 2},
        )

        assert not model.gts_predictor
        model.fit(train_data=DUMMY_DATASET, time_limit=time_limit)
        assert isinstance(model.gts_predictor, GluonTSPredictor)


@pytest.mark.timeout(4)
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_low_time_limit_when_fit_called_then_model_training_does_not_exceed_time_limit(
    model_class
):
    with tempfile.TemporaryDirectory() as tp:
        model = model_class(
            tp,
            freq="H",
            prediction_length=5,
            hyperparameters={"epochs": 20000},
        )

        assert not model.gts_predictor
        model.fit(train_data=DUMMY_DATASET, time_limit=2)
        assert isinstance(model.gts_predictor, GluonTSPredictor)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_hyperparameter_spaces_when_tune_called_then_tuning_works(model_class):
    pass

# TODO: test other inherited functionality
# TODO: test model hyperparameters are passed correctly
