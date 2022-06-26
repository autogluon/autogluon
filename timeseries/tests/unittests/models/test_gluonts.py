from functools import partial

import pytest
from flaky import flaky
from gluonts.model.prophet import PROPHET_IS_INSTALLED
from gluonts.model.predictor import Predictor as GluonTSPredictor
from gluonts.model.seq2seq import MQRNNEstimator
from gluonts.model.transformer import TransformerEstimator

import autogluon.core as ag
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.timeseries.models.gluonts import (
    DeepARModel,
    # AutoTabularModel,
    GenericGluonTSModel,
    MQCNNModel,
    MQRNNModel,
    ProphetModel,
    SimpleFeedForwardModel,
    TransformerModel,
)
from autogluon.timeseries.models.gluonts.models import GenericGluonTSModelFactory

from ..common import DUMMY_TS_DATAFRAME

TESTABLE_MODELS = [
    # AutoTabularModel,  # TODO: enable tests when model is stabilized
    DeepARModel,
    MQCNNModel,
    # MQRNNModel,
    SimpleFeedForwardModel,
    # TransformerModel,
    partial(
        GenericGluonTSModel, gluonts_estimator_class=MQRNNEstimator
    ),  # partial constructor for generic model
    GenericGluonTSModelFactory(TransformerEstimator),
]

# if PROPHET_IS_INSTALLED:
#     TESTABLE_MODELS += [ProphetModel]


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
    model.fit(train_data=DUMMY_TS_DATAFRAME, time_limit=time_limit)
    assert isinstance(model.gts_predictor, GluonTSPredictor)


# @flaky(max_runs=3)
# @pytest.mark.timeout(4)
@pytest.mark.skip(reason="Timeout spuriously fails in CI")
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
    model.fit(train_data=DUMMY_TS_DATAFRAME, time_limit=2)
    assert isinstance(model.gts_predictor, GluonTSPredictor)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_models_saved_then_gluonts_predictors_can_be_loaded(
    model_class, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        quantile_levels=[0.1, 0.9],
        hyperparameters={
            "epochs": 1,
        },
    )
    model.fit(
        train_data=DUMMY_TS_DATAFRAME,
    )
    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert model.gluonts_estimator_class is loaded_model.gluonts_estimator_class
    assert loaded_model.gts_predictor == model.gts_predictor


@pytest.mark.skipif(
    not PROPHET_IS_INSTALLED,
    reason="Prophet is not installed. Run `pip install prophet`",
)
@pytest.mark.parametrize("growth", ["linear", "logistic"])
@pytest.mark.parametrize("n_changepoints", [3, 5])
def test_when_fit_called_on_prophet_then_hyperparameters_are_passed_to_underlying_model(
    growth, n_changepoints, temp_model_path
):
    model = ProphetModel(
        path=temp_model_path,
        freq="H",
        prediction_length=4,
        hyperparameters={"growth": growth, "n_changepoints": n_changepoints},
    )

    model.fit(train_data=DUMMY_TS_DATAFRAME)

    assert model.gts_predictor.prophet_params.get("growth") == growth  # noqa
    assert (
        model.gts_predictor.prophet_params.get("n_changepoints")
        == n_changepoints  # noqa
    )  # noqa


@pytest.mark.skipif(
    not PROPHET_IS_INSTALLED,
    reason="Prophet is not installed. Run `pip install prophet`",
)
@pytest.mark.parametrize("growth", ["linear", "logistic"])
@pytest.mark.parametrize("n_changepoints", [3, 5])
def test_when_prophet_model_saved_then_prophet_parameters_are_loaded(
    growth, n_changepoints, temp_model_path
):
    model = ProphetModel(
        path=temp_model_path,
        freq="H",
        quantile_levels=[0.1, 0.9],
        hyperparameters={"growth": growth, "n_changepoints": n_changepoints},
    )
    model.fit(
        train_data=DUMMY_TS_DATAFRAME,
    )
    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert loaded_model.gts_predictor.prophet_params.get("growth") == growth  # noqa
    assert (
        loaded_model.gts_predictor.prophet_params.get("n_changepoints")  # noqa
        == n_changepoints
    )  # noqa


@pytest.mark.skipif(
    not PROPHET_IS_INSTALLED,
    reason="Prophet is not installed. Run `pip install prophet`",
)
def test_when_hyperparameter_tune_called_on_prophet_then_hyperparameters_are_passed_to_underlying_model(
    temp_model_path,
):
    scheduler_options = scheduler_factory(hyperparameter_tune_kwargs="auto")

    model = ProphetModel(
        path=temp_model_path,
        freq="H",
        prediction_length=4,
        hyperparameters={"growth": "linear", "n_changepoints": ag.Int(3, 4)},
    )
    _, _, results = model.hyperparameter_tune(
        scheduler_options=scheduler_options,
        time_limit=100,
        train_data=DUMMY_TS_DATAFRAME,
        val_data=DUMMY_TS_DATAFRAME,
    )

    assert len(results["config_history"]) == 2
    assert results["config_history"][0]["n_changepoints"] == 3
    assert results["config_history"][1]["n_changepoints"] == 4

    assert all(c["growth"] == "linear" for c in results["config_history"].values())
