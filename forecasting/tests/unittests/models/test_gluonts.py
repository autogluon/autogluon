from functools import partial

import pytest
from gluonts.model.prophet import PROPHET_IS_INSTALLED
from gluonts.model.predictor import Predictor as GluonTSPredictor
from gluonts.model.seq2seq import MQRNNEstimator
from gluonts.model.transformer import TransformerEstimator

import autogluon.core as ag
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.forecasting.models.gluonts import (
    DeepARModel,
    # AutoTabularModel,
    GenericGluonTSModel,
    MQCNNModel,
    ProphetModel,
    SimpleFeedForwardModel,
)

from .common import DUMMY_DATASET

TESTABLE_MODELS = [
    # AutoTabularModel,  # TODO: enable tests when model is stabilized
    DeepARModel,
    MQCNNModel,
    SimpleFeedForwardModel,
    partial(
        GenericGluonTSModel, gluonts_estimator_class=MQRNNEstimator
    ),  # partial constructor for generic model
    partial(GenericGluonTSModel, gluonts_estimator_class=TransformerEstimator),
]

if PROPHET_IS_INSTALLED:
    TESTABLE_MODELS += [ProphetModel]


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
def test_given_hyperparameter_spaces_to_init_when_fit_called_then_error_is_raised(
    model_class, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        quantile_levels=[0.1, 0.9],
        hyperparameters={
            "epochs": ag.Int(3, 4),
        },
    )
    with pytest.raises(ValueError, match=".*hyperparameter_tune.*"):
        model.fit(
            train_data=DUMMY_DATASET,
        )


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
        train_data=DUMMY_DATASET,
    )
    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert model.gluonts_estimator_class is loaded_model.gluonts_estimator_class
    assert loaded_model.gts_predictor == model.gts_predictor


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize(
    "quantile_levels",
    [
        [0.1, 0.44, 0.9],
        [0.1, 0.5, 0.9],
    ],
)
def test_when_fit_called_then_models_train_and_returned_predictor_inference_has_mean_and_correct_quantiles(
    model_class, quantile_levels, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=3,
        quantile_levels=quantile_levels,
        hyperparameters={
            "epochs": 2,
        },
    )

    model.fit(train_data=DUMMY_DATASET)

    predictions = model.predict(DUMMY_DATASET, quantile_levels=quantile_levels)

    assert len(predictions) == len(DUMMY_DATASET)
    for k in ["mean"] + [str(q) for q in quantile_levels]:
        assert all(k in df.columns for _, df in predictions.items())


@pytest.mark.skipif(not PROPHET_IS_INSTALLED, reason="Prophet is not installed. Run `pip install fbprophet`")
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

    model.fit(train_data=DUMMY_DATASET)

    assert model.gts_predictor.prophet_params.get("growth") == growth  # noqa
    assert (
        model.gts_predictor.prophet_params.get("n_changepoints") == n_changepoints
    )  # noqa


@pytest.mark.skipif(not PROPHET_IS_INSTALLED, reason="Prophet is not installed. Run `pip install fbprophet`")
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
        train_data=DUMMY_DATASET,
    )
    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert loaded_model.gts_predictor.prophet_params.get("growth") == growth  # noqa
    assert (
        loaded_model.gts_predictor.prophet_params.get("n_changepoints")
        == n_changepoints
    )  # noqa


@pytest.mark.skipif(not PROPHET_IS_INSTALLED, reason="Prophet is not installed. Run `pip install fbprophet`")
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
        train_data=DUMMY_DATASET,
        val_data=DUMMY_DATASET,
    )

    assert len(results["config_history"]) == 2
    assert results["config_history"][0]["n_changepoints"] == 3
    assert results["config_history"][1]["n_changepoints"] == 4

    assert all(c["growth"] == "linear" for c in results["config_history"].values())
