import pytest

import autogluon.core as ag
from autogluon.timeseries.models.gluonts.prophet import ProphetModel

from ...common import DUMMY_TS_DATAFRAME

pytest.importorskip("prophet")


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
    assert model.gts_predictor.prophet_params.get("n_changepoints") == n_changepoints  # noqa  # noqa  # noqa


@pytest.mark.parametrize("growth", ["linear", "logistic"])
@pytest.mark.parametrize("n_changepoints", [3, 5])
def test_when_prophet_model_saved_then_prophet_parameters_are_loaded(growth, n_changepoints, temp_model_path):
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
    assert loaded_model.gts_predictor.prophet_params.get("n_changepoints") == n_changepoints  # noqa  # noqa


def test_when_hyperparameter_tune_called_on_prophet_then_hyperparameters_are_passed_to_underlying_model(
    temp_model_path,
):
    model = ProphetModel(
        path=temp_model_path,
        freq="H",
        prediction_length=4,
        hyperparameters={"growth": "linear", "n_changepoints": ag.Int(3, 4)},
    )
    hyperparameter_tune_kwargs = "auto"

    models, results = model.hyperparameter_tune(
        time_limit=100,
        train_data=DUMMY_TS_DATAFRAME,
        val_data=DUMMY_TS_DATAFRAME,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )

    assert len(results["config_history"]) == 2
    assert results["config_history"][0]["n_changepoints"] == 3
    assert results["config_history"][1]["n_changepoints"] == 4

    assert all(c["growth"] == "linear" for c in results["config_history"].values())
