"""Unit tests and utils common to all models"""
from unittest import mock

import numpy as np
import pytest

import autogluon.core as ag
from autogluon.core.scheduler.scheduler_factory import scheduler_factory

from autogluon.timeseries import TimeSeriesEvaluator
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

from ..common import DUMMY_TS_DATAFRAME, dict_equal_primitive
from .test_gluonts import TESTABLE_MODELS as GLUONTS_TESTABLE_MODELS


TESTABLE_MODELS = GLUONTS_TESTABLE_MODELS
AVAILABLE_METRICS = TimeSeriesEvaluator.AVAILABLE_METRICS


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_models_can_be_initialized(model_class, temp_model_path):
    model = model_class(path=temp_model_path, freq="H", prediction_length=24)
    assert isinstance(model, AbstractTimeSeriesModel)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
@pytest.mark.parametrize("metric", AVAILABLE_METRICS)
def test_when_fit_called_then_models_train_and_all_scores_can_be_computed(
    model_class, prediction_length, metric, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=prediction_length,
        hyperparameters={"epochs": 2},
    )

    model.fit(train_data=DUMMY_TS_DATAFRAME)
    score = model.score(DUMMY_TS_DATAFRAME, metric)

    assert isinstance(score, float)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
def test_when_predict_for_scoring_called_then_model_receives_truncated_data(
    model_class, prediction_length, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=prediction_length,
        hyperparameters={"epochs": 1},
    )
    model.fit(train_data=DUMMY_TS_DATAFRAME)

    with mock.patch.object(model, "predict") as patch_method:
        _ = model.predict_for_scoring(DUMMY_TS_DATAFRAME)

        call_df, = patch_method.call_args[0]

        for j in DUMMY_TS_DATAFRAME.iter_items():
            assert np.allclose(
                call_df.loc[j], DUMMY_TS_DATAFRAME.loc[j][:-prediction_length]
            )


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_models_saved_then_they_can_be_loaded(model_class, temp_model_path):
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

    assert dict_equal_primitive(model.params, loaded_model.params)
    assert dict_equal_primitive(model.params_aux, loaded_model.params_aux)
    assert dict_equal_primitive(model.metadata, loaded_model.metadata)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_hyperparameter_spaces_when_tune_called_then_tuning_output_correct(
    model_class, temp_model_path
):
    scheduler_options = scheduler_factory(hyperparameter_tune_kwargs="auto")

    model = model_class(
        path=temp_model_path,
        freq="H",
        quantile_levels=[0.1, 0.9],
        hyperparameters={
            "epochs": ag.Int(3, 4),
        },
    )

    _, _, results = model.hyperparameter_tune(
        scheduler_options=scheduler_options,
        time_limit=100,
        train_data=DUMMY_TS_DATAFRAME,
        val_data=DUMMY_TS_DATAFRAME,
    )

    assert len(results["config_history"]) == 2
    assert results["config_history"][0]["epochs"] == 3
    assert results["config_history"][1]["epochs"] == 4
