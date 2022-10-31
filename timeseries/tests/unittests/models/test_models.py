"""Unit tests and utils common to all models"""
import itertools
import os
import shutil
import tempfile
from unittest import mock

import numpy as np
import pytest
from flaky import flaky

import autogluon.core as ag
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesEvaluator
from autogluon.timeseries.models import ETSModel, DeepARModel
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

from ..common import DUMMY_TS_DATAFRAME, dict_equal_primitive, get_data_frame_with_item_index
from .test_autogluon_tabular import TESTABLE_MODELS as TABULAR_TESTABLE_MODELS
from .gluonts.test_gluonts import TESTABLE_MODELS as GLUONTS_TESTABLE_MODELS
from .test_local import TESTABLE_MODELS as LOCAL_TESTABLE_MODELS
from .test_sktime import TESTABLE_MODELS as SKTIME_TESTABLE_MODELS
from .test_statsmodels import TESTABLE_MODELS as STATSMODELS_TESTABLE_MODELS

AVAILABLE_METRICS = TimeSeriesEvaluator.AVAILABLE_METRICS
TESTABLE_MODELS = (
    GLUONTS_TESTABLE_MODELS
    + SKTIME_TESTABLE_MODELS
    + STATSMODELS_TESTABLE_MODELS
    + TABULAR_TESTABLE_MODELS
    + LOCAL_TESTABLE_MODELS
)
TESTABLE_PREDICTION_LENGTHS = [1, 5]


@pytest.fixture(scope="module")
def trained_models():
    models = {}
    model_paths = []
    for model_class, prediction_length in itertools.product(TESTABLE_MODELS, TESTABLE_PREDICTION_LENGTHS):
        temp_model_path = tempfile.mkdtemp()
        model = model_class(
            path=temp_model_path + os.path.sep,
            freq="H",
            prediction_length=prediction_length,
            hyperparameters={"epochs": 1, "maxiter": 1},
        )

        model.fit(train_data=DUMMY_TS_DATAFRAME)
        models[(prediction_length, repr(model_class))] = model
        model_paths.append(temp_model_path)

    yield models

    for td in model_paths:
        shutil.rmtree(td)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_models_can_be_initialized(model_class, temp_model_path):
    model = model_class(path=temp_model_path, freq="H", prediction_length=24)
    assert isinstance(model, AbstractTimeSeriesModel)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
@pytest.mark.parametrize("metric", AVAILABLE_METRICS)
def test_when_fit_called_then_models_train_and_all_scores_can_be_computed(
    model_class, prediction_length, metric, trained_models
):
    model = trained_models[(prediction_length, repr(model_class))]
    score = model.score(DUMMY_TS_DATAFRAME, metric)

    assert isinstance(score, float)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
def test_when_predict_for_scoring_called_then_model_receives_truncated_data(
    model_class, prediction_length, trained_models
):
    model = trained_models[(prediction_length, repr(model_class))]

    with mock.patch.object(model, "predict") as patch_method:
        _ = model.predict_for_scoring(DUMMY_TS_DATAFRAME)

        (call_df,) = patch_method.call_args[0]

        for j in DUMMY_TS_DATAFRAME.item_ids:
            assert np.allclose(call_df.loc[j], DUMMY_TS_DATAFRAME.loc[j][:-prediction_length])


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", TESTABLE_PREDICTION_LENGTHS)
def test_when_models_saved_then_they_can_be_loaded(model_class, trained_models, prediction_length):
    model = trained_models[(prediction_length, repr(model_class))]

    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert dict_equal_primitive(model.params, loaded_model.params)
    assert dict_equal_primitive(model.params_aux, loaded_model.params_aux)
    assert dict_equal_primitive(model.metadata, loaded_model.metadata)


@flaky
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_hyperparameter_spaces_when_tune_called_then_tuning_output_correct(model_class, temp_model_path):
    model = model_class(
        path=temp_model_path,
        freq="H",
        quantile_levels=[0.1, 0.9],
        hyperparameters={
            "epochs": ag.Int(1, 3),
        },
    )
    if model.name == "AutoGluonTabular":
        pytest.skip("AutoGluonTabular model doesn't support HPO")

    num_trials = 2

    hpo_results, _ = model.hyperparameter_tune(
        hyperparameter_tune_kwargs={"num_trials": num_trials, "scheduler": "local", "searcher": "random"},
        time_limit=300,
        train_data=DUMMY_TS_DATAFRAME,
        val_data=DUMMY_TS_DATAFRAME,
    )
    assert len(hpo_results) == num_trials
    for result in hpo_results.values():
        assert 1 <= result["hyperparameters"]["epochs"] <= 3


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_no_freq_argument_when_fit_called_with_freq_then_model_does_not_raise_error(
    model_class, temp_model_path
):
    model = model_class(path=temp_model_path)
    try:
        model.fit(train_data=DUMMY_TS_DATAFRAME, time_limit=2, freq="H")
    except ValueError:
        pytest.fail("unexpected ValueError raised in fit")


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_hyperparameter_spaces_to_init_when_fit_called_then_error_is_raised(model_class, temp_model_path):
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
            train_data=DUMMY_TS_DATAFRAME,
        )


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
            "epochs": 1,
            "maxiter": 1,
        },
    )
    # TFT cannot handle arbitrary quantiles
    if "TemporalFusionTransformer" in model.name:
        return

    model.fit(train_data=DUMMY_TS_DATAFRAME)
    predictions = model.predict(DUMMY_TS_DATAFRAME, quantile_levels=quantile_levels)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(k in predictions.columns for k in ["mean"] + [str(q) for q in quantile_levels])


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
def test_when_fit_called_then_models_train_and_returned_predictor_inference_correct(
    model_class, prediction_length, trained_models
):
    train_data = DUMMY_TS_DATAFRAME
    model = trained_models[(prediction_length, repr(model_class))]

    predictions = model.predict(train_data)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == train_data.item_ids)
    assert all(len(predictions.loc[i]) == prediction_length for i in predicted_item_index)
    assert all(predictions.loc[i].index[0].hour > 0 for i in predicted_item_index)


@pytest.mark.parametrize(
    "model_class",
    [DeepARModel, ETSModel],
)
@pytest.mark.parametrize("test_data_index", [["A", "B"], ["C", "D"], ["A"]])
def test_when_fit_called_then_models_train_and_returned_predictor_inference_aligns_with_time(
    model_class, test_data_index, temp_model_path
):
    prediction_length = 3
    train_data = get_data_frame_with_item_index(["A", "B"], data_length=10)
    test_data = get_data_frame_with_item_index(test_data_index, data_length=15)

    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=prediction_length,
        hyperparameters={"epochs": 1},
    )

    model.fit(train_data=train_data)

    max_hour_in_test = test_data.index.levels[1].max().hour
    predictions = model.predict(test_data)
    min_hour_in_pred = predictions.index.levels[1].min().hour

    assert min_hour_in_pred == max_hour_in_test + 1


@pytest.mark.parametrize(
    "model_class",
    [DeepARModel, ETSModel],
)
@pytest.mark.parametrize(
    "train_data, test_data",
    [
        (get_data_frame_with_item_index([0, 1, 2, 3]), get_data_frame_with_item_index([0, 1, 2, 3])),
        (
            get_data_frame_with_item_index(["A", "B", "C"]),
            get_data_frame_with_item_index(["A", "B", "C"]),
        ),
        (
            get_data_frame_with_item_index(["A", "B", "C"]),
            get_data_frame_with_item_index(["A", "B", "D"]),
        ),
        (
            get_data_frame_with_item_index(["A", "B", "C"]),
            get_data_frame_with_item_index(["A", "B"]),
        ),
        (
            get_data_frame_with_item_index(["A", "B"]),
            get_data_frame_with_item_index(["A", "B", "C"]),
        ),
    ],
)
def test_when_predict_called_with_test_data_then_predictor_inference_correct(
    model_class, temp_model_path, train_data, test_data
):
    prediction_length = 5
    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=prediction_length,
        hyperparameters={"epochs": 1},
    )

    model.fit(train_data=train_data)

    predictions = model.predict(test_data)

    assert isinstance(predictions, TimeSeriesDataFrame)
    assert len(predictions) == test_data.num_items * prediction_length

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == test_data.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == prediction_length for i in predicted_item_index)
    assert all(predictions.loc[i].index[0].hour > 0 for i in predicted_item_index)
