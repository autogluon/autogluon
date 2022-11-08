"""Unit tests for learners"""
import os
import shutil
import tempfile
from collections import defaultdict
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import autogluon.core as ag
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.learner import TimeSeriesLearner
from autogluon.timeseries.models import DeepARModel, ETSModel
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_single_time_series

from .common import DUMMY_TS_DATAFRAME, get_data_frame_with_variable_lengths, get_static_features

TEST_HYPERPARAMETER_SETTINGS = [
    {"SimpleFeedForward": {"epochs": 1, "num_batches_per_epoch": 1}},
    {"DeepAR": {"epochs": 1, "num_batches_per_epoch": 1}, "Naive": {}},
]
TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS = [1, 2]


@pytest.fixture(scope="module")
def trained_learners():
    learners = {}
    model_paths = []
    for hp in TEST_HYPERPARAMETER_SETTINGS:
        temp_model_path = tempfile.mkdtemp()
        learner = TimeSeriesLearner(
            path_context=temp_model_path + os.path.sep,
            eval_metric="MASE",
            prediction_length=3,
        )
        learner.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hp,
            val_data=DUMMY_TS_DATAFRAME,
        )
        learners[repr(hp)] = learner
        model_paths.append(temp_model_path)

    yield learners

    for td in model_paths:
        shutil.rmtree(td)


def test_learner_can_be_initialized(temp_model_path):
    learner = TimeSeriesLearner(path_context=temp_model_path)
    assert isinstance(learner, TimeSeriesLearner)


# smoke test for the short 'happy path'
@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_when_learner_called_then_training_is_performed(hyperparameters, trained_learners):
    learner = trained_learners[repr(hyperparameters)]
    assert learner.load_trainer().get_model_names()


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_hyperparameters_when_learner_called_then_leaderboard_is_correct(
    trained_learners, hyperparameters, expected_board_length
):
    learner = trained_learners[repr(hyperparameters)]
    leaderboard = learner.leaderboard()

    if learner.load_trainer().enable_ensemble and len(hyperparameters) > 1:
        expected_board_length += 1

    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_hyperparameters_when_learner_called_then_model_can_predict(
    trained_learners, hyperparameters, expected_board_length
):
    learner = trained_learners[repr(hyperparameters)]
    predictions = learner.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("model_name", ["DeepAR", "SimpleFeedForward"])
def test_given_hyperparameters_with_spaces_when_learner_called_then_hpo_is_performed(temp_model_path, model_name):
    hyperparameters = {model_name: {"epochs": ag.Int(1, 3)}}
    num_trials = 2
    # mock the default hps factory to prevent preset hyperparameter configurations from
    # creeping into the test case
    with mock.patch("autogluon.timeseries.models.presets.get_default_hps") as default_hps_mock:
        default_hps_mock.return_value = defaultdict(dict)
        learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric="MAPE")
        learner.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hyperparameters,
            val_data=DUMMY_TS_DATAFRAME,
            hyperparameter_tune_kwargs={
                "searcher": "random",
                "scheduler": "local",
                "num_trials": num_trials,
            },
        )

        leaderboard = learner.leaderboard()

    assert len(leaderboard) == num_trials + 1  # include ensemble

    hpo_results_for_model = learner.load_trainer().hpo_results[model_name]
    config_history = [result["hyperparameters"] for result in hpo_results_for_model.values()]
    assert len(config_history) == 2
    assert all(1 <= config["epochs"] <= 3 for config in config_history)


@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    [
        ({DeepARModel: {"epochs": 1}}, 1),
        (
            {
                ETSModel: {},
                DeepARModel: {"epochs": 1},
            },
            2,
        ),
    ],
)
def test_given_hyperparameters_and_custom_models_when_learner_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric=eval_metric)
    learner.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )
    leaderboard = learner.leaderboard()

    if learner.load_trainer().enable_ensemble and len(hyperparameters) > 1:
        expected_board_length += 1

    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_hyperparameters_when_learner_called_and_loaded_back_then_all_models_can_predict(
    temp_model_path, hyperparameters, expected_board_length
):
    learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric="MAPE", prediction_length=2)
    learner.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )
    learner.save()
    del learner

    loaded_learner = TimeSeriesLearner.load(temp_model_path)

    for model_name in loaded_learner.load_trainer().get_model_names():
        predictions = loaded_learner.predict(DUMMY_TS_DATAFRAME, model=model_name)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.item_ids
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


def test_when_static_features_in_tuning_data_are_missing_then_exception_is_raised(temp_model_path):
    train_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2"])
    )
    val_data = get_data_frame_with_variable_lengths({"B": 25, "A": 20}, static_features=None)
    learner = TimeSeriesLearner(path_context=temp_model_path)
    with pytest.raises(ValueError, match="Provided tuning_data has no static_features,"):
        learner._preprocess_static_features(train_data=train_data, val_data=val_data)


def test_when_static_features_columns_in_tuning_data_are_missing_then_exception_is_raised(temp_model_path):
    train_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2"])
    )
    val_data = get_data_frame_with_variable_lengths(
        {"B": 25, "A": 20}, static_features=get_static_features(["B", "A"], feature_names=["f1"])
    )
    learner = TimeSeriesLearner(path_context=temp_model_path)
    with pytest.raises(ValueError, match="are missing in tuning_data.static_features but were present"):
        learner._preprocess_static_features(train_data=train_data, val_data=val_data)


def test_when_train_data_has_no_static_features_but_val_data_has_static_features_then_val_data_features_get_removed(
    temp_model_path,
):
    train_data = get_data_frame_with_variable_lengths({"B": 25, "A": 20}, static_features=None)
    val_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2"])
    )
    learner = TimeSeriesLearner(path_context=temp_model_path)
    _, val_data_processed = learner._preprocess_static_features(train_data=train_data, val_data=val_data)
    assert val_data.static_features is not None
    assert val_data_processed.static_features is None


def test_when_train_data_static_features_are_subset_of_val_data_static_features_then_columns_are_correct(
    temp_model_path,
):
    train_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2"])
    )
    val_data = get_data_frame_with_variable_lengths(
        {"B": 25, "A": 20}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2", "f3"])
    )
    learner = TimeSeriesLearner(path_context=temp_model_path)
    train_data_processed, val_data_processed = learner._preprocess_static_features(
        train_data=train_data, val_data=val_data
    )
    assert sorted(val_data.static_features.columns) == ["f1", "f2", "f3"]
    for data in [val_data_processed, train_data, train_data_processed]:
        assert sorted(data.static_features.columns) == ["f1", "f2"]


def test_when_static_features_are_preprocessed_then_dtypes_are_correct(temp_model_path):
    train_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2", "f3"])
    )
    learner = TimeSeriesLearner(path_context=temp_model_path)
    train_data_processed, _ = learner._preprocess_static_features(train_data=train_data, val_data=None)
    assert train_data_processed.static_features["f1"].dtype == np.float64
    assert train_data_processed.static_features["f2"].dtype == "category"
    assert train_data_processed.static_features["f3"].dtype == np.float64


def test_when_train_data_has_static_feat_but_pred_data_has_no_static_feat_then_exception_is_raised(temp_model_path):
    train_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2"])
    )
    pred_data = get_data_frame_with_variable_lengths({"B": 20, "A": 15}, static_features=None)
    learner = TimeSeriesLearner(path_context=temp_model_path)
    learner.fit(train_data=train_data, hyperparameters={"ETS": {"maxiter": 1}})
    with pytest.raises(ValueError, match="Provided data has no static_features"):
        learner.predict(pred_data)


ITEM_ID_TO_LENGTH = {"B": 15, "A": 23, "C": 17}
HYPERPARAMETERS_DUMMY = {"Naive": {}}


def test_given_expected_known_covariates_missing_from_train_data_when_learner_fits_then_exception_is_raised(
    temp_model_path,
):
    learner = TimeSeriesLearner(path_context=temp_model_path, known_covariates_names=["Y", "Z", "X"])
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["X", "Z"])
    with pytest.raises(ValueError, match="\\['Y'\\] provided as known_covariates_names are missing from train_data"):
        learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)


def test_given_extra_covariates_are_present_in_dataframe_when_learner_fits_then_they_are_ignored(temp_model_path):
    learner = TimeSeriesLearner(path_context=temp_model_path, known_covariates_names=["Y", "X"])
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["X", "Y", "Z"])
    with mock.patch("autogluon.timeseries.trainer.auto_trainer.AutoTimeSeriesTrainer.fit") as mock_fit:
        learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)
        passed_train = mock_fit.call_args[1]["train_data"]
        assert len(passed_train.columns.symmetric_difference(["Y", "X", "target"])) == 0


def test_given_known_covariates_have_non_numeric_dtypes_when_learner_fits_then_exception_is_raised(temp_model_path):
    learner = TimeSeriesLearner(path_context=temp_model_path, known_covariates_names=["Y", "Z", "X"])
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["X", "Z", "Y"])
    train_data["Y"] = np.random.choice(["foo", "bar", "baz"], size=len(train_data)).astype("O")
    with pytest.raises(ValueError, match="must all have numeric \(float or int\) dtypes"):
        learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)


def test_given_expected_known_covariates_missing_from_data_when_learner_predicts_then_exception_is_raised(
    temp_model_path,
):
    prediction_length = 5
    learner = TimeSeriesLearner(
        path_context=temp_model_path, known_covariates_names=["X", "Y"], prediction_length=prediction_length
    )
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["Y", "X"])
    learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)

    pred_data = train_data.slice_by_timestep(None, -prediction_length)
    known_covariates = train_data.slice_by_timestep(-prediction_length, None).drop("target", axis=1)
    known_covariates.drop("X", axis=1, inplace=True)
    with pytest.raises(
        ValueError, match="\\['X'\\] provided as known_covariates_names are missing from known_covariates."
    ):
        learner.predict(data=pred_data, known_covariates=known_covariates)


def test_given_extra_covariates_are_present_in_dataframe_when_learner_predicts_then_they_are_ignored(temp_model_path):
    prediction_length = 5
    learner = TimeSeriesLearner(
        path_context=temp_model_path, known_covariates_names=["Y", "X"], prediction_length=prediction_length
    )
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["Y", "X"])
    learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)

    data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["Z", "Y", "X"])
    pred_data = data.slice_by_timestep(None, -prediction_length)
    known_covariates = data.slice_by_timestep(-prediction_length, None).drop("target", axis=1)
    with mock.patch("autogluon.timeseries.trainer.auto_trainer.AutoTimeSeriesTrainer.predict") as mock_predict:
        learner.predict(data=pred_data, known_covariates=known_covariates)
        passed_data = mock_predict.call_args[1]["data"]
        passed_known_covariates = mock_predict.call_args[1]["known_covariates"]
        assert len(passed_data.columns.symmetric_difference(["Y", "X", "target"])) == 0
        assert len(passed_known_covariates.columns.symmetric_difference(["Y", "X"])) == 0


@pytest.mark.parametrize("prediction_length", [5, 2])
def test_given_extra_items_and_timestamps_are_present_in_dataframe_when_learner_predicts_then_correct_subset_is_selected(
    temp_model_path,
    prediction_length,
):
    learner = TimeSeriesLearner(
        path_context=temp_model_path, known_covariates_names=["Y", "X"], prediction_length=prediction_length
    )
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["Y", "X"])
    learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)

    data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["Y", "X"])
    pred_data = data.slice_by_timestep(None, -prediction_length)

    # known_covariates includes additional item_ids and additional timestamps
    extended_item_id_to_length = {"D": 25, "E": 37, "F": 14}
    for item_id, length in ITEM_ID_TO_LENGTH.items():
        extended_item_id_to_length[item_id] = length + 7
    extended_data = get_data_frame_with_variable_lengths(extended_item_id_to_length, known_covariates_names=["Y", "X"])
    known_covariates = extended_data.drop("target", axis=1)

    with mock.patch("autogluon.timeseries.trainer.auto_trainer.AutoTimeSeriesTrainer.predict") as mock_predict:
        learner.predict(data=pred_data, known_covariates=known_covariates)
        passed_known_covariates = mock_predict.call_args[1]["known_covariates"]
        assert len(passed_known_covariates.item_ids.symmetric_difference(pred_data.item_ids)) == 0
        for item_id in pred_data.item_ids:
            expected_forecast_timestamps = get_forecast_horizon_index_single_time_series(
                pred_data.loc[item_id].index, freq=pred_data.freq, prediction_length=prediction_length
            )
            assert (passed_known_covariates.loc[item_id].index == expected_forecast_timestamps).all()


def test_given_ignore_index_is_true_and_covariates_too_short_when_learner_predicts_then_exception_is_raised(
    temp_model_path,
):
    prediction_length = 5
    learner = TimeSeriesLearner(
        path_context=temp_model_path,
        known_covariates_names=["Y", "X"],
        prediction_length=prediction_length,
        ignore_time_index=True,
    )
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["Y", "X"])
    learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)
    short_known_covariates = get_data_frame_with_variable_lengths(
        {k: 4 for k in ITEM_ID_TO_LENGTH.keys()}, known_covariates_names=["X", "Y"]
    )
    with pytest.raises(ValueError, match=f"should include the values for prediction_length={prediction_length}"):
        learner.predict(train_data, known_covariates=short_known_covariates)


def test_when_ignore_index_is_true_and_known_covariates_available_then_learner_can_predict(temp_model_path):
    prediction_length = 5
    learner = TimeSeriesLearner(
        path_context=temp_model_path,
        known_covariates_names=["Y", "X"],
        prediction_length=prediction_length,
        ignore_time_index=True,
    )
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["Y", "X"])
    learner.fit(train_data=train_data, hyperparameters={"DeepAR": {"epochs": 1, "num_batches_per_epoch": 1}})
    known_covariates = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, known_covariates_names=["X", "Y"])
    preds = learner.predict(train_data, known_covariates=known_covariates)
    assert preds.item_ids.equals(train_data.item_ids)
