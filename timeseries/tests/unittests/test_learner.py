"""Unit tests for learners"""

import shutil
import sys
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from autogluon.common import space
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.learner import TimeSeriesLearner
from autogluon.timeseries.models import DeepARModel, ETSModel
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_single_time_series

from .common import (
    DUMMY_TS_DATAFRAME,
    get_data_frame_with_covariates,
    get_data_frame_with_variable_lengths,
    get_static_features,
)

TEST_HYPERPARAMETER_SETTINGS = [
    {"SimpleFeedForward": {"max_epochs": 1, "num_batches_per_epoch": 1}},
    {"DeepAR": {"max_epochs": 1, "num_batches_per_epoch": 1}, "Naive": {}},
]
TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS = [1, 2]


@pytest.fixture(scope="module")
def trained_learners():
    learners = {}
    model_paths = []
    for hp in TEST_HYPERPARAMETER_SETTINGS:
        temp_model_path = tempfile.mkdtemp()
        learner = TimeSeriesLearner(
            path_context=temp_model_path,
            eval_metric="MASE",
            prediction_length=3,
        )
        learner.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hp,
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


@pytest.mark.skipif(sys.platform.startswith("win"), reason="HPO tests lead to known issues in Windows platform tests")
@pytest.mark.parametrize("model_name", ["DeepAR", "SimpleFeedForward"])
def test_given_hyperparameters_with_spaces_when_learner_called_then_hpo_is_performed(temp_model_path, model_name):
    hyperparameters = {model_name: {"max_epochs": space.Int(1, 3)}}
    num_trials = 2

    learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric="MAPE")
    learner.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
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
    assert all(1 <= config["max_epochs"] <= 3 for config in config_history)


@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    [
        ({DeepARModel: {"max_epochs": 1}}, 1),
        (
            {
                ETSModel: {},
                DeepARModel: {"max_epochs": 1},
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
    with pytest.raises(ValueError, match="Provided tuning_data must contain static_features"):
        learner.fit(train_data=train_data, hyperparameters={}, val_data=val_data)


def test_when_static_features_columns_in_tuning_data_are_missing_then_exception_is_raised(temp_model_path):
    train_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2"])
    )
    val_data = get_data_frame_with_variable_lengths(
        {"B": 25, "A": 20}, static_features=get_static_features(["B", "A"], feature_names=["f1"])
    )
    learner = TimeSeriesLearner(path_context=temp_model_path)
    with pytest.raises(KeyError, match="required columns are missing from the provided"):
        learner.fit(train_data=train_data, hyperparameters={}, val_data=val_data)


def test_when_train_data_has_no_static_features_but_val_data_has_static_features_then_val_data_features_get_removed(
    temp_model_path,
):
    train_data = get_data_frame_with_variable_lengths({"B": 25, "A": 20}, static_features=None)
    val_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2"])
    )
    learner = TimeSeriesLearner(path_context=temp_model_path)
    learner.feature_generator.fit(train_data)
    val_data_processed = learner.feature_generator.transform(val_data)
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
    learner.feature_generator.fit(train_data)
    train_data_processed = learner.feature_generator.transform(train_data)
    val_data_processed = learner.feature_generator.transform(val_data)
    assert sorted(val_data.static_features.columns) == ["f1", "f2", "f3"]
    for data in [val_data_processed, train_data, train_data_processed]:
        assert sorted(data.static_features.columns) == ["f1", "f2"]


def test_when_static_features_are_preprocessed_then_dtypes_are_correct(temp_model_path):
    train_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2", "f3"])
    )
    learner = TimeSeriesLearner(path_context=temp_model_path)
    train_data_processed = learner.feature_generator.fit_transform(train_data)
    assert train_data_processed.static_features["f1"].dtype == np.float32
    assert train_data_processed.static_features["f2"].dtype == "category"
    assert train_data_processed.static_features["f3"].dtype == np.float32


def test_when_train_data_has_static_feat_but_pred_data_has_no_static_feat_then_exception_is_raised(temp_model_path):
    train_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=get_static_features(["B", "A"], feature_names=["f1", "f2"])
    )
    pred_data = get_data_frame_with_variable_lengths({"B": 20, "A": 15}, static_features=None)
    learner = TimeSeriesLearner(path_context=temp_model_path)
    learner.fit(train_data=train_data, hyperparameters={"ETS": {"maxiter": 1}})
    with pytest.raises(ValueError, match="Provided data must contain static_features"):
        learner.predict(pred_data)


ITEM_ID_TO_LENGTH = {"B": 15, "A": 23, "C": 17}
HYPERPARAMETERS_DUMMY = {"Naive": {}}


def test_given_expected_known_covariates_missing_from_train_data_when_learner_fits_then_exception_is_raised(
    temp_model_path,
):
    learner = TimeSeriesLearner(path_context=temp_model_path, known_covariates_names=["Y", "Z", "X"])
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, covariates_names=["X", "Z"])
    with pytest.raises(ValueError, match="columns are missing from train_data: \\['Y'\\]"):
        learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)


def test_given_expected_known_covariates_missing_from_data_when_learner_predicts_then_exception_is_raised(
    temp_model_path,
):
    prediction_length = 5
    learner = TimeSeriesLearner(
        path_context=temp_model_path,
        known_covariates_names=["X", "Y"],
        prediction_length=prediction_length,
    )
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, covariates_names=["Y", "X"])
    learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)

    pred_data = train_data.slice_by_timestep(None, -prediction_length)
    known_covariates = train_data.slice_by_timestep(-prediction_length, None).drop("target", axis=1)
    known_covariates.drop("X", axis=1, inplace=True)
    with pytest.raises(ValueError, match="columns are missing from known_covariates: \\['X'\\]"):
        learner.predict(data=pred_data, known_covariates=known_covariates)


def test_given_extra_covariates_are_present_in_dataframe_when_learner_predicts_then_they_are_ignored(temp_model_path):
    prediction_length = 5
    learner = TimeSeriesLearner(
        path_context=temp_model_path, known_covariates_names=["Y", "X"], prediction_length=prediction_length
    )
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, covariates_names=["Y", "X"])
    learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)

    data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, covariates_names=["Y", "X", "Z"])
    pred_data = data.slice_by_timestep(None, -prediction_length)
    known_covariates = data.slice_by_timestep(-prediction_length, None).drop("target", axis=1)
    with mock.patch("autogluon.timeseries.trainer.TimeSeriesTrainer.predict") as mock_predict:
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
    train_data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, covariates_names=["Y", "X"])
    learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)

    data = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH, covariates_names=["Y", "X"])
    pred_data = data.slice_by_timestep(None, -prediction_length)

    # known_covariates includes additional item_ids and additional timestamps
    extended_item_id_to_length = {"D": 25, "E": 37, "F": 14}
    for item_id, length in ITEM_ID_TO_LENGTH.items():
        extended_item_id_to_length[item_id] = length + 7
    extended_data = get_data_frame_with_variable_lengths(extended_item_id_to_length, covariates_names=["Y", "X"])
    known_covariates = extended_data.drop("target", axis=1)

    with mock.patch("autogluon.timeseries.trainer.TimeSeriesTrainer.predict") as mock_predict:
        learner.predict(data=pred_data, known_covariates=known_covariates)
        passed_known_covariates = mock_predict.call_args[1]["known_covariates"]
        assert len(passed_known_covariates.item_ids.symmetric_difference(pred_data.item_ids)) == 0
        for item_id in pred_data.item_ids:
            expected_forecast_timestamps = get_forecast_horizon_index_single_time_series(
                pred_data.loc[item_id].index, freq=pred_data.freq, prediction_length=prediction_length
            )
            assert (passed_known_covariates.loc[item_id].index == expected_forecast_timestamps).all()


@pytest.mark.parametrize("pred_data_present", [True, False])
@pytest.mark.parametrize("static_features_present", [True, False])
@pytest.mark.parametrize("known_covariates_present", [True, False])
@pytest.mark.parametrize("past_covariates_present", [True, False])
def test_when_train_data_has_static_or_dynamic_feat_then_leaderboard_works(
    temp_model_path, pred_data_present, static_features_present, known_covariates_present, past_covariates_present
):
    if static_features_present:
        static_features = get_static_features(["B", "A"], feature_names=["f1", "f2"])
    else:
        static_features = None

    covariates_names = []
    if known_covariates_present:
        known_covariates_names = ["X", "Y"]
        covariates_names.extend(known_covariates_names)
    else:
        known_covariates_names = None

    if past_covariates_present:
        covariates_names.extend(["A", "B", "C"])

    train_data = get_data_frame_with_variable_lengths(
        {"B": 20, "A": 15}, static_features=static_features, covariates_names=covariates_names
    )

    if pred_data_present:
        pred_data = get_data_frame_with_variable_lengths(
            {"B": 20, "A": 15}, static_features=static_features, covariates_names=covariates_names
        )
    else:
        pred_data = None

    learner = TimeSeriesLearner(path_context=temp_model_path)
    learner.fit(train_data=train_data, hyperparameters=HYPERPARAMETERS_DUMMY)
    leaderboard = learner.leaderboard(data=pred_data)
    assert len(leaderboard) > 0
    assert ("score_test" in leaderboard.columns) == pred_data_present


def test_when_features_are_all_nan_and_learner_is_loaded_then_mode_or_median_are_imputed(temp_model_path):
    covariates_cat = ["known_cat", "past_cat"]
    covariates_real = ["known_real", "past_real"]
    data = get_data_frame_with_covariates(
        covariates_cat=covariates_cat,
        covariates_real=covariates_real,
        static_features_cat=["static_cat"],
        static_features_real=["static_real"],
    )
    known_covariates_names = ["known_cat", "known_real"]
    prediction_length = 3
    learner = TimeSeriesLearner(
        path_context=temp_model_path,
        known_covariates_names=known_covariates_names,
        prediction_length=prediction_length,
    )
    learner.fit(data, hyperparameters={"Naive": {}})
    data_transformed = learner.feature_generator.transform(data)
    learner.save()
    del learner

    loaded_learner = TimeSeriesLearner.load(temp_model_path)
    data_with_nan = data.copy()
    for col in data_with_nan.columns:
        if col != "target":
            data_with_nan[col] = float("nan")
    for col in data_with_nan.static_features.columns:
        data_with_nan.static_features[col] = float("nan")
    data_with_nan, known_covariates_with_nan = data_with_nan.get_model_inputs_for_scoring(
        prediction_length, known_covariates_names
    )
    with mock.patch("autogluon.timeseries.trainer.TimeSeriesTrainer.predict") as trainer_predict:
        loaded_learner.predict(data_with_nan, known_covariates=known_covariates_with_nan)
        trainer_predict_call_args = trainer_predict.call_args[1]
        imputed_data = trainer_predict_call_args["data"]
        imputed_known_covariates = trainer_predict_call_args["known_covariates"]
        imputed_static = imputed_data.static_features

    def get_mode(series: pd.Series):
        # series.mode() can result in ties. We copy tiebreaking logic from CategoryFeatureGenerator
        return series.value_counts().sort_values().index[-1]

    for col in covariates_cat:
        column_mode_train = get_mode(data_transformed[col])
        assert (imputed_data[col] == column_mode_train).all()
        if col in known_covariates_names:
            assert (imputed_known_covariates[col] == column_mode_train).all()

    for col in covariates_real:
        column_median_train = data_transformed[col].median()
        assert np.allclose(imputed_data[col], column_median_train)
        if col in known_covariates_names:
            assert np.allclose(imputed_known_covariates[col], column_median_train)

    assert (imputed_static["static_cat"] == get_mode(data_transformed.static_features["static_cat"])).all()
    assert np.allclose(imputed_static["static_real"], data_transformed.static_features["static_real"].median())
