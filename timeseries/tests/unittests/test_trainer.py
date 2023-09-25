"""Unit tests for trainers"""
import copy
import shutil
import tempfile
from collections import defaultdict
from unittest import mock

import numpy as np
import pytest

import autogluon.core as ag
from autogluon.common import space
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models import DeepARModel, ETSModel
from autogluon.timeseries.models.ensemble.greedy_ensemble import TimeSeriesGreedyEnsemble
from autogluon.timeseries.trainer.auto_trainer import AutoTimeSeriesTrainer

from .common import DATAFRAME_WITH_COVARIATES, DUMMY_TS_DATAFRAME, dict_equal_primitive, get_data_frame_with_item_index

DUMMY_TRAINER_HYPERPARAMETERS = {"SimpleFeedForward": {"epochs": 1}}
TEST_HYPERPARAMETER_SETTINGS = [
    {"SimpleFeedForward": {"epochs": 1}},
    {"DeepAR": {"epochs": 1}, "ETS": {}},
]
TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS = [1, 2]


@pytest.fixture(scope="module")
def trained_trainers():
    trainers = {}
    model_paths = []
    for hp in TEST_HYPERPARAMETER_SETTINGS:
        temp_model_path = tempfile.mkdtemp()
        trainer = AutoTimeSeriesTrainer(
            path=temp_model_path,
            eval_metric="MAPE",
            prediction_length=3,
        )
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hp,
        )
        trainers[repr(hp)] = trainer
        model_paths.append(temp_model_path)

    yield trainers

    for td in model_paths:
        shutil.rmtree(td)


def test_trainer_can_be_initialized(temp_model_path):
    model = AutoTimeSeriesTrainer(path=temp_model_path, prediction_length=24)
    assert isinstance(model, AutoTimeSeriesTrainer)


# smoke test for the short 'happy path'
@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_when_trainer_called_then_training_is_performed(trained_trainers, hyperparameters):
    assert trained_trainers[repr(hyperparameters)].get_model_names()


@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_hyperparameters_when_trainer_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric=eval_metric)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
    )
    leaderboard = trainer.leaderboard()

    if len(hyperparameters) > 1:
        expected_board_length += int(trainer.enable_ensemble)
    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_test_data_when_trainer_called_then_leaderboard_is_correct(
    trained_trainers, hyperparameters, expected_board_length
):
    trainer = trained_trainers[repr(hyperparameters)]
    test_data = get_data_frame_with_item_index(["A", "B", "C"])

    leaderboard = trainer.leaderboard(test_data)

    if len(hyperparameters) > 1:
        expected_board_length += int(trainer.enable_ensemble)

    assert len(leaderboard) == expected_board_length
    assert not np.any(np.isnan(leaderboard["score_test"]))
    assert np.all(leaderboard["score_test"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_hyperparameters_when_trainer_called_then_model_can_predict(
    trained_trainers, hyperparameters, expected_board_length
):
    trainer = trained_trainers[repr(hyperparameters)]
    predictions = trainer.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"DeepAR": {"epochs": 4}, "SimpleFeedForward": {"epochs": 1}},
        {
            "SimpleFeedForward": {"context_length": 44, "epochs": 2},
            "DeepAR": {"epochs": 3},
        },
    ],
)
def test_given_hyperparameters_when_trainer_model_templates_called_then_hyperparameters_set_correctly(
    temp_model_path, hyperparameters
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE")
    models = trainer.construct_model_templates(
        hyperparameters=hyperparameters,
    )

    for model in models:
        for k, v in hyperparameters[model.name].items():
            assert model._user_params[k] == v


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"DeepAR": {"epochs": 1}, "SimpleFeedForward": {"epochs": 1}},
        {
            "SimpleFeedForward": {"context_length": 44, "epochs": 1},
            "DeepAR": {"epochs": 1},
        },
    ],
)
def test_given_hyperparameters_when_trainer_fit_then_freq_set_correctly(temp_model_path, hyperparameters):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE")
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
    )

    for model_name in trainer.get_model_names():
        model = trainer.load_model(model_name)
        assert model.freq == DUMMY_TS_DATAFRAME.freq


@pytest.mark.parametrize("model_name", ["DeepAR", "SimpleFeedForward"])
def test_given_hyperparameters_with_spaces_when_trainer_called_then_hpo_is_performed(temp_model_path, model_name):
    hyperparameters = {model_name: {"epochs": space.Int(1, 4)}}
    # mock the default hps factory to prevent preset hyperparameter configurations from
    # creeping into the test case
    with mock.patch("autogluon.timeseries.models.presets.get_default_hps") as default_hps_mock:
        default_hps_mock.return_value = defaultdict(dict)
        trainer = AutoTimeSeriesTrainer(path=temp_model_path)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs={
                "num_trials": 2,
                "searcher": "random",
                "scheduler": "local",
            },
        )
        leaderboard = trainer.leaderboard()

    assert len(leaderboard) == 2 + 1  # include ensemble

    hpo_results_first_model = next(iter(trainer.hpo_results.values()))
    config_history = [result["hyperparameters"] for result in hpo_results_first_model.values()]
    assert len(config_history) == 2
    assert all(1 <= config["epochs"] <= 4 for config in config_history)


@pytest.mark.parametrize(
    "hyperparameters, expected_model_names",
    [
        ({"Naive": [{}, {}, {"ag_args": {"name_suffix": "_extra"}}]}, ["Naive", "Naive_2", "Naive_extra"]),
        ({"Naive": [{"ag_args": {"name": "CustomNaive"}}], "SeasonalNaive": {}}, ["CustomNaive", "SeasonalNaive"]),
    ],
)
def test_given_hyperparameters_with_lists_when_trainer_called_then_multiple_models_are_trained(
    temp_model_path, hyperparameters, expected_model_names
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, enable_ensemble=False)
    trainer.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters=hyperparameters)
    leaderboard = trainer.leaderboard()
    assert len(leaderboard) == len(expected_model_names)
    assert all(name in leaderboard["model"].values for name in expected_model_names)


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
def test_given_hyperparameters_and_custom_models_when_trainer_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric=eval_metric)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
    )
    leaderboard = trainer.leaderboard()

    if len(hyperparameters) > 1:  # account for ensemble
        expected_board_length += int(trainer.enable_ensemble)
    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameter_list, expected_number_of_unique_names, expected_suffixes",
    [
        (
            [
                {DeepARModel: {"epochs": 1}},
                {DeepARModel: {"epochs": 1}},
            ],
            3,
            ["AR_2"],
        ),
        (
            [
                {DeepARModel: {"epochs": 1}, "ETS": {}},
                {DeepARModel: {"epochs": 1}},
                {DeepARModel: {"epochs": 1}},
            ],
            7,
            ["AR_2", "AR_3", "Ensemble_2", "Ensemble_3"],
        ),
        (
            [
                {DeepARModel: {"epochs": 1}, "DeepAR": {"epochs": 1}, "ETS": {}},
                {DeepARModel: {"epochs": 1}},
            ],
            6,
            ["AR_2", "AR_3", "Ensemble_2"],
        ),
    ],
)
def test_given_repeating_model_when_trainer_called_incrementally_then_name_collisions_are_prevented(
    temp_model_path,
    hyperparameter_list,
    expected_number_of_unique_names,
    expected_suffixes,
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path)

    # incrementally train with new hyperparameters
    for hp in hyperparameter_list:
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hp,
        )

    model_names = trainer.get_model_names()

    # account for the ensemble if it should be fitted, and drop ensemble names
    assert len(model_names) == expected_number_of_unique_names
    for suffix in expected_suffixes:
        assert any(name.endswith(suffix) for name in model_names)


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"DeepAR": {"epochs": 1}, "SimpleFeedForward": {"epochs": 1}},
        {
            "SimpleFeedForward": {"context_length": 44, "epochs": 1},
            "DeepAR": {"epochs": 1},
        },
    ],
)
@pytest.mark.parametrize("low_memory", [True, False])
def test_when_trainer_fit_and_deleted_models_load_back_correctly_and_can_predict(
    temp_model_path, hyperparameters, low_memory
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE", prediction_length=2)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
    )
    model_names = copy.copy(trainer.get_model_names())
    trainer.save()
    del trainer

    loaded_trainer = AutoTimeSeriesTrainer.load(path=temp_model_path)

    for m in model_names:
        loaded_model = loaded_trainer.load_model(m)
        if isinstance(loaded_model, TimeSeriesGreedyEnsemble):
            continue

        predictions = loaded_model.predict(DUMMY_TS_DATAFRAME)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.item_ids
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


def test_when_trainer_fit_and_deleted_then_oof_predictions_can_be_loaded(temp_model_path):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE", prediction_length=2)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={
            "Naive": {},
            "ETS": {},
            "AutoETS": {"n_jobs": 1},
            "DirectTabular": {"tabular_hyperparameters": {"GBM": {}}},
            "DeepAR": {"epochs": 1, "num_batches_per_epoch": 1},
        },
    )
    model_names = copy.copy(trainer.get_model_names())
    trainer.save()
    del trainer

    loaded_trainer = AutoTimeSeriesTrainer.load(path=temp_model_path)

    oof_data = loaded_trainer._get_ensemble_oof_data(DUMMY_TS_DATAFRAME, val_data=None)
    for m in model_names:
        if "WeightedEnsemble" not in m:
            oof_predictions = loaded_trainer._get_model_oof_predictions(m)
            for window_idx, oof_pred in enumerate(oof_predictions):
                assert isinstance(oof_pred, TimeSeriesDataFrame)
                loaded_trainer._score_with_predictions(oof_data[window_idx], oof_pred)


def test_when_known_covariates_present_then_all_ensemble_base_models_can_predict(temp_model_path):
    df = DATAFRAME_WITH_COVARIATES.copy()
    prediction_length = 2
    df_train = df.slice_by_timestep(None, -prediction_length)
    df_future = df.slice_by_timestep(-prediction_length, None)
    known_covariates = df_future.drop("target", axis=1)

    trainer = AutoTimeSeriesTrainer(
        path=temp_model_path, prediction_length=prediction_length, enable_ensemble=False, cache_predictions=False
    )
    trainer.fit(df_train, hyperparameters={"ETS": {"maxiter": 1}, "DeepAR": {"epochs": 1, "num_batches_per_epoch": 1}})

    # Manually add ensemble to ensure that both models have non-zero weight
    ensemble = TimeSeriesGreedyEnsemble(name="WeightedEnsemble")
    ensemble.model_to_weight = {"DeepAR": 0.5, "ETS": 0.5}
    trainer._add_model(model=ensemble, base_models=["DeepAR", "ETS"])
    trainer.save_model(model=ensemble)
    with mock.patch(
        "autogluon.timeseries.models.ensemble.greedy_ensemble.TimeSeriesGreedyEnsemble.predict"
    ) as mock_predict:
        trainer.predict(df_train, model="WeightedEnsemble", known_covariates=known_covariates)
        inputs = mock_predict.call_args[0][0]
        # No models failed during prediction
        assert inputs["DeepAR"] is not None
        assert inputs["ETS"] is not None


@pytest.fixture(scope="module")
def trained_and_refit_trainers():
    def fit_trainer():
        temp_model_path = tempfile.mkdtemp()
        trainer = AutoTimeSeriesTrainer(
            path=temp_model_path,
            prediction_length=3,
        )
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={
                "Naive": {},
                "ETS": {"maxiter": 1},
                "DeepAR": {"epochs": 1, "num_batches_per_epoch": 1},
                "DirectTabular": {"tabular_hyperparameters": {"GBM": {}}},
                "RecursiveTabular": {},
            },
        )
        return trainer

    trainer = fit_trainer()
    refit_trainer = fit_trainer()
    refit_trainer.refit_full("all")

    yield trainer, refit_trainer


def test_when_refit_full_called_then_all_models_are_retrained(trained_and_refit_trainers):
    trainer, refit_trainer = trained_and_refit_trainers
    leaderboard_initial = trainer.leaderboard()
    leaderboard_refit = refit_trainer.leaderboard()

    expected_refit_full_dict = {name: name + ag.constants.REFIT_FULL_SUFFIX for name in trainer.get_model_names()}
    assert dict_equal_primitive(refit_trainer.model_full_dict, expected_refit_full_dict)
    assert len(leaderboard_refit) == len(leaderboard_initial) + len(expected_refit_full_dict)


def test_when_refit_full_called_then_all_models_can_predict(trained_and_refit_trainers):
    _, refit_trainer = trained_and_refit_trainers
    for model in refit_trainer.get_model_names():
        preds = refit_trainer.predict(DUMMY_TS_DATAFRAME, model=model)
        assert isinstance(preds, TimeSeriesDataFrame)
        assert len(preds) == DUMMY_TS_DATAFRAME.num_items * refit_trainer.prediction_length


def test_when_refit_full_called_with_model_name_then_single_model_is_updated(temp_model_path):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path)
    trainer.fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={
            "DeepAR": {"epochs": 1, "num_batches_per_epoch": 1},
            "SimpleFeedForward": {"epochs": 1, "num_batches_per_epoch": 1},
        },
    )
    model_full_dict = trainer.refit_full("DeepAR")
    assert list(model_full_dict.values()) == ["DeepAR_FULL"]


@pytest.mark.parametrize(
    "hyperparameters, expected_model_names",
    [
        ({"Naive": {}, "SeasonalNaiveModel": {}}, ["Naive", "SeasonalNaive"]),
        ({"Naive": {}, "NaiveModel": {}}, ["Naive", "Naive_2"]),
    ],
)
def test_when_some_models_have_incorrect_suffix_then_correct_model_are_trained(
    temp_model_path, hyperparameters, expected_model_names
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, enable_ensemble=False)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters=hyperparameters)
    leaderboard = trainer.leaderboard()
    assert sorted(leaderboard["model"].values) == expected_model_names


@pytest.mark.parametrize("excluded_model_types", [["DeepAR"], ["DeepARModel"]])
def test_when_excluded_model_names_provided_then_excluded_models_are_not_trained(
    temp_model_path, excluded_model_types
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path)
    trainer.fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={
            "DeepAR": {"epochs": 1, "num_batches_per_epoch": 1},
            "SimpleFeedForward": {"epochs": 1, "num_batches_per_epoch": 1},
        },
        excluded_model_types=excluded_model_types,
    )
    leaderboard = trainer.leaderboard()
    assert leaderboard["model"].values == ["SimpleFeedForward"]


@pytest.mark.parametrize("model_names", [["WeightedEnsemble"], ["WeightedEnsemble", "DeepAR"], ["DeepAR"]])
def test_when_get_model_pred_dict_called_then_it_contains_all_required_keys(trained_trainers, model_names):
    trainer = trained_trainers[repr(TEST_HYPERPARAMETER_SETTINGS[1])]
    model_pred_dict = trainer.get_model_pred_dict(model_names=model_names, data=DUMMY_TS_DATAFRAME)
    assert sorted(model_pred_dict.keys()) == sorted(model_names)


@pytest.mark.parametrize("model_names", [["WeightedEnsemble"], ["WeightedEnsemble", "DeepAR"], ["DeepAR"]])
def test_when_get_model_pred_dict_called_then_pred_time_dict_contains_all_required_keys(trained_trainers, model_names):
    trainer = trained_trainers[repr(TEST_HYPERPARAMETER_SETTINGS[1])]
    model_pred_dict, pred_time_dict = trainer.get_model_pred_dict(
        model_names=model_names, data=DUMMY_TS_DATAFRAME, record_pred_time=True
    )
    assert sorted(pred_time_dict.keys()) == sorted(model_names)


def test_given_dfs_are_identical_then_identical_hash_is_computed(temp_model_path):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path)
    df = DATAFRAME_WITH_COVARIATES
    df_other = DATAFRAME_WITH_COVARIATES.copy()
    df_other = df_other.reindex(reversed(df_other.columns), axis=1)
    assert df is not df_other
    assert trainer._compute_dataset_hash(df) == trainer._compute_dataset_hash(df_other)


def test_given_cache_predictions_is_true_when_calling_get_model_pred_dict_then_predictions_are_cached(temp_model_path):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}, "SeasonalNaive": {}})
    assert not trainer._cached_predictions_path.exists()
    trainer.get_model_pred_dict(trainer.get_model_names(), data=DUMMY_TS_DATAFRAME, record_pred_time=True)

    dataset_hash = trainer._compute_dataset_hash(DUMMY_TS_DATAFRAME)
    model_pred_dict, pred_time_dict = trainer._get_cached_pred_dicts(dataset_hash)
    assert pred_time_dict.keys() == model_pred_dict.keys() == set(trainer.get_model_names())
    assert all(isinstance(v, TimeSeriesDataFrame) for v in model_pred_dict.values())
    assert all(isinstance(v, float) for v in pred_time_dict.values())


def test_given_cache_predictions_is_true_when_predicting_multiple_times_then_cached_predictions_are_updated(
    temp_model_path,
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}, "SeasonalNaive": {}})

    dataset_hash = trainer._compute_dataset_hash(DUMMY_TS_DATAFRAME)

    trainer.predict(DUMMY_TS_DATAFRAME, model="Naive")
    model_pred_dict, pred_time_dict = trainer._get_cached_pred_dicts(dataset_hash)
    assert sorted(model_pred_dict.keys()) == sorted(pred_time_dict.keys()) == ["Naive"]

    trainer.predict(DUMMY_TS_DATAFRAME, model="SeasonalNaive")
    model_pred_dict, pred_time_dict = trainer._get_cached_pred_dicts(dataset_hash)
    assert sorted(model_pred_dict.keys()) == sorted(pred_time_dict.keys()) == ["Naive", "SeasonalNaive"]


def test_given_cache_predictions_is_true_when_predicting_multiple_times_then_cached_predictions_are_used(
    temp_model_path,
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}})
    mock_return_value = "mock_return_value"
    with mock.patch("autogluon.timeseries.models.local.naive.NaiveModel.predict") as naive_predict:
        naive_predict.return_value = mock_return_value
        trainer.predict(DUMMY_TS_DATAFRAME, model="Naive")

    with mock.patch("autogluon.timeseries.models.local.naive.NaiveModel.predict") as naive_predict:
        predictions = trainer.predict(DUMMY_TS_DATAFRAME, model="Naive")
        naive_predict.assert_not_called()
        assert predictions == mock_return_value


def test_given_cache_predictions_is_false_when_calling_get_model_pred_dict_then_predictions_are_not_cached(
    temp_model_path,
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, cache_predictions=False)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters=DUMMY_TRAINER_HYPERPARAMETERS)
    assert not trainer._cached_predictions_path.exists()
    trainer.get_model_pred_dict(trainer.get_model_names(), data=DUMMY_TS_DATAFRAME)
    assert not trainer._cached_predictions_path.exists()
