"""Unit tests for predictors"""
import copy
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from autogluon.common import space
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP
from autogluon.timeseries.models import DeepARModel, SimpleFeedForwardModel
from autogluon.timeseries.predictor import TimeSeriesPredictor

from .common import DUMMY_TS_DATAFRAME, get_data_frame_with_variable_lengths

TEST_HYPERPARAMETER_SETTINGS = [
    {"SimpleFeedForward": {"epochs": 1, "num_batches_per_epoch": 1}},
    {"ETS": {"maxiter": 1}, "SimpleFeedForward": {"epochs": 1, "num_batches_per_epoch": 1}},
]


def test_predictor_can_be_initialized(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    assert isinstance(predictor, TimeSeriesPredictor)


# smoke test for the short 'happy path'
def test_when_predictor_called_then_training_is_performed(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path, eval_metric="MAPE")
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    assert "SimpleFeedForward" in predictor.get_model_names()


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS + ["fast_training"])  # noqa
def test_given_hyperparameters_when_predictor_called_then_model_can_predict(temp_model_path, hyperparameters):
    predictor = TimeSeriesPredictor(path=temp_model_path, eval_metric="MAPE", prediction_length=3)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    predictions = predictor.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS + ["fast_training"])  # noqa
def test_given_different_target_name_when_predictor_called_then_model_can_predict(temp_model_path, hyperparameters):
    df = TimeSeriesDataFrame(copy.copy(DUMMY_TS_DATAFRAME))
    df.rename(columns={"target": "mytarget"}, inplace=True)

    predictor = TimeSeriesPredictor(
        target="mytarget",
        path=temp_model_path,
        eval_metric="MAPE",
        prediction_length=3,
    )
    predictor.fit(
        train_data=df,
        hyperparameters=hyperparameters,
    )
    predictions = predictor.predict(df)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_no_tuning_data_when_predictor_called_then_model_can_predict(temp_model_path, hyperparameters):
    predictor = TimeSeriesPredictor(path=temp_model_path, eval_metric="MAPE", prediction_length=3)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
    )
    predictions = predictor.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_hyperparameters_and_quantiles_when_predictor_called_then_model_can_predict(
    temp_model_path, hyperparameters
):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        prediction_length=3,
        quantile_levels=[0.1, 0.4, 0.9],
    )

    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    predictions = predictor.predict(DUMMY_TS_DATAFRAME)

    assert tuple(predictions.columns) == ("mean", "0.1", "0.4", "0.9")


@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    [
        ({DeepARModel: {"epochs": 1}}, 1),
        (
            {
                DeepARModel: {"epochs": 1},
                SimpleFeedForwardModel: {"epochs": 1},
            },
            2,
        ),
    ],
)
def test_given_hyperparameters_and_custom_models_when_predictor_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    predictor = TimeSeriesPredictor(path=temp_model_path, eval_metric=eval_metric)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    leaderboard = predictor.leaderboard()

    if predictor._trainer.enable_ensemble and len(hyperparameters) > 1:
        expected_board_length += 1

    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_hyperparameters_when_predictor_called_and_loaded_back_then_all_models_can_predict(
    temp_model_path, hyperparameters
):
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=2)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    predictor.save()
    del predictor

    loaded_predictor = TimeSeriesPredictor.load(temp_model_path)

    for model_name in loaded_predictor.get_model_names():
        predictions = loaded_predictor.predict(DUMMY_TS_DATAFRAME, model=model_name)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.item_ids
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("target_column", ["target", "custom"])
@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"ETS": {"maxiter": 1}, "SimpleFeedForward": {"epochs": 1}},
        {"ETS": {"maxiter": 1}, "SimpleFeedForward": {"epochs": space.Int(1, 3)}},
    ],
)
def test_given_hp_spaces_and_custom_target_when_predictor_called_predictor_can_predict(
    temp_model_path, hyperparameters, target_column
):
    df = DUMMY_TS_DATAFRAME.rename(columns={"target": target_column})

    fit_kwargs = dict(
        train_data=df,
        hyperparameters=hyperparameters,
        tuning_data=df,
    )
    init_kwargs = dict(path=temp_model_path, prediction_length=2)
    if target_column != "target":
        init_kwargs.update({"target": target_column})

    for hps in hyperparameters.values():
        if any(isinstance(v, space.Space) for v in hps.values()):
            fit_kwargs.update(
                {
                    "hyperparameter_tune_kwargs": {
                        "scheduler": "local",
                        "searcher": "random",
                        "num_trials": 2,
                    },
                }
            )
            break

    predictor = TimeSeriesPredictor(**init_kwargs)
    predictor.fit(**fit_kwargs)

    assert predictor.get_model_names()

    for model_name in predictor.get_model_names():
        predictions = predictor.predict(df, model=model_name)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.item_ids
        assert all(predicted_item_index == df.item_ids)  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_hyperparameters_when_predictor_called_and_loaded_back_then_loaded_learner_can_predict(
    temp_model_path, hyperparameters
):
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=2)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    predictor.save()
    del predictor

    loaded_predictor = TimeSeriesPredictor.load(temp_model_path)

    predictions = loaded_predictor._learner.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


def test_given_enable_ensemble_true_when_predictor_called_then_ensemble_is_fitted(temp_model_path):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={
            "SimpleFeedForward": {"epochs": 1},
            "DeepAR": {"epochs": 1},
        },
    )
    assert any("ensemble" in n.lower() for n in predictor.get_model_names())


def test_given_enable_ensemble_true_and_only_one_model_when_predictor_called_then_ensemble_is_not_fitted(
    temp_model_path,
):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    assert not any("ensemble" in n.lower() for n in predictor.get_model_names())


def test_given_enable_ensemble_false_when_predictor_called_then_ensemble_is_not_fitted(temp_model_path):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
        enable_ensemble=False,
    )
    assert not any("ensemble" in n.lower() for n in predictor.get_model_names())


def test_given_model_fails_when_predictor_predicts_then_exception_is_raised(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters={"ETS": {}})
    with mock.patch("autogluon.timeseries.models.local.statsforecast.ETSModel.predict") as arima_predict:
        arima_predict.side_effect = RuntimeError("Numerical error")
        with pytest.raises(RuntimeError, match="Following models failed to predict: \\['ETS'\\]"):
            predictor.predict(DUMMY_TS_DATAFRAME)


def test_given_model_fails_when_predictor_scores_then_exception_is_raised(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters={"ETS": {}})
    with mock.patch("autogluon.timeseries.models.local.statsforecast.ETSModel.predict") as arima_predict:
        arima_predict.side_effect = RuntimeError("Numerical error")
        with pytest.raises(RuntimeError, match="Following models failed to predict: \\['ETS'\\]"):
            predictor.score(DUMMY_TS_DATAFRAME)


def test_given_no_searchspace_and_hyperparameter_tune_kwargs_when_predictor_fits_then_exception_is_raised(
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    with pytest.raises(ValueError, match="no model contains a hyperparameter search space"):
        predictor.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"SimpleFeedForward": {"epochs": 1}},
            hyperparameter_tune_kwargs="random",
        )


def test_given_searchspace_and_no_hyperparameter_tune_kwargs_when_predictor_fits_then_exception_is_raised(
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    with pytest.raises(
        ValueError, match="Hyperparameter tuning not specified, so hyperparameters must have fixed values"
    ):
        predictor.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"SimpleFeedForward": {"epochs": space.Categorical(1, 2)}},
        )


def test_given_mixed_searchspace_and_hyperparameter_tune_kwargs_when_predictor_fits_then_no_exception_is_raised(
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": space.Categorical(1, 2), "ETS": {}}},
        hyperparameter_tune_kwargs={
            "scheduler": "local",
            "searcher": "random",
            "num_trials": 2,
        },
    )


@pytest.mark.parametrize("target_column", ["target", "CUSTOM_TARGET"])
def test_when_target_included_in_known_covariates_then_exception_is_raised(temp_model_path, target_column):
    with pytest.raises(ValueError, match="cannot be one of the known covariates"):
        TimeSeriesPredictor(
            path=temp_model_path, target=target_column, known_covariates_names=["Y", target_column, "X"]
        )


@pytest.mark.parametrize(
    "hyperparameters, num_models",
    [
        ({"Naive": {}}, 1),
        ({"Naive": {}, "DeepAR": {"epochs": 1, "num_batches_per_epoch": 1}}, 3),  # + 1 for ensemble
    ],
)
def test_when_fit_summary_is_called_then_all_keys_and_models_are_included(
    temp_model_path, hyperparameters, num_models
):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(DUMMY_TS_DATAFRAME, hyperparameters=hyperparameters)
    expected_keys = [
        "model_types",
        "model_performance",
        "model_best",
        "model_paths",
        "model_fit_times",
        "model_pred_times",
        "model_hyperparams",
        "leaderboard",
    ]
    fit_summary = predictor.fit_summary()
    for key in expected_keys:
        assert key in fit_summary
        # All keys except model_best return a dict with results per model
        if key != "model_best":
            assert len(fit_summary[key]) == num_models


@pytest.mark.parametrize(
    "hyperparameters, num_models",
    [
        ({"Naive": {}}, 1),
        ({"Naive": {}, "DeepAR": {"epochs": 1, "num_batches_per_epoch": 1}}, 3),  # + 1 for ensemble
    ],
)
def test_when_info_is_called_then_all_keys_and_models_are_included(temp_model_path, hyperparameters, num_models):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(DUMMY_TS_DATAFRAME, hyperparameters=hyperparameters)
    expected_keys = [
        "path",
        "version",
        "time_fit_training",
        "time_limit",
        "best_model",
        "best_model_score_val",
        "num_models_trained",
        "model_info",
    ]
    info = predictor.info()
    for key in expected_keys:
        assert key in info

    assert len(info["model_info"]) == num_models


def test_when_train_data_contains_nans_then_predictor_can_fit(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    df = DUMMY_TS_DATAFRAME.copy()
    df.iloc[5] = np.nan
    predictor.fit(
        df,
        hyperparameters=TEST_HYPERPARAMETER_SETTINGS[0],
    )
    assert "SimpleFeedForward" in predictor.get_model_names()


def test_when_prediction_data_contains_nans_then_predictor_can_predict(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}})
    df = DUMMY_TS_DATAFRAME.copy()
    df.iloc[5] = np.nan
    predictions = predictor.predict(df)
    assert isinstance(predictions, TimeSeriesDataFrame)
    assert not np.any(np.isnan(predictions))


def test_when_some_time_series_contain_only_nans_then_exception_is_raised(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    df = TimeSeriesDataFrame.from_iterable_dataset(
        [
            {"target": [float(5)] * 10, "start": pd.Period("2020-01-01", "D")},
            {"target": [float("nan")] * 10, "start": pd.Period("2020-01-01", "D")},
        ]
    )
    with pytest.raises(ValueError, match="consist completely of NaN values"):
        predictor._check_and_prepare_data_frame(df)


@pytest.mark.parametrize("method", ["score", "leaderboard"])
def test_when_scoring_method_receives_only_future_data_then_exception_is_raised(temp_model_path, method):
    prediction_length = 3
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=prediction_length)
    predictor.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}})
    future_data = DUMMY_TS_DATAFRAME.slice_by_timestep(-prediction_length, None)
    with pytest.raises(ValueError, match=" data includes both historic and future data"):
        getattr(predictor, method)(data=future_data)


def test_when_fit_receives_only_future_data_as_tuning_data_then_exception_is_raised(temp_model_path):
    prediction_length = 3
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=prediction_length)
    future_data = DUMMY_TS_DATAFRAME.slice_by_timestep(-prediction_length, None)
    with pytest.raises(ValueError, match="tuning\_data includes both historic and future data"):
        predictor.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}}, tuning_data=future_data)


def test_given_data_is_in_dataframe_format_then_predictor_works(temp_model_path):
    df = pd.DataFrame(DUMMY_TS_DATAFRAME.reset_index())
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(df, hyperparameters={"Naive": {}})
    predictor.leaderboard(df)
    predictor.score(df)
    predictions = predictor.predict(df)
    assert isinstance(predictions, TimeSeriesDataFrame)


def test_given_data_is_in_str_format_then_predictor_works(temp_model_path):
    df = pd.DataFrame(DUMMY_TS_DATAFRAME.reset_index())
    with tempfile.NamedTemporaryFile("w") as data_path:
        df.to_csv(data_path, index=False)
        predictor = TimeSeriesPredictor(path=temp_model_path)
        predictor.fit(df, hyperparameters={"Naive": {}})
        predictor.leaderboard(df)
        predictor.score(df)
        predictions = predictor.predict(df)
        assert isinstance(predictions, TimeSeriesDataFrame)


@pytest.mark.parametrize("rename_columns", [{TIMESTAMP: "custom_timestamp"}, {ITEMID: "custom_item_id"}])
def test_given_data_cannot_be_interpreted_as_tsdf_then_exception_raised(temp_model_path, rename_columns):
    df = pd.DataFrame(DUMMY_TS_DATAFRAME.reset_index())
    df = df.rename(columns=rename_columns)
    predictor = TimeSeriesPredictor(path=temp_model_path)
    with pytest.raises(ValueError, match="cannot be automatically converted to a TimeSeriesDataFrame"):
        predictor.fit(df, hyperparameters={"Naive": {}})


def test_given_data_is_not_sorted_then_predictor_can_fit_and_predict(temp_model_path):
    shuffled_df = pd.DataFrame(DUMMY_TS_DATAFRAME).sample(frac=1.0)
    ts_df = TimeSeriesDataFrame(shuffled_df)

    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=2)
    predictor.fit(ts_df, hyperparameters={"Naive": {}})
    predictions = predictor.predict(ts_df)
    assert len(predictions) == predictor.prediction_length * ts_df.num_items


def test_given_data_is_not_sorted_then_preprocessed_data_is_sorted(temp_model_path):
    shuffled_df = pd.DataFrame(DUMMY_TS_DATAFRAME).sample(frac=1.0)
    ts_df = TimeSeriesDataFrame(shuffled_df)

    predictor = TimeSeriesPredictor(path=temp_model_path)
    ts_df_processed = predictor._check_and_prepare_data_frame(ts_df)
    assert ts_df_processed.index.is_monotonic_increasing


def test_when_both_argument_aliases_are_passed_to_init_then_exception_is_raised(temp_model_path):
    with pytest.raises(ValueError, match="Please specify at most one of these arguments"):
        TimeSeriesPredictor(path=temp_model_path, target="custom_target", label="custom_target")


def test_when_invalid_argument_passed_to_init_then_exception_is_raised(temp_model_path):
    with pytest.raises(TypeError, match="unexpected keyword argument 'invalid_argument'"):
        TimeSeriesPredictor(path=temp_model_path, invalid_argument=23)


def test_when_invalid_argument_passed_to_fit_then_exception_is_raised(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    with pytest.raises(TypeError, match="unexpected keyword argument 'invalid_argument'"):
        predictor.fit(DUMMY_TS_DATAFRAME, invalid_argument=23)


@pytest.mark.parametrize("set_best_to_refit_full", [True, False])
def test_when_refit_full_called_then_best_model_is_updated(temp_model_path, set_best_to_refit_full):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={
            "DeepAR": {"epochs": 1, "num_batches_per_epoch": 1},
            "SimpleFeedForward": {"epochs": 1, "num_batches_per_epoch": 1},
        },
    )
    model_best_before = predictor.get_model_best()
    model_full_dict = predictor.refit_full(set_best_to_refit_full=set_best_to_refit_full)
    model_best_after = predictor.get_model_best()
    if set_best_to_refit_full:
        assert model_best_after == model_full_dict[model_best_before]
    else:
        assert model_best_after == model_best_before


@pytest.mark.parametrize("tuning_data, refit_called", [(None, True), (DUMMY_TS_DATAFRAME, False)])
def test_when_refit_full_is_passed_to_fit_then_refit_full_is_skipped(temp_model_path, tuning_data, refit_called):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    with mock.patch("autogluon.timeseries.predictor.TimeSeriesPredictor.refit_full") as refit_method:
        predictor.fit(
            DUMMY_TS_DATAFRAME,
            tuning_data=tuning_data,
            hyperparameters=TEST_HYPERPARAMETER_SETTINGS[0],
            refit_full=True,
        )
        if refit_called:
            refit_method.assert_called()
        else:
            refit_method.assert_not_called()


def test_when_excluded_model_names_provided_then_excluded_models_are_not_trained(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={
            "DeepAR": {"epochs": 1, "num_batches_per_epoch": 1},
            "SimpleFeedForward": {"epochs": 1, "num_batches_per_epoch": 1},
        },
        excluded_model_types=["DeepAR"],
    )
    leaderboard = predictor.leaderboard()
    assert leaderboard["model"].values == ["SimpleFeedForward"]


@pytest.mark.parametrize("method_name", ["leaderboard", "predict", "score", "evaluate"])
@pytest.mark.parametrize("use_cache", [True, False])
def test_when_use_cache_is_set_to_false_then_cached_predictions_are_ignored(temp_model_path, use_cache, method_name):
    predictor = TimeSeriesPredictor(path=temp_model_path, cache_predictions=True).fit(
        DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}}
    )
    # Cache predictions
    predictor.predict(DUMMY_TS_DATAFRAME)

    with mock.patch(
        "autogluon.timeseries.trainer.abstract_trainer.AbstractTimeSeriesTrainer._get_cached_pred_dicts"
    ) as mock_get_cached_pred_dicts:
        mock_get_cached_pred_dicts.return_value = {}, {}
        getattr(predictor, method_name)(DUMMY_TS_DATAFRAME, use_cache=use_cache)
        if use_cache:
            mock_get_cached_pred_dicts.assert_called()
        else:
            mock_get_cached_pred_dicts.assert_not_called()


@pytest.fixture(
    scope="module",
    params=[
        [
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:01:00",
                "2020-01-04 00:01:00",
                "2020-01-06 00:01:00",
                "2020-01-07 00:01:00",
            ],
        ],
        [
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
                "2020-01-06 00:00:00",
            ],
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
                "2020-01-06 00:00:00",
            ],
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
                "2020-01-06 00:00:01",
            ],
        ],
    ],
)
def irregular_timestamp_data_frame(request):
    df_tuples = []
    for i, ts in enumerate(request.param):
        for t in ts:
            df_tuples.append((i, pd.Timestamp(t), np.random.rand()))
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(df_tuples, columns=[ITEMID, TIMESTAMP, "target"]))


def test_given_irregular_time_series_when_predictor_called_with_freq_then_predictor_can_predict(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        freq="D",
    )
    predictor.fit(
        train_data=df,
        hyperparameters=TEST_HYPERPARAMETER_SETTINGS[0],
        tuning_data=df,
    )
    predictions = predictor.predict(df)
    assert isinstance(df, TimeSeriesDataFrame)
    assert not np.any(np.isnan(predictions))
    assert all(len(predictions.loc[i]) == 1 for i in df.item_ids)
    assert "SimpleFeedForward" in predictor.get_model_names()


def test_given_irregular_time_series_and_no_tuning_when_predictor_called_with_freq_then_predictor_can_predict(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        freq="D",
    )
    predictor.fit(
        train_data=df,
        hyperparameters=TEST_HYPERPARAMETER_SETTINGS[0],
    )
    predictions = predictor.predict(df)
    assert isinstance(df, TimeSeriesDataFrame)
    assert not np.any(np.isnan(predictions))
    assert all(len(predictions.loc[i]) == 1 for i in df.item_ids)
    assert "SimpleFeedForward" in predictor.get_model_names()


@pytest.mark.parametrize("predictor_freq", ["H", "2H", "20T"])
def test_given_regular_time_series_when_predictor_called_with_freq_then_predictions_have_predictor_freq(
    temp_model_path, predictor_freq
):
    df = DUMMY_TS_DATAFRAME.copy()
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        freq=predictor_freq,
        prediction_length=3,
    )
    predictor.fit(
        train_data=df,
        hyperparameters=TEST_HYPERPARAMETER_SETTINGS[0],
    )
    predictions = predictor.predict(df)
    assert predictions.freq == predictor_freq


def test_given_irregular_time_series_when_predictor_called_without_freq_then_training_fails(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
    )
    with pytest.raises(ValueError, match="expected data frequency"):
        predictor.fit(
            train_data=df,
            hyperparameters=TEST_HYPERPARAMETER_SETTINGS[0],
        )


def test_given_regular_time_series_when_predictor_called_without_freq_then_freq_is_inferred(temp_model_path):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
    )
    assert predictor.freq is None
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=TEST_HYPERPARAMETER_SETTINGS[0],
    )
    assert predictor.freq is not None
    assert predictor.freq == DUMMY_TS_DATAFRAME.freq


def test_given_regular_time_series_when_predictor_loaded_from_disk_then_inferred_freq_persists(temp_model_path):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
    )
    assert predictor.freq is None
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=TEST_HYPERPARAMETER_SETTINGS[0],
    )
    predictor.save()
    del predictor

    loaded_predictor = TimeSeriesPredictor.load(temp_model_path)
    assert loaded_predictor.freq is not None
    assert loaded_predictor.freq == DUMMY_TS_DATAFRAME.freq


@pytest.mark.parametrize("prediction_length", [2, 7])
@pytest.mark.parametrize("num_val_windows", [1, 5])
@pytest.mark.parametrize("val_step_size", [1, 4])
def test_given_short_and_long_series_in_train_data_when_fit_called_then_trainer_receives_only_long_series(
    temp_model_path, prediction_length, num_val_windows, val_step_size
):
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=prediction_length, freq="H")
    min_train_length = predictor._min_train_length

    item_id_to_length = {
        "long_series_1": min_train_length + prediction_length + num_val_windows * val_step_size,
        "long_series_2": min_train_length + prediction_length + (num_val_windows - 1) * val_step_size,
        "short_series_1": min_train_length + (num_val_windows - 1) * val_step_size,
        "short_series_2": min_train_length + 1,
        "short_series_3": 2,
    }
    data = get_data_frame_with_variable_lengths(item_id_to_length, freq="H")
    with mock.patch("autogluon.timeseries.learner.TimeSeriesLearner.fit") as learner_fit:
        predictor.fit(data, num_val_windows=num_val_windows, val_step_size=val_step_size)
        learner_fit_kwargs = learner_fit.call_args[1]
        item_ids_received_by_learner = learner_fit_kwargs["train_data"].item_ids
        assert (item_ids_received_by_learner == ["long_series_1", "long_series_2"]).all()


@pytest.mark.parametrize("prediction_length", [1, 7])
def test_given_short_and_long_series_in_train_data_and_tuning_data_when_fit_called_then_trainer_receives_only_long_series(
    temp_model_path, prediction_length
):
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=prediction_length, freq="H")
    min_train_length = predictor._min_train_length

    item_id_to_length = {
        "long_series_1": min_train_length,
        "short_series_1": min_train_length - 1,
        "short_series_2": 2,
    }
    data = get_data_frame_with_variable_lengths(item_id_to_length, freq="H")
    with mock.patch("autogluon.timeseries.learner.TimeSeriesLearner.fit") as learner_fit:
        predictor.fit(data, tuning_data=DUMMY_TS_DATAFRAME)
        learner_fit_kwargs = learner_fit.call_args[1]
        item_ids_received_by_learner = learner_fit_kwargs["train_data"].item_ids
        assert (item_ids_received_by_learner == ["long_series_1"]).all()


@pytest.mark.parametrize("num_val_windows", [1, 3, None])
def test_given_tuning_data_when_fit_called_then_num_val_windows_is_set_to_zero(temp_model_path, num_val_windows):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    with mock.patch("autogluon.timeseries.learner.TimeSeriesLearner.fit") as learner_fit:
        predictor.fit(DUMMY_TS_DATAFRAME, tuning_data=DUMMY_TS_DATAFRAME, num_val_windows=num_val_windows)
        learner_fit_kwargs = learner_fit.call_args[1]
        assert learner_fit_kwargs["val_splitter"].num_val_windows == 0


@pytest.mark.parametrize("prediction_length", [1, 5, 7])
@pytest.mark.parametrize("val_step_size", [1, 3])
def test_when_num_val_windows_is_recommended_then_increasing_num_val_windows_raises_error(
    temp_model_path, prediction_length, val_step_size
):
    df = DUMMY_TS_DATAFRAME.copy()
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=prediction_length)
    recommended_num_val_windows = predictor._recommend_num_val_windows(
        df, max_num_val_windows=100, val_step_size=val_step_size
    )
    # assert that recommended_num_val_windows is the highest value for num_val_windows that doesn't raise an exception
    assert predictor._filter_short_series(df, recommended_num_val_windows, val_step_size).num_items == df.num_items
    with pytest.raises(ValueError, match="At least some time series in train\_data must have length"):
        predictor._filter_short_series(df, recommended_num_val_windows + 1, val_step_size).num_items < df.num_items


@pytest.mark.parametrize("prediction_length", [1, 7])
@pytest.mark.parametrize("num_val_windows", [1, 3])
@pytest.mark.parametrize("val_step_size", [1, 3])
def test_given_only_short_series_in_train_data_when_fit_called_then_exception_is_raised(
    temp_model_path, prediction_length, num_val_windows, val_step_size
):
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=prediction_length, freq="H")
    min_train_length = predictor._min_train_length

    item_id_to_length = {
        "short_series_1": min_train_length + prediction_length - 1,
        "short_series_2": min_train_length,
        "short_series_3": 2,
    }
    data = get_data_frame_with_variable_lengths(item_id_to_length, freq="H")
    with pytest.raises(ValueError, match="At least some time series in train\_data must have length"):
        predictor.fit(data, num_val_windows=num_val_windows, val_step_size=val_step_size)


@pytest.mark.parametrize("prediction_length", [1, 7])
def test_given_only_short_series_when_num_val_windows_is_recommended_then_exception_is_raised(
    temp_model_path, prediction_length
):
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=prediction_length, freq="H")
    min_train_length = predictor._min_train_length

    item_id_to_length = {
        "short_series_1": min_train_length + prediction_length - 1,
        "short_series_2": min_train_length,
        "short_series_3": 2,
    }
    data = get_data_frame_with_variable_lengths(item_id_to_length, freq="H")
    with pytest.raises(ValueError, match="At least some time series in train\_data must have length"):
        predictor.fit(data, num_val_windows=None)


@pytest.mark.parametrize(
    "num_val_windows, refit_every_n_windows, expected_num_refits", [(5, None, 1), (7, 7, 1), (5, 1, 5), (6, 2, 3)]
)
def test_given_refit_every_n_windows_when_fit_then_model_is_fit_correct_number_of_times(
    temp_model_path, num_val_windows, refit_every_n_windows, expected_num_refits
):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(
        DUMMY_TS_DATAFRAME,
        num_val_windows=num_val_windows,
        refit_every_n_windows=refit_every_n_windows,
        hyperparameters={"Naive": {}},
    )
    models_info = predictor._trainer.get_models_info(["Naive"])
    actual_num_refits = 0
    for window_info in models_info["Naive"]["info_per_val_window"]:
        actual_num_refits += window_info["refit_this_window"]
    assert actual_num_refits == expected_num_refits
