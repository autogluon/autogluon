"""Unit tests for predictors"""
import copy
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from autogluon.common.space import Space, Categorical, Real, Int, Bool
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP
from autogluon.timeseries.models import DeepARModel, SimpleFeedForwardModel
from autogluon.timeseries.predictor import TimeSeriesPredictor

from .common import DUMMY_TS_DATAFRAME

TEST_HYPERPARAMETER_SETTINGS = [
    {"SimpleFeedForward": {"epochs": 1}},
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


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS + ["local_only"])  # noqa
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


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS + ["local_only"])  # noqa
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
@pytest.mark.parametrize("quantile_kwarg_name", ["quantiles", "quantile_levels"])
def test_given_hyperparameters_and_quantiles_when_predictor_called_then_model_can_predict(
    temp_model_path, hyperparameters, quantile_kwarg_name
):
    predictor_init_kwargs = dict(path=temp_model_path, eval_metric="MAPE", prediction_length=3)
    predictor_init_kwargs[quantile_kwarg_name] = [0.1, 0.4, 0.9]
    predictor = TimeSeriesPredictor(**predictor_init_kwargs)

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
        {"ETS": {"maxiter": 1}, "SimpleFeedForward": {"epochs": Int(1, 3)}},
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
        if any(isinstance(v, Space) for v in hps.values()):
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


@pytest.fixture(
    scope="module",
    params=[
        [
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00", "2020-01-04 00:01:00"],
        ],
        [
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00", "2020-01-04 00:00:00"],
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00", "2020-01-04 00:00:01"],
        ],
        [
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00", "2020-01-04 00:00:00"],
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00", "2020-01-05 00:00:00"],
        ],
        [
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00", "2020-01-04 00:00:00"],
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00", "2020-01-04 00:00:00"],
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00", "2020-01-04 00:00:00"],
        ],
    ],
)
def irregular_timestamp_data_frame(request):
    df_tuples = []
    for i, ts in enumerate(request.param):
        for t in ts:
            df_tuples.append((i, pd.Timestamp(t), np.random.rand()))
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(df_tuples, columns=[ITEMID, TIMESTAMP, "target"]))


def test_given_irregular_time_series_when_predictor_called_with_ignore_then_training_is_performed(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        ignore_time_index=True,
    )
    predictor.fit(
        train_data=df,
        hyperparameters={"DeepAR": {"epochs": 1}},
        tuning_data=df,
    )
    assert "DeepAR" in predictor.get_model_names()


def test_given_irregular_time_series_and_no_tuning_when_predictor_called_with_ignore_then_training_is_performed(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        ignore_time_index=True,
    )
    predictor.fit(
        train_data=df,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    assert "SimpleFeedForward" in predictor.get_model_names()


def test_given_irregular_time_series_when_predictor_called_without_ignore_then_training_fails(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        ignore_time_index=False,
    )
    with pytest.raises(ValueError):
        predictor.fit(
            train_data=df,
            hyperparameters={"SimpleFeedForward": {"epochs": 1}},
        )


def test_given_irregular_time_series_when_predictor_called_with_ignore_then_predictor_can_predict(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        ignore_time_index=True,
    )
    predictor.fit(
        train_data=df,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    predictions = predictor.predict(df)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == df.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 1 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


def test_given_irregular_time_series_when_predictor_called_without_ignore_then_predictor_cannot_predict(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        ignore_time_index=False,
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME.get_reindexed_view(),
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    with pytest.raises(ValueError, match="irregularly sampled"):
        _ = predictor.predict(df)


@pytest.mark.parametrize("ignore_time_index", [True, False])
def test_when_predictor_called_and_loaded_back_then_ignore_time_index_persists(temp_model_path, ignore_time_index):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        prediction_length=2,
        ignore_time_index=ignore_time_index,
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    predictor.save()
    del predictor

    loaded_predictor = TimeSeriesPredictor.load(temp_model_path)
    assert loaded_predictor.ignore_time_index == ignore_time_index


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


def test_given_model_fails_when_predictor_predicts_then_exception_is_caught_by_learner(temp_model_path):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"ARIMA": {"maxiter": 1, "seasonal_period": 1, "seasonal_order": (0, 0, 0)}},
    )
    with mock.patch("autogluon.timeseries.models.local.statsmodels.ARIMAModel.predict") as arima_predict:
        arima_predict.side_effect = RuntimeError("Numerical error")
        with pytest.raises(RuntimeError, match="Prediction failed, please provide a different model to"):
            predictor.predict(DUMMY_TS_DATAFRAME)


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
            hyperparameters={"SimpleFeedForward": {"epochs": Categorical(1, 2)}},
        )


def test_given_mixed_searchspace_and_hyperparameter_tune_kwargs_when_predictor_fits_then_no_exception_is_raised(
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": Categorical(1, 2), "ETS": {}}},
        hyperparameter_tune_kwargs={
            "scheduler": "local",
            "searcher": "random",
            "num_trials": 2,
        },
    )


@pytest.mark.parametrize("target_column", ["target", "CUSTOM_TARGET"])
def test_when_target_included_in_known_covariates_then_exception_is_raised(temp_model_path, target_column):
    with pytest.raises(ValueError, match="cannot be one of the known covariates"):
        predictor = TimeSeriesPredictor(
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


def test_when_train_data_contains_nans_then_exception_is_raised(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    df = DUMMY_TS_DATAFRAME.copy()
    df.iloc[5] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        predictor.fit(df)


def test_when_prediction_data_contains_nans_then_exception_is_raised(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}})
    df = DUMMY_TS_DATAFRAME.copy()
    df.iloc[5] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        predictor.predict(df)


def test_given_data_is_in_dataframe_format_then_predictor_works(temp_model_path):
    df = pd.DataFrame(DUMMY_TS_DATAFRAME.reset_index())
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


@pytest.mark.parametrize(
    "arg_1, arg_2, value",
    [
        ("quantile_levels", "quantiles", [0.1, 0.4]),
        ("target", "label", "custom_target"),
    ],
)
def test_when_both_argument_aliases_are_passed_to_init_then_exception_is_raised(temp_model_path, arg_1, arg_2, value):
    init_kwargs = {arg_1: value, arg_2: value}
    with pytest.raises(ValueError, match="Please specify at most one of these arguments"):
        predictor = TimeSeriesPredictor(path=temp_model_path, **init_kwargs)


def test_when_invalid_argument_passed_to_init_then_exception_is_raised(temp_model_path):
    with pytest.raises(TypeError, match="unexpected keyword argument 'invalid_argument'"):
        predictor = TimeSeriesPredictor(path=temp_model_path, invalid_argument=23)


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
