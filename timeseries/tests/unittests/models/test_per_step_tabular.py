import shutil
from pathlib import Path
from pickle import PicklingError
from unittest import mock

import numpy as np
import pytest

from ..common import DUMMY_TS_DATAFRAME, get_data_frame_with_variable_lengths

DUMMY_HYPERPARAMETERS = {"model_name": "DUMMY", "n_jobs": 1}
EVAL_METRICS = ["WAPE", "WQL"]


@pytest.mark.parametrize(
    "seasonal_lags, trailing_lags, expected_lags_per_step",
    [
        ([], [1, 2, 3, 4], [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
        ([1, 2, 3, 4], [], [[1, 2, 3, 4], [2, 3, 4], [3, 4]]),
        ([5, 10, 20], [], [[5, 10, 20], [5, 10, 20], [5, 10, 20]]),
        ([12, 24], [1, 2], [[1, 2, 12, 24], [2, 3, 12, 24], [3, 4, 12, 24]]),
    ],
)
@pytest.mark.parametrize("eval_metric", EVAL_METRICS)
def test_when_seasonal_and_trailing_lags_are_provided_then_each_model_receives_correct_lags(
    per_step_tabular_model_class,
    df_with_covariates_and_metadata,
    seasonal_lags,
    trailing_lags,
    expected_lags_per_step,
    eval_metric,
):
    df, covariate_metadata = df_with_covariates_and_metadata
    prediction_length = 3
    model = per_step_tabular_model_class(
        prediction_length=prediction_length,
        freq=df.freq,
        covariate_metadata=covariate_metadata,
        hyperparameters={"trailing_lags": trailing_lags, "seasonal_lags": seasonal_lags, **DUMMY_HYPERPARAMETERS},
        eval_metric=eval_metric,
    )

    with mock.patch.object(model, "_fit_single_model") as mock_fit_single_model:
        model.fit(train_data=df)
        for step, call_args in enumerate(mock_fit_single_model.call_args_list):
            call_kwargs = call_args[1]
            assert call_kwargs["step"] == step
            assert sorted(call_kwargs["lags"]) == expected_lags_per_step[step]


@pytest.mark.parametrize("eval_metric", EVAL_METRICS)
def test_when_model_predicts_then_tabular_models_receive_correct_data_for_inference(
    per_step_tabular_model_class, df_with_covariates_and_metadata, eval_metric
):
    df, covariate_metadata = df_with_covariates_and_metadata
    df = df.sort_index()
    prediction_length = 3
    model = per_step_tabular_model_class(
        prediction_length=prediction_length,
        freq=df.freq,
        covariate_metadata=covariate_metadata,
        eval_metric=eval_metric,
        hyperparameters={"trailing_lags": [1, 2, 3], "seasonal_lags": [5, 10], **DUMMY_HYPERPARAMETERS},
    )
    model.fit(train_data=df)
    past_data, known_covariates = df.get_model_inputs_for_scoring(
        model.prediction_length, known_covariates_names=covariate_metadata.known_covariates
    )
    with mock.patch("autogluon.core.models.dummy.dummy_model.DummyModel.predict") as mock_tabular_predict:
        if model.eval_metric.needs_quantile:
            mock_tabular_predict.return_value = np.zeros([df.num_items, len(model.quantile_levels)])
        else:
            mock_tabular_predict.return_value = np.zeros([df.num_items])
        model.predict(past_data, known_covariates)
        for step, call_args in enumerate(mock_tabular_predict.call_args_list):
            X = call_args[0][0]
            item_ids_timestamp_for_step = known_covariates.slice_by_timestep(step, step + 1).index.to_frame(
                index=False
            )
            assert (X["unique_id"].values == item_ids_timestamp_for_step["item_id"].values).all()
            assert (X["ds"].values == item_ids_timestamp_for_step["timestamp"].values).all()


@pytest.mark.parametrize("n_jobs", [2, -1])
def test_when_n_jobs_provided_via_hyperparameters_then_it_is_stored_as_attribute(per_step_tabular_model_class, n_jobs):
    data = DUMMY_TS_DATAFRAME.copy()
    model = per_step_tabular_model_class(freq=data.freq, hyperparameters={"n_jobs": n_jobs, "model_name": "DUMMY"})
    model.fit(train_data=data)
    assert model._n_jobs == n_jobs


@pytest.mark.parametrize(
    "n_jobs, time_limit, expected_time_limit_per_model",
    [
        (1, 3.0, 1.0),
        (3, 10.0, 10.0),
    ],
)
def test_when_models_are_fitted_then_time_limit_is_distributed_evenly(
    per_step_tabular_model_class, n_jobs, time_limit, expected_time_limit_per_model
):
    data = DUMMY_TS_DATAFRAME.copy()
    prediction_length = 3
    model = per_step_tabular_model_class(
        freq=data.freq, prediction_length=prediction_length, hyperparameters={"n_jobs": n_jobs, "model_name": "DUMMY"}
    )
    with mock.patch.object(model, "_fit_single_model") as mock_fit_single_model:
        mock_fit_single_model.return_value = "dummy_path"
        model.fit(train_data=data, time_limit=time_limit)
        for call_args in mock_fit_single_model.call_args_list:
            assert np.isclose(
                call_args[1]["time_limit"],
                model.default_max_time_limit_ratio * expected_time_limit_per_model,
                atol=0.1,
            )


@pytest.mark.parametrize(
    "trailing_lags, seasonal_lags, date_features",
    [
        ([], [], []),
        ([], [1, 2, 3], []),
        ([1, 2, 3], [], []),
        ([], [5, 10, 15], ["quarter"]),
        ([1, 2, 3], [30, 40], ["day_of_week", "hour"]),
    ],
)
@pytest.mark.parametrize("eval_metric", EVAL_METRICS)
def test_when_per_step_models_are_fit_then_each_model_receives_correct_features(
    per_step_tabular_model_class,
    df_with_covariates_and_metadata,
    trailing_lags,
    seasonal_lags,
    date_features,
    eval_metric,
):
    df, covariate_metadata = df_with_covariates_and_metadata
    model = per_step_tabular_model_class(
        freq=df.freq,
        prediction_length=3,
        covariate_metadata=covariate_metadata,
        eval_metric=eval_metric,
        hyperparameters={
            **DUMMY_HYPERPARAMETERS,
            "trailing_lags": trailing_lags,
            "seasonal_lags": seasonal_lags,
            "date_features": date_features,
        },
    )
    with mock.patch("autogluon.core.models.dummy.dummy_model.DummyModel.fit") as mock_dummy_fit:
        try:
            model.fit(train_data=df)
        # Mock leads to AttributeError during fit
        except AttributeError:
            pass
        expected_num_features = (
            len(trailing_lags)
            + len(seasonal_lags)
            + len(date_features)
            + len(covariate_metadata.known_covariates)
            + len(covariate_metadata.static_features)
            + len(covariate_metadata.known_covariates_real)
        )
        for step, call_args in enumerate(mock_dummy_fit.call_args_list):
            received_num_features = len(call_args[1]["X"].columns)
            expected_num_features_for_step = expected_num_features - sum(lag <= step for lag in seasonal_lags)
            assert expected_num_features_for_step == received_num_features


@pytest.mark.parametrize(
    "trailing_lags, seasonal_lags",
    [
        ([0, 1, 2], []),
        ([], [0, 1, 2]),
        ([-1], []),
        ([], [-1]),
        ([-2], [-1]),
    ],
)
def test_when_invalid_lags_are_passed_then_exception_is_raised_during_fit(
    per_step_tabular_model_class, trailing_lags, seasonal_lags
):
    data = DUMMY_TS_DATAFRAME.copy()
    model = per_step_tabular_model_class(
        freq=data.freq, hyperparameters={"trailing_lags": trailing_lags, "seasonal_lags": seasonal_lags}
    )
    with pytest.raises(AssertionError, match="must be >= 1"):
        model.fit(train_data=data)


@pytest.mark.parametrize("eval_metric", EVAL_METRICS)
def test_when_model_is_copied_to_new_folder_then_loaded_model_can_still_predict(
    per_step_tabular_model_class, tmp_path, eval_metric
):
    data = DUMMY_TS_DATAFRAME.copy()
    model = per_step_tabular_model_class(
        freq=data.freq,
        prediction_length=3,
        hyperparameters=DUMMY_HYPERPARAMETERS,
        path=str(tmp_path),
        eval_metric=eval_metric,
    )
    model.fit(train_data=data)
    model.save()

    new_path = str(tmp_path / "new_location")
    shutil.copytree(model.path, new_path)
    loaded_model = per_step_tabular_model_class.load(new_path)

    pred1 = model.predict(data)
    pred2 = loaded_model.predict(data)

    assert pred1.equals(pred2)


@pytest.mark.parametrize("prediction_length", [1, 7])
def test_when_max_num_samples_provided_then_train_df_is_shortened(per_step_tabular_model_class, prediction_length):
    data = get_data_frame_with_variable_lengths({k: 10000 for k in range(5)})
    max_num_samples = 1000
    model = per_step_tabular_model_class(
        freq=data.freq,
        prediction_length=prediction_length,
        hyperparameters={**DUMMY_HYPERPARAMETERS, "max_num_samples": max_num_samples},
    )

    with mock.patch.object(model, "_fit_single_model") as mock_fit_single_model:
        model.fit(train_data=data)
        train_df_received = mock_fit_single_model.call_args[1]["train_df"]
        assert len(train_df_received) == max_num_samples + model.prediction_length * data.num_items


def test_when_max_num_items_provided_then_train_df_removes_items(per_step_tabular_model_class):
    data = get_data_frame_with_variable_lengths({k: 20 for k in range(1000)})
    max_num_items = 4
    model = per_step_tabular_model_class(
        freq=data.freq, hyperparameters={**DUMMY_HYPERPARAMETERS, "max_num_items": max_num_items}
    )
    with mock.patch.object(model, "_fit_single_model") as mock_fit_single_model:
        model.fit(train_data=data)
        train_df_received = mock_fit_single_model.call_args[1]["train_df"]
        assert train_df_received["unique_id"].nunique() == max_num_items


def test_when_validation_fraction_is_nonzero_then_validation_set_is_created(per_step_tabular_model_class):
    val_frac = 0.2
    N = 100
    data = get_data_frame_with_variable_lengths({"A": N})
    data["target"] = range(len(data))
    model = per_step_tabular_model_class(
        freq=data.freq,
        hyperparameters={**DUMMY_HYPERPARAMETERS, "validation_fraction": val_frac, "target_scaler": None},
    )
    with mock.patch("autogluon.core.models.dummy.dummy_model.DummyModel.fit") as mock_model_fit:
        model.fit(train_data=data)
        call_kwargs = mock_model_fit.call_args[1]
        assert call_kwargs["X_val"] is not None
        assert (call_kwargs["y_val"].values == list(range(N - int(N * val_frac), N))).all()


@pytest.mark.parametrize("validation_fraction", [None, 0.0])
def test_when_validation_fraction_is_zero_then_no_validation_set_created(
    per_step_tabular_model_class, validation_fraction
):
    data = DUMMY_TS_DATAFRAME.copy()
    model = per_step_tabular_model_class(
        freq=data.freq, hyperparameters={**DUMMY_HYPERPARAMETERS, "validation_fraction": validation_fraction}
    )

    with mock.patch("autogluon.core.models.dummy.dummy_model.DummyModel.fit") as mock_model_fit:
        model.fit(train_data=data)
        call_kwargs = mock_model_fit.call_args[1]
        assert call_kwargs["X_val"] is None
        assert call_kwargs["y_val"] is None


@pytest.mark.parametrize("eval_metric", EVAL_METRICS)
def test_when_model_is_fit_then_internal_model_receives_correct_hyperparameters(
    per_step_tabular_model_class, eval_metric
):
    data = DUMMY_TS_DATAFRAME.copy()
    hyperparameters = {
        "n_jobs": 1,
        "model_name": "GBM",
        "model_hyperparameters": {"max_depth": 7, "n_estimators": 97},
    }
    model = per_step_tabular_model_class(
        freq=data.freq, prediction_length=2, hyperparameters=hyperparameters, eval_metric=eval_metric
    )

    with mock.patch("autogluon.tabular.models.lgb.lgb_model.train_lgb_model") as mock_train_lgb:
        # Using mock breaks the fitting process with a PicklingError
        try:
            model.fit(train_data=data)
        except PicklingError:
            pass
        call_kwargs = mock_train_lgb.call_args[1]
        assert call_kwargs["params"]["max_depth"] == 7
        assert call_kwargs["params"]["n_estimators"] == 97


@pytest.mark.parametrize("model_name, eval_metric", [("LR", "WQL"), ("FASTTEXT", "WAPE")])
def test_when_model_does_not_support_required_problem_type_then_exception_raised(
    per_step_tabular_model_class, model_name, eval_metric
):
    data = DUMMY_TS_DATAFRAME.copy()
    model = per_step_tabular_model_class(
        freq=data.freq, eval_metric=eval_metric, hyperparameters={"model_name": model_name}
    )

    with pytest.raises(ValueError, match=r"does not support problem_type"):
        model.fit(train_data=data)


@pytest.mark.parametrize(
    "forecasting_eval_metric, problem_type, tabular_eval_metric",
    [
        ("WQL", "quantile", "pinball_loss"),
        ("SQL", "quantile", "pinball_loss"),
        ("WAPE", "regression", "mean_absolute_error"),
        ("RMSSE", "regression", "root_mean_squared_error"),
    ],
)
def test_when_eval_metric_is_chosen_then_tabular_model_receives_correct_problem_type_and_eval_metric(
    per_step_tabular_model_class, forecasting_eval_metric, problem_type, tabular_eval_metric
):
    data = DUMMY_TS_DATAFRAME.copy()
    model = per_step_tabular_model_class(
        freq=data.freq, eval_metric=forecasting_eval_metric, hyperparameters=DUMMY_HYPERPARAMETERS
    )
    with mock.patch(
        "autogluon.core.models.dummy.dummy_model.DummyModel.__init__", side_effect=RuntimeError
    ) as mock_dummy_init:
        try:
            model.fit(train_data=data)
        # Intercept the error raised by mock
        except RuntimeError:
            pass
        call_kwargs = mock_dummy_init.call_args[1]
        assert call_kwargs["problem_type"] == problem_type
        assert call_kwargs["eval_metric"] == tabular_eval_metric


def test_when_regression_mode_is_used_then_residuals_are_saved(per_step_tabular_model_class, tmp_path):
    data = DUMMY_TS_DATAFRAME.copy()
    model = per_step_tabular_model_class(
        freq=data.freq,
        eval_metric="WAPE",
        hyperparameters=DUMMY_HYPERPARAMETERS,
        path=str(tmp_path),
        prediction_length=4,
    )
    model.fit(train_data=data)
    for step in range(model.prediction_length):
        residuals_path = Path(model.path) / f"step_{step}" / model._model_cls.__name__ / "residuals_std.pkl"
        assert residuals_path.exists()


def test_when_regression_mode_predicts_then_quantile_columns_are_strictly_increasing(per_step_tabular_model_class):
    data = DUMMY_TS_DATAFRAME.copy()
    model = per_step_tabular_model_class(
        freq=data.freq, eval_metric="WAPE", hyperparameters=DUMMY_HYPERPARAMETERS, prediction_length=4
    )
    model.fit(train_data=data)
    predictions = model.predict(data)

    quantile_cols = [col for col in predictions.columns if col != "mean"]
    quantile_values = predictions[quantile_cols].to_numpy()
    assert np.all(quantile_values[:, :-1] <= quantile_values[:, 1:])
