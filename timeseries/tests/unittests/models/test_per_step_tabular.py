from unittest import mock

import numpy as np
import pytest

from ..common import DUMMY_TS_DATAFRAME

DUMMY_HYPERPARAMETERS = {"model_name": "DUMMY", "n_jobs": 1}


@pytest.mark.parametrize(
    "seasonal_lags, trailing_lags, expected_lags_per_step",
    [
        ([], [1, 2, 3, 4], [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
        ([1, 2, 3, 4], [], [[1, 2, 3, 4], [2, 3, 4], [3, 4]]),
        ([5, 10, 20], [], [[5, 10, 20], [5, 10, 20], [5, 10, 20]]),
        ([12, 24], [1, 2], [[1, 2, 12, 24], [2, 3, 12, 24], [3, 4, 12, 24]]),
    ],
)
def test_when_seasonal_and_trailing_lags_are_provided_then_each_model_receives_correct_lags(
    per_step_tabular_model_class, df_with_covariates_and_metadata, seasonal_lags, trailing_lags, expected_lags_per_step
):
    df, covariate_metadata = df_with_covariates_and_metadata
    prediction_length = 3
    model = per_step_tabular_model_class(
        prediction_length=prediction_length,
        freq=df.freq,
        covariate_metadata=covariate_metadata,
        hyperparameters={"trailing_lags": trailing_lags, "seasonal_lags": seasonal_lags, **DUMMY_HYPERPARAMETERS},
    )

    with mock.patch.object(model, "_fit_single_model") as mock_fit_single_model:
        model.fit(train_data=df)
        for step, call_args in enumerate(mock_fit_single_model.call_args_list):
            call_kwargs = call_args[1]
            assert call_kwargs["step"] == step
            assert sorted(call_kwargs["lags"]) == expected_lags_per_step[step]


def test_when_model_predicts_then_tabular_models_receive_correct_data_for_inference(
    per_step_tabular_model_class, df_with_covariates_and_metadata
):
    df, covariate_metadata = df_with_covariates_and_metadata
    df = df.sort_index()
    prediction_length = 3
    model = per_step_tabular_model_class(
        prediction_length=prediction_length,
        freq=df.freq,
        covariate_metadata=covariate_metadata,
        hyperparameters={"trailing_lags": [1, 2, 3], "seasonal_lags": [5, 10], **DUMMY_HYPERPARAMETERS},
    )
    model.fit(train_data=df)
    past_data, known_covariates = df.get_model_inputs_for_scoring(
        model.prediction_length, known_covariates_names=covariate_metadata.known_covariates
    )
    with mock.patch("autogluon.core.models.dummy.dummy_model.DummyModel.predict") as mock_tabular_predict:
        mock_tabular_predict.return_value = np.zeros([df.num_items, len(model.quantile_levels)])
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
def test_when_models_are_fitten_then_time_limit_is_distributed_evenly(
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
