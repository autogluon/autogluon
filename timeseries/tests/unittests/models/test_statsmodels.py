import logging

import pandas as pd
import pytest

from autogluon.timeseries.models.statsmodels import ARIMAModel, ETSModel, ThetaModel

from ..common import DUMMY_TS_DATAFRAME, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, get_data_frame_with_item_index

TESTABLE_MODELS = [
    ARIMAModel,
    ETSModel,
    ThetaModel,
]


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_statsmodels_model_saved_then_fitted_models_can_be_loaded(model_class, temp_model_path):
    model = model_class(path=temp_model_path)
    model.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters={"maxiter": 1})
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    for ts_hash, model in model._fitted_models.items():
        assert ts_hash in loaded_model._fitted_models
        assert loaded_model._fitted_models[ts_hash] == model


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_statsmodels_model_is_saved_and_loaded_then_model_can_predict(model_class, temp_model_path):
    model = model_class(path=temp_model_path)
    model.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters={"maxiter": 1})
    model.save()
    loaded_model = model.__class__.load(path=model.path)
    loaded_model.predict(data=DUMMY_TS_DATAFRAME)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("n_jobs", [0.5, 3])
def test_when_statsmodels_models_saved_then_n_jobs_is_saved(model_class, n_jobs, temp_model_path):
    model = model_class(path=temp_model_path, hyperparameters={"n_jobs": n_jobs, "maxiter": 1})
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    assert model.n_jobs == loaded_model.n_jobs


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 3, 10])
def test_when_statsmodels_model_predicts_then_time_index_is_correct(model_class, prediction_length, temp_model_path):
    data = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME
    model = model_class(path=temp_model_path, prediction_length=prediction_length)
    model.fit(train_data=data, hyperparameters={"maxiter": 1})
    predictions = model.predict(data=data)
    for item_id in data.item_ids:
        cutoff = data.loc[item_id].index.max()
        start = cutoff + pd.tseries.frequencies.to_offset(data.freq)
        expected_timestamps = pd.date_range(start, periods=prediction_length, freq=data.freq)
        assert (predictions.loc[item_id].index == expected_timestamps).all()


@pytest.mark.skip("Skip for now because of the logging changes.")
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("train_data_size, test_data_size", [(5, 10), (10, 5), (5, 5)])
def test_when_predict_called_with_test_data_then_predictor_inference_correct(
    model_class, temp_model_path, train_data_size, test_data_size, caplog
):
    train_data = get_data_frame_with_item_index(list(range(train_data_size)))
    test_data = get_data_frame_with_item_index(list(range(test_data_size)))

    model = model_class(path=temp_model_path, prediction_length=3, hyperparameters={"maxiter": 1})
    model.fit(train_data=train_data)

    with caplog.at_level(logging.INFO):
        _ = model.predict(test_data)
        assert f"received {test_data_size} items not seen during training, re-running fit" in caplog.text


def get_seasonal_period_from_fitted_local_model(model, model_name):
    if model_name == "ARIMA":
        return model.sm_model_init_args["seasonal_order"][-1]
    elif model_name == "ETS":
        return model.sm_model_init_args["seasonal_periods"]
    elif model_name == "Theta":
        return model.sm_model_init_args["period"]


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("hyperparameters", [{"seasonal_period": None}, {}])
@pytest.mark.parametrize(
    "freqstr, ts_length, expected_seasonal_period",
    [
        ("H", 100, 24),
        ("2H", 100, 12),
        ("B", 100, 5),
        ("D", 100, 7),
        ("M", 100, 12),
    ],
)
def test_when_seasonal_period_is_set_to_none_then_inferred_period_is_used(
    model_class,
    hyperparameters,
    temp_model_path,
    freqstr,
    ts_length,
    expected_seasonal_period,
):
    train_data = get_data_frame_with_item_index(["A", "B", "C"], data_length=ts_length, freq=freqstr)
    model = model_class(path=temp_model_path, prediction_length=3, hyperparameters=hyperparameters)

    model.fit(train_data=train_data)
    single_fitted_model = next(iter(model._fitted_models.values()))
    model_seasonal_period = get_seasonal_period_from_fitted_local_model(single_fitted_model, model.name)
    assert model_seasonal_period == expected_seasonal_period


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize(
    "freqstr, ts_length, provided_seasonal_period",
    [
        ("H", 100, 12),
        ("2H", 100, 5),
        ("B", 100, 10),
        ("D", 100, 8),
        ("M", 100, 24),
    ],
)
def test_when_seasonal_period_is_provided_then_inferred_period_is_overriden(
    model_class,
    temp_model_path,
    freqstr,
    ts_length,
    provided_seasonal_period,
):
    train_data = get_data_frame_with_item_index(["A", "B", "C"], data_length=ts_length, freq=freqstr)
    model = model_class(
        path=temp_model_path, prediction_length=3, hyperparameters={"seasonal_period": provided_seasonal_period}
    )

    model.fit(train_data=train_data)
    # Select one of the fitted models
    single_fitted_model = next(iter(model._fitted_models.values()))
    model_seasonal_period = get_seasonal_period_from_fitted_local_model(single_fitted_model, model.name)
    assert model_seasonal_period == provided_seasonal_period


@pytest.mark.skip("Skip for now because of the logging changes.")
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_invalid_model_arguments_provided_then_statsmodels_ignores_them(model_class, temp_model_path, caplog):
    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters={"bad_argument": 33},
    )
    with caplog.at_level(logging.WARNING):
        model.fit(train_data=DUMMY_TS_DATAFRAME)
        assert "ignores following hyperparameters: ['bad_argument']" in caplog.text
        single_fitted_model = next(iter(model._fitted_models.values()))
        assert "bad_argument" not in single_fitted_model.sm_model_init_args


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_train_and_test_data_have_different_freq_then_exception_is_raised(model_class, temp_model_path):
    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters={"maxiter": 1},
    )
    train_data = get_data_frame_with_item_index([1, 2, 3], freq="H")
    test_data = get_data_frame_with_item_index([1, 2, 3], freq="D")

    model.fit(train_data=train_data)
    with pytest.raises(RuntimeError, match="must match the frequency"):
        model.predict(test_data)
