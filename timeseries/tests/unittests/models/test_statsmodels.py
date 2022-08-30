import logging

import pytest

from autogluon.timeseries.models.statsmodels import ARIMAModel, ETSModel

from ..common import DUMMY_TS_DATAFRAME, get_data_frame_with_item_index

TESTABLE_MODELS = [
    ARIMAModel,
    ETSModel,
]


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_statsmodels_models_saved_then_fitted_models_can_be_loaded(model_class, temp_model_path):
    model = model_class(freq=DUMMY_TS_DATAFRAME.freq, path=temp_model_path)
    model.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters={"maxiter": 1})
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    for ts_hash, model in model._fitted_models.items():
        assert ts_hash in loaded_model._fitted_models
        assert loaded_model._fitted_models[ts_hash] == model


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("n_jobs", [0.5, 3])
def test_when_statsmodels_models_saved_then_n_jobs_is_saved(model_class, n_jobs, temp_model_path):
    model = model_class(freq=DUMMY_TS_DATAFRAME.freq, path=temp_model_path, hyperparameters={"n_jobs": n_jobs})
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    assert model.n_jobs == loaded_model.n_jobs


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("train_data_size, test_data_size", [(5, 10), (10, 5), (5, 5)])
def test_when_predict_called_with_test_data_then_predictor_inference_correct(
    model_class, temp_model_path, train_data_size, test_data_size, caplog
):
    train_data = get_data_frame_with_item_index(list(range(train_data_size)))
    test_data = get_data_frame_with_item_index(list(range(test_data_size)))

    model = model_class(
        path=temp_model_path, prediction_length=3, hyperparameters={"maxiter": 1}, freq=train_data.freq
    )
    model.fit(train_data=train_data)

    with caplog.at_level(logging.INFO):
        _ = model.predict(test_data)
        assert f"received {test_data_size} items not seen during training, re-running fit" in caplog.text


def get_seasonal_period_from_fitted_local_model(model, model_name):
    if model_name == "ARIMA":
        return model.model_init_args["seasonal_order"][-1]
    elif model_name == "ETS":
        return model.model_init_args["seasonal_periods"]


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("hyperparameters", [{"seasonal_period": None}, {}])
@pytest.mark.parametrize(
    "freqstr, ts_length, expected_seasonal_period",
    [
        ("H", 100, 24),
        ("2H", 100, 12),
        ("B", 100, 5),
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
    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters=hyperparameters,
        freq=train_data.freq,
    )

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
        path=temp_model_path,
        prediction_length=3,
        hyperparameters={"seasonal_period": provided_seasonal_period},
        freq=train_data.freq,
    )

    model.fit(train_data=train_data)
    # Select one of the fitted models
    single_fitted_model = next(iter(model._fitted_models.values()))
    model_seasonal_period = get_seasonal_period_from_fitted_local_model(single_fitted_model, model.name)
    assert model_seasonal_period == provided_seasonal_period


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
        assert "bad_argument" not in single_fitted_model.model_init_args
