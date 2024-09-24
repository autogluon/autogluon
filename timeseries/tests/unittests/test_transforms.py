from unittest import mock

import numpy as np
import pytest

from autogluon.timeseries.models import NaiveModel
from autogluon.timeseries.transforms.scaler import (
    AVAILABLE_SCALERS,
    LocalMeanAbsScaler,
    LocalMinMaxScaler,
    LocalRobustScaler,
    LocalStandardScaler,
    get_target_scaler_from_name,
)

from .common import DUMMY_TS_DATAFRAME
from .models.test_models import get_multi_window_deepar

TESTABLE_MODELS = [NaiveModel, get_multi_window_deepar]


@pytest.mark.parametrize("scaler_name", AVAILABLE_SCALERS)
def test_when_scaler_transforms_then_input_data_is_not_modified(scaler_name):
    scaler = get_target_scaler_from_name(scaler_name)
    data = DUMMY_TS_DATAFRAME.copy()
    data_original = data.copy()
    data_transformed = scaler.fit_transform(data)
    assert data.equals(data_original)
    assert not data.equals(data_transformed)


@pytest.mark.parametrize("scaler_name", AVAILABLE_SCALERS)
def test_when_scaler_transforms_then_no_new_nans_appear(scaler_name):
    scaler = get_target_scaler_from_name(scaler_name)
    data = DUMMY_TS_DATAFRAME.copy()
    data_transformed = scaler.fit_transform(data)
    assert data.isna().equals(data_transformed.isna())


@pytest.mark.parametrize("scaler_name", AVAILABLE_SCALERS)
def test_when_inverse_transform_applied_then_output_matches_input(scaler_name):
    scaler = get_target_scaler_from_name(scaler_name)
    data = DUMMY_TS_DATAFRAME.copy()
    data_transformed = scaler.fit_transform(data)
    data_inverse = scaler.inverse_transform(data_transformed)
    assert np.allclose(data, data_inverse, equal_nan=True)


@pytest.mark.parametrize(
    "model_class, expected_call_count",
    [
        (NaiveModel, 1),
        (get_multi_window_deepar, 2),  # call once during fit(), once during score_and_cache_oof()
    ],
)
def test_when_model_fits_then_fit_transform_called_as_many_times_as_expected(model_class, expected_call_count):
    data = DUMMY_TS_DATAFRAME.copy()
    model = model_class(
        prediction_length=4,
        freq=data.freq,
        hyperparameters={
            "max_epochs": 1,
            "num_batches_per_epoch": 1,
            "target_scaler": "standard",
        },
    )
    with mock.patch(
        "autogluon.timeseries.transforms.scaler.LocalTargetScaler.fit_transform",
        side_effect=lambda x: x,
    ) as scaler_fit_transform:
        model.fit(train_data=data)
    assert scaler_fit_transform.call_count == expected_call_count


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_model_predicts_then_fit_transform_called_once(model_class):
    data = DUMMY_TS_DATAFRAME.copy()
    model = model_class(
        prediction_length=4,
        freq=data.freq,
        hyperparameters={
            "max_epochs": 1,
            "num_batches_per_epoch": 1,
            "target_scaler": "min_max",
        },
    )
    model.fit(train_data=data)
    with mock.patch(
        "autogluon.timeseries.transforms.scaler.LocalTargetScaler.fit_transform",
        side_effect=lambda x: x,
    ) as scaler_fit_transform:
        model.predict(data)
    assert scaler_fit_transform.call_count == 1


@pytest.mark.parametrize(
    "scaler_name, scaler_cls",
    [
        ("mean_abs", LocalMeanAbsScaler),
        ("min_max", LocalMinMaxScaler),
        ("robust", LocalRobustScaler),
        ("standard", LocalStandardScaler),
    ],
)
def test_given_target_scaler_param_set_when_model_fits_then_target_scaler_created(scaler_name, scaler_cls):
    data = DUMMY_TS_DATAFRAME.copy()
    model = NaiveModel(
        prediction_length=4,
        freq=data.freq,
        hyperparameters={"target_scaler": scaler_name},
    )
    model.fit(train_data=data)
    assert isinstance(model.target_scaler, scaler_cls)


def test_given_invalid_scaler_name_when_model_fits_then_exception_is_raised():
    model = NaiveModel(
        prediction_length=4,
        hyperparameters={"target_scaler": "invalid_scaler"},
    )
    with pytest.raises(KeyError, match="not supported. Available scalers"):
        model.fit(train_data=DUMMY_TS_DATAFRAME)


@pytest.mark.parametrize("hyperparameters", [{}, {"target_scaler": None}])
def test_given_no_scaler_name_when_model_fits_then_no_scaler_is_added(hyperparameters):
    data = DUMMY_TS_DATAFRAME.copy()
    model = NaiveModel(
        prediction_length=4,
        freq=data.freq,
        hyperparameters=hyperparameters,
    )
    model.fit(train_data=data)
    assert model.target_scaler is None
