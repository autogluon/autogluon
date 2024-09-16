from unittest import mock

import numpy as np
import pytest

from autogluon.timeseries.models import NaiveModel
from autogluon.timeseries.transforms.scaler import get_target_scaler

from .common import DUMMY_TS_DATAFRAME
from .models.test_models import get_multi_window_deepar

SCALER_TYPES = ["standard", "mean_abs", "min_max", "robust"]
TESTABLE_MODELS = [NaiveModel, get_multi_window_deepar]


@pytest.mark.parametrize("scaler_type", SCALER_TYPES)
def test_when_scaler_transforms_then_input_data_is_not_modified(scaler_type):
    scaler = get_target_scaler(scaler_type=scaler_type)
    data = DUMMY_TS_DATAFRAME.copy()
    data_original = data.copy()
    data_transformed = scaler.fit_transform(data)
    assert data.equals(data_original)
    assert not data.equals(data_transformed)


@pytest.mark.parametrize("scaler_type", SCALER_TYPES)
def test_when_scaler_transforms_then_no_new_nans_appear(scaler_type):
    scaler = get_target_scaler(scaler_type=scaler_type)
    data = DUMMY_TS_DATAFRAME.copy()
    data_transformed = scaler.fit_transform(data)
    assert data.isna().equals(data_transformed.isna())


@pytest.mark.parametrize("scaler_type", SCALER_TYPES)
def test_when_inverse_transform_applied_then_output_matches_input(scaler_type):
    scaler = get_target_scaler(scaler_type=scaler_type)
    data = DUMMY_TS_DATAFRAME.copy()
    data_transformed = scaler.fit_transform(data)
    data_inverse = scaler.inverse_transform(data_transformed)
    assert np.allclose(data, data_inverse, equal_nan=True)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("scaler_type", SCALER_TYPES)
def test_when_model_fits_then_fit_transform_called_once(model_class, scaler_type):
    data = DUMMY_TS_DATAFRAME.copy()
    model = model_class(
        prediction_length=4,
        freq=data.freq,
        hyperparameters={
            "max_epochs": 1,
            "num_batches_per_epoch": 1,
            "target_transform": scaler_type,
        },
    )
    with mock.patch(
        "autogluon.timeseries.transforms.scaler.LocalTargetScaler.fit_transform",
        side_effect=lambda x: x,
    ) as scaler_fit_transform:
        model.fit(train_data=data)
    assert scaler_fit_transform.call_count == 1


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("scaler_type", SCALER_TYPES)
def test_when_model_predicts_then_fit_transform_called_once(model_class, scaler_type):
    data = DUMMY_TS_DATAFRAME.copy()
    model = model_class(
        prediction_length=4,
        freq=data.freq,
        hyperparameters={
            "max_epochs": 1,
            "num_batches_per_epoch": 1,
            "target_transform": scaler_type,
        },
    )
    model.fit(train_data=data)
    with mock.patch(
        "autogluon.timeseries.transforms.scaler.LocalTargetScaler.fit_transform",
        side_effect=lambda x: x,
    ) as scaler_fit_transform:
        model.predict(data)
    assert scaler_fit_transform.call_count == 1
