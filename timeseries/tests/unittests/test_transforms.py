from unittest import mock

import numpy as np
import pandas as pd
import pytest

from autogluon.common import space
from autogluon.timeseries.models import NaiveModel
from autogluon.timeseries.transforms.covariate_scaler import (
    AVAILABLE_COVARIATE_SCALERS,
    GlobalCovariateScaler,
    get_covariate_scaler,
)
from autogluon.timeseries.transforms.target_scaler import (
    AVAILABLE_TARGET_SCALERS,
    LocalMeanAbsScaler,
    LocalMinMaxScaler,
    LocalRobustScaler,
    LocalStandardScaler,
    get_target_scaler,
)
from autogluon.timeseries.utils.features import CovariateMetadata

from .common import DUMMY_TS_DATAFRAME
from .models.common import get_multi_window_deepar

TESTABLE_MODELS = [NaiveModel, get_multi_window_deepar]


@pytest.mark.parametrize("scaler_name", AVAILABLE_TARGET_SCALERS)
def test_when_scaler_transforms_then_input_data_is_not_modified(scaler_name):
    scaler = get_target_scaler(scaler_name)
    data = DUMMY_TS_DATAFRAME.copy()
    data_original = data.copy()
    data_transformed = scaler.fit_transform(data)
    assert data.equals(data_original)
    assert not data.equals(data_transformed)


@pytest.mark.parametrize("scaler_name", AVAILABLE_TARGET_SCALERS)
def test_when_scaler_transforms_then_no_new_nans_appear(scaler_name):
    scaler = get_target_scaler(scaler_name)
    data = DUMMY_TS_DATAFRAME.copy()
    data_transformed = scaler.fit_transform(data)
    assert data.isna().equals(data_transformed.isna())


@pytest.mark.parametrize("scaler_name", AVAILABLE_TARGET_SCALERS)
def test_when_inverse_transform_applied_then_output_matches_input(scaler_name):
    scaler = get_target_scaler(scaler_name)
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
        "autogluon.timeseries.transforms.target_scaler.LocalTargetScaler.fit_transform",
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
        "autogluon.timeseries.transforms.target_scaler.LocalTargetScaler.fit_transform",
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
    model = NaiveModel(prediction_length=4, hyperparameters={"target_scaler": "invalid_scaler"})
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


def test_when_global_covariate_scaler_used_then_correct_feature_types_are_detected():
    covariate_scaler = GlobalCovariateScaler(covariate_metadata=CovariateMetadata())
    N = 500
    df = pd.DataFrame(
        {
            "bool": np.random.choice([0, 1], size=N).astype(float),
            "skewed": np.random.exponential(size=N),
            "normal": np.random.normal(size=N),
        }
    )
    pipeline = covariate_scaler._get_transformer_for_columns(df, df.columns)
    normal_pipeline, skewed_pipeline = pipeline.transformers
    assert normal_pipeline[-1] == ["normal"]
    assert skewed_pipeline[-1] == ["skewed"]


@pytest.mark.parametrize("scaler_name", AVAILABLE_COVARIATE_SCALERS)
def test_when_covariate_scaler_is_used_then_original_data_is_not_modified(
    scaler_name, df_with_covariates_and_metadata
):
    df, covariate_metadata = df_with_covariates_and_metadata
    scaler = get_covariate_scaler(scaler_name, covariate_metadata=covariate_metadata)
    data, known_covariates = df.get_model_inputs_for_scoring(
        prediction_length=2, known_covariates_names=covariate_metadata.known_covariates
    )
    data_orig = data.copy()
    known_covariates_orig = known_covariates.copy()
    static_features_orig = data.static_features.copy()

    scaler.fit_transform(data)
    scaler.transform_known_covariates(known_covariates)

    assert data_orig.equals(data)
    assert known_covariates_orig.equals(known_covariates)
    assert static_features_orig.equals(data.static_features)


def test_when_global_covariate_scaler_is_fit_then_column_transformers_are_created(df_with_covariates_and_metadata):
    df, covariate_metadata = df_with_covariates_and_metadata
    scaler = GlobalCovariateScaler(covariate_metadata=covariate_metadata, skew_threshold=1e10)
    scaler.fit_transform(df)
    assert scaler.is_fit()
    assert scaler._column_transformers["known"].transformers_[-1][-1] == ["cov2"]
    assert scaler._column_transformers["static"].transformers_[-1][-1] == ["feat1", "feat3"]


def test_when_global_covariate_scaler_transforms_then_real_columns_are_standardized(df_with_covariates_and_metadata):
    def is_standardized(series: pd.Series, atol: float = 1e-2) -> bool:
        return bool(np.isclose(series.mean(), 0, atol=atol) and np.isclose(series.std(ddof=0), 1, atol=atol))

    df, covariate_metadata = df_with_covariates_and_metadata
    # ensure that StandardScaler is used for all features by setting large skew_threshold
    scaler = GlobalCovariateScaler(covariate_metadata=covariate_metadata, skew_threshold=1e10)
    df_out = scaler.fit_transform(df)
    assert is_standardized(df_out["cov2"])
    assert is_standardized(df_out.static_features["feat1"])
    assert is_standardized(df_out.static_features["feat3"])


def test_when_hyperparameter_spaces_of_transforms_provided_to_init_then_model_can_tune(temp_model_path):
    model = NaiveModel(
        path=temp_model_path,
        freq=DUMMY_TS_DATAFRAME.freq,
        hyperparameters={
            "target_scaler": space.Categorical("standard", "mean_abs"),
        },
    )
    hpo_models, _ = model.hyperparameter_tune(
        hyperparameter_tune_kwargs={"num_trials": 3, "scheduler": "local", "searcher": "random"},
        time_limit=30,
        train_data=DUMMY_TS_DATAFRAME,
        val_data=DUMMY_TS_DATAFRAME,
    )
    assert len(hpo_models) >= 2
