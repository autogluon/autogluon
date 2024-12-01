import time
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries.dataset.ts_dataframe import ITEMID
from autogluon.timeseries.models import ZeroModel
from autogluon.timeseries.models.multi_window import MultiWindowBacktestingModel
from autogluon.timeseries.regressor import CovariateRegressor
from autogluon.timeseries.utils.features import CovariateMetadata


def get_multi_window_zero_model(hyperparameters=None, **kwargs):
    """Wrap ZeroModel inside MultiWindowBacktestingModel."""
    if hyperparameters is None:
        hyperparameters = {"n_jobs": 1, "use_fallback_model": False}
    model_base_kwargs = {**kwargs, "hyperparameters": hyperparameters}
    return MultiWindowBacktestingModel(model_base=ZeroModel, model_base_kwargs=model_base_kwargs, **kwargs)


TESTABLE_MODELS = [ZeroModel, get_multi_window_zero_model]
MODEL_HPS = {"model_hyperparameters": {"max_iter": 5}}


@pytest.fixture(scope="module")
def get_model_with_regressor(dummy_hyperparameters, df_with_covariates_and_metadata):
    """Used as get_model_with_regressor(model_cls, covariate_regressor, extra_hyperparameters)"""

    def _get_model(model_class, covariate_regressor=None, extra_hyperparameters=None):
        if extra_hyperparameters is None:
            extra_hyperparameters = {}
        data, metadata = df_with_covariates_and_metadata
        return model_class(
            freq=data.freq,
            metadata=metadata,
            prediction_length=2,
            hyperparameters={
                "covariate_regressor": covariate_regressor,
                **extra_hyperparameters,
                **dummy_hyperparameters,
            },
        )

    return _get_model


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("refit_during_predict", [True, False])
def test_when_refit_during_predict_is_true_then_regressor_is_trained_during_predict(
    model_class, get_model_with_regressor, df_with_covariates_and_metadata, refit_during_predict
):
    df, metadata = df_with_covariates_and_metadata
    regressor = CovariateRegressor("LR", **MODEL_HPS, refit_during_predict=refit_during_predict)
    model = get_model_with_regressor(model_class, regressor)
    model.fit(train_data=df)
    past, known_covariates = df.get_model_inputs_for_scoring(
        model.prediction_length, known_covariates_names=metadata.known_covariates
    )
    with mock.patch("autogluon.timeseries.regressor.CovariateRegressor.fit", wraps=regressor.fit) as mock_fit:
        model.predict(past, known_covariates)
        if refit_during_predict:
            mock_fit.assert_called()
        else:
            mock_fit.assert_not_called()


def test_when_model_is_used_with_regressor_then_regressor_methods_are_called_the_expected_number_of_times(
    df_with_covariates_and_metadata, get_model_with_regressor
):
    df, metadata = df_with_covariates_and_metadata
    model = get_model_with_regressor(ZeroModel, "LR")
    past, known_covariates = df.get_model_inputs_for_scoring(
        model.prediction_length, known_covariates_names=metadata.known_covariates
    )
    model.initialize()
    regressor = model.covariate_regressor
    with mock.patch("autogluon.timeseries.regressor.CovariateRegressor.fit", wraps=regressor.fit) as mock_fit:
        with mock.patch(
            "autogluon.timeseries.regressor.CovariateRegressor.transform", wraps=regressor.transform
        ) as mock_transform:
            with mock.patch(
                "autogluon.timeseries.regressor.CovariateRegressor.inverse_transform",
                wraps=regressor.inverse_transform,
            ) as mock_inverse_transform:
                model.fit(train_data=df)
                model.predict(past, known_covariates)
    assert mock_fit.call_count == 1
    assert mock_transform.call_count == 2
    assert mock_inverse_transform.call_count == 1


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("include_past_covariates", [True, False])
def test_when_data_contains_no_known_covariates_or_static_features_then_regressor_is_not_used(
    model_class, include_past_covariates, df_with_covariates_and_metadata
):
    df, metadata_full = df_with_covariates_and_metadata
    if include_past_covariates:
        metadata = CovariateMetadata(
            past_covariates_cat=metadata_full.covariates_cat, past_covariates_real=metadata_full.covariates_real
        )
    else:
        metadata = CovariateMetadata()
    model = model_class(freq=df.freq, metadata=metadata, hyperparameters={"covariate_regressor": "LR"})
    model.fit(train_data=df)
    assert model.covariate_regressor is None


def test_when_regressor_is_used_then_tabular_df_contains_correct_features(
    df_with_covariates_and_metadata, get_model_with_regressor
):
    df, metadata = df_with_covariates_and_metadata
    model = get_model_with_regressor(ZeroModel, "LR")
    with mock.patch("autogluon.tabular.models.LinearModel.fit") as mock_lr_fit:
        try:
            model.fit(train_data=df)
        except KeyError:
            # Ignore KeyError produced by mock
            pass
        features = mock_lr_fit.call_args[1]["X"].columns
    assert set(features) == set(metadata.static_features + metadata.known_covariates + [ITEMID])


def test_when_target_scaler_and_regressor_are_used_then_regressor_receives_scaled_data_as_input(
    df_with_covariates_and_metadata, get_model_with_regressor
):
    df, metadata = df_with_covariates_and_metadata
    model = get_model_with_regressor(ZeroModel, "LR", extra_hyperparameters={"target_scaler": "min_max"})
    model.fit(train_data=df)
    past, known_covariates = df.get_model_inputs_for_scoring(
        model.prediction_length, known_covariates_names=metadata.known_covariates
    )
    regressor = model.covariate_regressor
    with mock.patch(
        "autogluon.timeseries.regressor.CovariateRegressor.fit_transform", wraps=regressor.fit_transform
    ) as mock_transform:
        model.predict(past, known_covariates)

    input_data_stats = mock_transform.call_args[0][0][model.target].groupby(ITEMID).agg(["min", "max"])
    assert np.allclose(input_data_stats["min"], 0)
    assert np.allclose(input_data_stats["max"], 1)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_covariate_regressor_used_then_residuals_are_subtracted_before_forecaster_fits(
    model_class,
    df_with_covariates_and_metadata,
    dummy_hyperparameters,
):
    df, _ = df_with_covariates_and_metadata
    # Shift the mean of each item; assert that the shift is removed by the regressor before model receives the data
    df["target"] += pd.Series([10, 20, 30, 40], index=df.item_ids)
    df["covariate"] = np.ones(len(df))
    df.static_features = None
    metadata = CovariateMetadata(known_covariates_real=["covariate"])
    model = model_class(
        freq=df.freq,
        metadata=metadata,
        hyperparameters={"covariate_regressor": "LR", **dummy_hyperparameters},
    )
    with mock.patch("autogluon.timeseries.models.ZeroModel._fit") as mock_fit:
        try:
            model.fit(train_data=df)
        except AttributeError:
            # Ignore AttributeError produced by mock
            pass
    input_data_mean = mock_fit.call_args[1]["train_data"][model.target].groupby(ITEMID).mean()
    assert np.allclose(input_data_mean, 0, atol=2.0)


def test_when_validation_fraction_is_set_then_tabular_model_uses_val_data(df_with_covariates_and_metadata):
    df, metadata = df_with_covariates_and_metadata
    regressor = CovariateRegressor("LR", **MODEL_HPS, validation_fraction=0.1, metadata=metadata)
    with mock.patch("autogluon.tabular.models.LinearModel.fit") as mock_lr_fit:
        regressor.fit(df)
    assert mock_lr_fit.call_args[1]["X_val"] is not None
    assert mock_lr_fit.call_args[1]["y_val"] is not None


def test_when_validation_fraction_is_none_then_tabular_model_doesnt_use_val_data(df_with_covariates_and_metadata):
    df, metadata = df_with_covariates_and_metadata
    regressor = CovariateRegressor("LR", **MODEL_HPS, validation_fraction=None, metadata=metadata)
    with mock.patch("autogluon.tabular.models.LinearModel.fit") as mock_lr_fit:
        regressor.fit(df)
    assert mock_lr_fit.call_args[1]["X_val"] is None
    assert mock_lr_fit.call_args[1]["y_val"] is None


def test_when_not_enough_time_is_left_to_predict_then_regressor_is_disabled(df_with_covariates_and_metadata):
    df, metadata = df_with_covariates_and_metadata
    regressor = CovariateRegressor("CAT", metadata=metadata)

    def predict_with_sleep(X, **kwargs):
        time.sleep(5)
        return np.zeros(len(X))

    with mock.patch("autogluon.tabular.models.CatBoostModel.predict", side_effect=predict_with_sleep):
        regressor.fit(df, time_limit=5)

    assert regressor.disabled_due_to_time_limit


@pytest.mark.parametrize("use_fit_transform", [True, False])
@pytest.mark.parametrize("refit_during_predict", [True, False])
def test_when_regressor_is_disabled_then_data_is_not_modified_during_transform(
    df_with_covariates_and_metadata, use_fit_transform, refit_during_predict
):
    df, metadata = df_with_covariates_and_metadata
    regressor = CovariateRegressor("LR", **MODEL_HPS, metadata=metadata, refit_during_predict=refit_during_predict)
    regressor.fit(df)
    regressor.disabled_due_to_time_limit = True
    if use_fit_transform:
        df_out = regressor.fit_transform(df)
    else:
        df_out = regressor.transform(df)
    assert df_out is df
