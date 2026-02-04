from __future__ import annotations

import time
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models import ZeroModel
from autogluon.timeseries.models.multi_window import MultiWindowBacktestingModel
from autogluon.timeseries.regressor import GlobalCovariateRegressor
from autogluon.timeseries.utils.features import CovariateMetadata

from .common import DUMMY_TS_DATAFRAME


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

    def _get_model(
        model_class, covariate_regressor: str | dict | None = None, extra_hyperparameters: dict | None = None
    ):
        if extra_hyperparameters is None:
            extra_hyperparameters = {}
        if covariate_regressor is not None:
            assert isinstance(covariate_regressor, (str, dict))
            extra_hyperparameters["covariate_regressor"] = covariate_regressor

        data, covariate_metadata = df_with_covariates_and_metadata
        model = model_class(
            freq=data.freq,
            covariate_metadata=covariate_metadata,
            prediction_length=2,
            hyperparameters={
                **extra_hyperparameters,
                **dummy_hyperparameters,
            },
        )
        return model

    return _get_model


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("refit_during_predict", [True, False])
def test_when_refit_during_predict_is_true_then_regressor_is_trained_during_predict(
    model_class, get_model_with_regressor, df_with_covariates_and_metadata, refit_during_predict
):
    df, covariate_metadata = df_with_covariates_and_metadata
    model = get_model_with_regressor(model_class, {"model_name": "LR", "refit_during_predict": refit_during_predict})
    model.fit(train_data=df)
    past, known_covariates = df.get_model_inputs_for_scoring(
        model.prediction_length, known_covariates_names=covariate_metadata.known_covariates
    )
    with mock.patch("autogluon.timeseries.regressor.GlobalCovariateRegressor.fit") as mock_fit:
        model.predict(past, known_covariates)
        if refit_during_predict:
            mock_fit.assert_called()
        else:
            mock_fit.assert_not_called()


def test_when_model_is_used_with_regressor_then_regressor_methods_are_called_the_expected_number_of_times(
    df_with_covariates_and_metadata, get_model_with_regressor
):
    df, covariate_metadata = df_with_covariates_and_metadata
    model = get_model_with_regressor(ZeroModel, "LR")
    past, known_covariates = df.get_model_inputs_for_scoring(
        model.prediction_length, known_covariates_names=covariate_metadata.known_covariates
    )
    # We need to create the regressor before `model.fit()` to be able to patch it with `mock.patch(..., wraps=...)`
    model._initialize_transforms_and_regressor()
    regressor = model.covariate_regressor
    model._initialize_transforms_and_regressor = lambda *args: None
    with mock.patch("autogluon.timeseries.regressor.GlobalCovariateRegressor.fit", wraps=regressor.fit) as mock_fit:
        with mock.patch(
            "autogluon.timeseries.regressor.GlobalCovariateRegressor.transform", wraps=regressor.transform
        ) as mock_transform:
            with mock.patch(
                "autogluon.timeseries.regressor.GlobalCovariateRegressor.inverse_transform",
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
    df, covariate_metadata_full = df_with_covariates_and_metadata
    if include_past_covariates:
        covariate_metadata = CovariateMetadata(
            past_covariates_cat=covariate_metadata_full.covariates_cat,
            past_covariates_real=covariate_metadata_full.covariates_real,
            past_cat_cardinality={col: 2 for col in covariate_metadata_full.covariates_cat},
        )
    else:
        covariate_metadata = CovariateMetadata()
    model = model_class(
        freq=df.freq, covariate_metadata=covariate_metadata, hyperparameters={"covariate_regressor": "LR"}
    )
    model.fit(train_data=df)
    assert model.covariate_regressor is None


@pytest.mark.parametrize("include_static_features", [True, False])
@pytest.mark.parametrize("include_item_id", [True, False])
def test_when_regressor_is_used_then_tabular_df_contains_correct_features(
    df_with_covariates_and_metadata,
    get_model_with_regressor,
    include_static_features,
    include_item_id,
):
    df, covariate_metadata = df_with_covariates_and_metadata
    with mock.patch("autogluon.tabular.models.LinearModel.fit") as mock_lr_fit:
        model = get_model_with_regressor(
            ZeroModel,
            {
                "model_name": "LR",
                "include_static_features": include_static_features,
                "include_item_id": include_item_id,
            },
        )
        try:
            model.fit(train_data=df)
        except KeyError:
            # Ignore KeyError produced by mock
            pass
        features = mock_lr_fit.call_args[1]["X"].columns
    expected_features = covariate_metadata.known_covariates
    if include_item_id:
        expected_features += [TimeSeriesDataFrame.ITEMID]
    if include_static_features:
        expected_features += covariate_metadata.static_features
    assert set(features) == set(expected_features)


def test_when_target_scaler_and_regressor_are_used_then_regressor_receives_scaled_data_as_input(
    df_with_covariates_and_metadata, get_model_with_regressor
):
    df, covariate_metadata = df_with_covariates_and_metadata
    model = get_model_with_regressor(ZeroModel, "LR", extra_hyperparameters={"target_scaler": "min_max"})
    model.fit(train_data=df)
    past, known_covariates = df.get_model_inputs_for_scoring(
        model.prediction_length, known_covariates_names=covariate_metadata.known_covariates
    )
    regressor = model.covariate_regressor
    with mock.patch(
        "autogluon.timeseries.regressor.GlobalCovariateRegressor.fit_transform", wraps=regressor.fit_transform
    ) as mock_transform:
        model.predict(past, known_covariates)

    input_data_stats = (
        mock_transform.call_args[0][0][model.target].groupby(TimeSeriesDataFrame.ITEMID).agg(["min", "max"])
    )
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
    df["covariate"] = df.index.get_level_values(TimeSeriesDataFrame.ITEMID).astype("category")
    df.static_features = None
    covariate_metadata = CovariateMetadata(known_covariates_cat=["covariate"], known_cat_cardinality={"covariate": 2})
    model = model_class(
        freq=df.freq,
        covariate_metadata=covariate_metadata,
        hyperparameters={"covariate_regressor": "LR", **dummy_hyperparameters},
    )
    with mock.patch("autogluon.timeseries.models.ZeroModel._fit") as mock_fit:
        try:
            model.fit(train_data=df)
        except AttributeError:
            # Ignore AttributeError produced by mock
            pass
    input_data_mean = mock_fit.call_args[1]["train_data"][model.target].groupby(TimeSeriesDataFrame.ITEMID).mean()
    assert np.allclose(input_data_mean, 0, atol=2.0)


def test_when_validation_fraction_is_set_then_tabular_model_uses_val_data(df_with_covariates_and_metadata):
    df, covariate_metadata = df_with_covariates_and_metadata
    regressor = GlobalCovariateRegressor(
        "LR", **MODEL_HPS, validation_fraction=0.1, covariate_metadata=covariate_metadata
    )
    with mock.patch("autogluon.tabular.models.LinearModel.fit") as mock_lr_fit:
        regressor.fit(df)
    assert mock_lr_fit.call_args[1]["X_val"] is not None
    assert mock_lr_fit.call_args[1]["y_val"] is not None


def test_when_validation_fraction_is_none_then_tabular_model_doesnt_use_val_data(df_with_covariates_and_metadata):
    df, covariate_metadata = df_with_covariates_and_metadata
    regressor = GlobalCovariateRegressor(
        "LR", **MODEL_HPS, validation_fraction=None, covariate_metadata=covariate_metadata
    )
    with mock.patch("autogluon.tabular.models.LinearModel.fit") as mock_lr_fit:
        regressor.fit(df)
    assert mock_lr_fit.call_args[1]["X_val"] is None
    assert mock_lr_fit.call_args[1]["y_val"] is None


def test_when_not_enough_time_is_left_to_predict_then_regressor_is_disabled(df_with_covariates_and_metadata):
    df, covariate_metadata = df_with_covariates_and_metadata
    regressor = GlobalCovariateRegressor("CAT", covariate_metadata=covariate_metadata)

    def predict_with_sleep(X, **kwargs):
        time.sleep(5)
        return np.zeros(len(X))

    with mock.patch("autogluon.tabular.models.CatBoostModel.predict", side_effect=predict_with_sleep):
        regressor.fit(df, time_limit=5)

    assert regressor.disabled


@pytest.mark.parametrize("use_fit_transform", [True, False])
@pytest.mark.parametrize("refit_during_predict", [True, False])
def test_when_regressor_is_disabled_then_data_is_not_modified_during_transform(
    df_with_covariates_and_metadata, use_fit_transform, refit_during_predict
):
    df, covariate_metadata = df_with_covariates_and_metadata
    regressor = GlobalCovariateRegressor(
        "LR", **MODEL_HPS, covariate_metadata=covariate_metadata, refit_during_predict=refit_during_predict
    )
    regressor.fit(df)
    regressor.disabled = True
    if use_fit_transform:
        df_out = regressor.fit_transform(df)
    else:
        df_out = regressor.transform(df)
    assert df_out is df


def test_when_all_features_are_constant_then_regressor_is_not_fit():
    df = DUMMY_TS_DATAFRAME.copy()
    df["cov1"] = [0.0] * len(df)
    df["cov2"] = ["a"] * len(df)
    df.static_features = pd.DataFrame(
        {"static1": [0.0] * df.num_items, "static2": ["a"] * df.num_items}, index=df.item_ids
    )
    df["cov2"] = df["cov2"].astype("category")
    df.static_features["static2"] = df.static_features["static2"].astype("category")
    covariate_metadata = CovariateMetadata(
        known_covariates_real=["cov1"],
        known_covariates_cat=["cov2"],
        static_features_real=["static1"],
        static_features_cat=["static2"],
        known_cat_cardinality={"cov2": 2},
        static_cat_cardinality={"static2": 2},
    )
    regressor = GlobalCovariateRegressor("LR", **MODEL_HPS, covariate_metadata=covariate_metadata)
    regressor.fit(df)
    assert regressor.disabled
    assert not regressor.model.is_fit()
