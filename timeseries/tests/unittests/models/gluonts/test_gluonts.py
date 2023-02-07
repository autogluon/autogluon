from unittest import mock

import numpy as np
import pytest
from gluonts.model.predictor import Predictor as GluonTSPredictor

import autogluon.timeseries as agts
from autogluon.timeseries.models.gluonts import DeepARModel, SimpleFeedForwardModel, TemporalFusionTransformerModel
from autogluon.timeseries.models.gluonts.torch.models import AbstractGluonTSPyTorchModel
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from ...common import DATAFRAME_WITH_COVARIATES, DATAFRAME_WITH_STATIC, DUMMY_TS_DATAFRAME

if agts.MXNET_INSTALLED:
    from .mx.test_mx import (
        TESTABLE_MX_MODELS,
        TESTABLE_MX_MODELS_WITH_KNOWN_COVARIATES,
        TESTABLE_MX_MODELS_WITH_STATIC_FEATURES,
    )
else:
    TESTABLE_MX_MODELS = []
    TESTABLE_MX_MODELS_WITH_STATIC_FEATURES = []
    TESTABLE_MX_MODELS_WITH_KNOWN_COVARIATES = []

MODELS_WITH_STATIC_FEATURES = [DeepARModel, TemporalFusionTransformerModel] + TESTABLE_MX_MODELS_WITH_STATIC_FEATURES
MODELS_WITH_KNOWN_COVARIATES = [DeepARModel, TemporalFusionTransformerModel] + TESTABLE_MX_MODELS_WITH_KNOWN_COVARIATES
MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES = [
    m for m in MODELS_WITH_STATIC_FEATURES if m in MODELS_WITH_KNOWN_COVARIATES
]
TESTABLE_PYTORCH_MODELS = [DeepARModel, SimpleFeedForwardModel, TemporalFusionTransformerModel]
TESTABLE_MODELS = TESTABLE_MX_MODELS + TESTABLE_PYTORCH_MODELS

DUMMY_HYPERPARAMETERS = {"epochs": 1, "num_batches_per_epoch": 1}


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("time_limit", [10, None])
def test_given_time_limit_when_fit_called_then_models_train_correctly(model_class, time_limit, temp_model_path):
    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=5,
        hyperparameters={"epochs": 2},
    )

    assert not model.gts_predictor
    model.fit(train_data=DUMMY_TS_DATAFRAME, time_limit=time_limit)
    assert isinstance(model.gts_predictor, GluonTSPredictor)


# @flaky(max_runs=3)
# @pytest.mark.timeout(4)
@pytest.mark.skip(reason="Timeout spuriously fails in CI")
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_low_time_limit_when_fit_called_then_model_training_does_not_exceed_time_limit(
    model_class, temp_model_path
):
    model = model_class(
        path=temp_model_path,
        freq="H",
        prediction_length=5,
        hyperparameters={"epochs": 20000},
    )

    assert not model.gts_predictor
    model.fit(train_data=DUMMY_TS_DATAFRAME, time_limit=2)
    assert isinstance(model.gts_predictor, GluonTSPredictor)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_models_saved_then_gluonts_predictors_can_be_loaded(model_class, temp_model_path):
    model = model_class(
        path=temp_model_path,
        freq="H",
        quantile_levels=[0.1, 0.9],
        hyperparameters=DUMMY_HYPERPARAMETERS,
    )
    model.fit(
        train_data=DUMMY_TS_DATAFRAME,
    )
    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert model.gluonts_estimator_class is loaded_model.gluonts_estimator_class
    if isinstance(model, AbstractGluonTSPyTorchModel):
        assert loaded_model.gts_predictor.to(model.gts_predictor.device) == model.gts_predictor
    else:
        assert loaded_model.gts_predictor == model.gts_predictor


@pytest.fixture(scope="module")
def df_with_static():
    feature_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=[])
    df = DATAFRAME_WITH_STATIC.copy(deep=False)
    df = feature_generator.fit_transform(df)
    return df, feature_generator.covariate_metadata


@pytest.fixture(scope="module")
def df_with_covariates():
    known_covariates_names = [col for col in DATAFRAME_WITH_COVARIATES.columns if col != "target"]
    feature_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    df = DATAFRAME_WITH_COVARIATES.copy(deep=False)
    df = feature_generator.fit_transform(df)
    return df, feature_generator.covariate_metadata


@pytest.mark.parametrize("model_class", MODELS_WITH_STATIC_FEATURES)
def test_when_static_features_present_then_they_are_passed_to_dataset(model_class, df_with_static):
    df, metadata = df_with_static
    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata)
    with mock.patch(
        "autogluon.timeseries.models.gluonts.abstract_gluonts.SimpleGluonTSDataset.__init__"
    ) as patch_dataset:
        try:
            model.fit(train_data=df)
        except TypeError:
            pass
        finally:
            call_kwargs = patch_dataset.call_args[1]
            feat_static_cat = call_kwargs["feat_static_cat"]
            feat_static_real = call_kwargs["feat_static_real"]
            assert (feat_static_cat.dtypes == "category").all()
            assert (feat_static_real.dtypes == "float").all()


@pytest.mark.parametrize("model_class", MODELS_WITH_STATIC_FEATURES)
def test_given_fit_with_static_features_when_predicting_then_static_features_are_used(model_class, df_with_static):
    df, metadata = df_with_static
    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata)
    model.fit(train_data=df)
    if model.name.endswith("MXNet"):
        predictor_method = "gluonts.mx.model.predictor.RepresentableBlockPredictor.predict"
    else:
        predictor_method = "gluonts.torch.model.predictor.PyTorchPredictor.predict"
    with mock.patch(predictor_method) as mock_predict:
        try:
            model.predict(df)
        except IndexError:  # expected because of mock
            pass
        finally:
            gluonts_dataset = mock_predict.call_args[1]["dataset"]
            item = next(iter(gluonts_dataset))
            assert item["feat_static_cat"].shape == (1,)
            assert item["feat_static_real"].shape == (2,)


@pytest.mark.parametrize("model_class", MODELS_WITH_STATIC_FEATURES)
def test_when_static_features_present_then_model_attributes_set_correctly(model_class, df_with_static):
    df, metadata = df_with_static
    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata)
    model.fit(train_data=df)
    assert model.num_feat_static_cat > 0
    assert model.num_feat_static_real > 0
    assert len(model.feat_static_cat_cardinality) == model.num_feat_static_cat
    assert 1 <= model.feat_static_cat_cardinality[0] <= 4


@pytest.mark.parametrize("model_class", MODELS_WITH_STATIC_FEATURES)
def test_when_disable_static_features_set_to_true_then_static_features_are_not_used(model_class, df_with_static):
    df, metadata = df_with_static
    model = model_class(hyperparameters={**DUMMY_HYPERPARAMETERS, "disable_static_features": True}, metadata=metadata)
    with mock.patch(
        "autogluon.timeseries.models.gluonts.abstract_gluonts.SimpleGluonTSDataset.__init__"
    ) as patch_dataset:
        try:
            model.fit(train_data=df)
        except TypeError:
            pass
        finally:
            call_kwargs = patch_dataset.call_args[1]
            feat_static_cat = call_kwargs["feat_static_cat"]
            feat_static_real = call_kwargs["feat_static_real"]
            assert feat_static_cat is None
            assert feat_static_real is None


@pytest.mark.parametrize("model_class", MODELS_WITH_KNOWN_COVARIATES)
def test_when_known_covariates_present_then_they_are_passed_to_dataset(model_class, df_with_covariates):
    df, metadata = df_with_covariates
    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata)
    with mock.patch(
        "autogluon.timeseries.models.gluonts.abstract_gluonts.SimpleGluonTSDataset.__init__"
    ) as patch_dataset:
        try:
            model.fit(train_data=df)
        except TypeError:
            pass
        finally:
            call_kwargs = patch_dataset.call_args[1]
            feat_dynamic_real = call_kwargs["feat_dynamic_real"]
            assert (feat_dynamic_real.dtypes == "float").all()


@pytest.mark.parametrize("model_class", MODELS_WITH_KNOWN_COVARIATES)
def test_when_known_covariates_present_then_model_attributes_set_correctly(model_class, df_with_covariates):
    df, metadata = df_with_covariates
    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata)
    model.fit(train_data=df)
    assert model.num_feat_dynamic_real > 0


@pytest.mark.parametrize("model_class", MODELS_WITH_KNOWN_COVARIATES)
def test_when_disable_known_covariates_set_to_true_then_known_covariates_are_not_used(model_class, df_with_covariates):
    df, metadata = df_with_covariates
    model = model_class(hyperparameters={**DUMMY_HYPERPARAMETERS, "disable_known_covariates": True}, metadata=metadata)
    with mock.patch(
        "autogluon.timeseries.models.gluonts.abstract_gluonts.SimpleGluonTSDataset.__init__"
    ) as patch_dataset:
        try:
            model.fit(train_data=df)
        except TypeError:
            pass
        finally:
            call_kwargs = patch_dataset.call_args[1]
            feat_dynamic_real = call_kwargs["feat_dynamic_real"]
            assert feat_dynamic_real is None


@pytest.mark.parametrize("model_class", MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES)
def test_when_static_and_dynamic_covariates_present_then_model_trains_normally(model_class):
    dataframe_with_static_and_covariates = DATAFRAME_WITH_STATIC.copy()
    known_covariates_names = ["cov1", "cov2"]
    for col_name in known_covariates_names:
        dataframe_with_static_and_covariates[col_name] = np.random.normal(
            size=len(dataframe_with_static_and_covariates)
        )

    gen = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    df = gen.fit_transform(dataframe_with_static_and_covariates)

    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=gen.covariate_metadata)
    model.fit(train_data=df)
    model.predict_for_scoring(df)
