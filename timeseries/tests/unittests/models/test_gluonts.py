from unittest import mock

import numpy as np
import pytest
from gluonts.model.predictor import Predictor as GluonTSPredictor

from autogluon.timeseries.models.gluonts import (
    DeepARModel,
    DLinearModel,
    PatchTSTModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
)
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from ..common import DATAFRAME_WITH_COVARIATES, DATAFRAME_WITH_STATIC, DUMMY_TS_DATAFRAME

MODELS_WITH_STATIC_FEATURES = [DeepARModel, TemporalFusionTransformerModel]
MODELS_WITH_KNOWN_COVARIATES = [DeepARModel, TemporalFusionTransformerModel]
MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES = [
    m for m in MODELS_WITH_STATIC_FEATURES if m in MODELS_WITH_KNOWN_COVARIATES
]
TESTABLE_MODELS = [
    DeepARModel,
    DLinearModel,
    PatchTSTModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
]


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

    assert model._get_estimator_class() is loaded_model._get_estimator_class()
    assert loaded_model.gts_predictor.to(model.gts_predictor.device) == model.gts_predictor


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
    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata, freq=df.freq)
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
    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata, freq=df.freq)
    model.fit(train_data=df)
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
    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata, freq=df.freq)
    model.fit(train_data=df)
    assert model.num_feat_static_cat > 0
    assert model.num_feat_static_real > 0
    assert len(model.feat_static_cat_cardinality) == model.num_feat_static_cat
    assert 1 <= model.feat_static_cat_cardinality[0] <= 4


@pytest.mark.parametrize("model_class", MODELS_WITH_STATIC_FEATURES)
def test_when_disable_static_features_set_to_true_then_static_features_are_not_used(model_class, df_with_static):
    df, metadata = df_with_static
    model = model_class(
        hyperparameters={**DUMMY_HYPERPARAMETERS, "disable_static_features": True}, metadata=metadata, freq=df.freq
    )
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
    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata, freq=df.freq)
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
    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata, freq=df.freq)
    model.fit(train_data=df)
    assert model.num_feat_dynamic_real > 0


@pytest.mark.parametrize("model_class", MODELS_WITH_KNOWN_COVARIATES)
def test_when_known_covariates_present_for_predict_then_covariates_have_correct_shape(model_class, df_with_covariates):
    df, metadata = df_with_covariates
    prediction_length = 5
    past_data, known_covariates = df.get_model_inputs_for_scoring(prediction_length, metadata.known_covariates_real)
    model = model_class(
        hyperparameters=DUMMY_HYPERPARAMETERS, metadata=metadata, freq=df.freq, prediction_length=prediction_length
    )
    model.fit(train_data=past_data)
    for ts in model._to_gluonts_dataset(past_data, known_covariates=known_covariates):
        expected_length = len(ts["target"]) + prediction_length
        assert ts["feat_dynamic_real"].shape == (len(metadata.known_covariates_real), expected_length)


@pytest.mark.parametrize("model_class", MODELS_WITH_KNOWN_COVARIATES)
def test_when_disable_known_covariates_set_to_true_then_known_covariates_are_not_used(model_class, df_with_covariates):
    df, metadata = df_with_covariates
    model = model_class(
        hyperparameters={**DUMMY_HYPERPARAMETERS, "disable_known_covariates": True}, metadata=metadata, freq=df.freq
    )
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

    model = model_class(hyperparameters=DUMMY_HYPERPARAMETERS, metadata=gen.covariate_metadata, freq=df.freq)
    model.fit(train_data=df)
    model.score_and_cache_oof(df)


@pytest.mark.parametrize("predict_batch_size", [30, 200])
def test_given_custom_predict_batch_size_then_predictor_uses_correct_batch_size(predict_batch_size):
    model = PatchTSTModel(hyperparameters={"predict_batch_size": predict_batch_size, **DUMMY_HYPERPARAMETERS})
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    assert model.gts_predictor.batch_size == predict_batch_size


def catch_trainer_kwargs(model):
    with mock.patch("pytorch_lightning.Trainer") as mock_trainer:
        try:
            model.fit(train_data=DUMMY_TS_DATAFRAME, val_data=DUMMY_TS_DATAFRAME)
        except IsADirectoryError:
            # Training fails because Trainer is a mock object
            pass
    return mock_trainer.call_args[1]


def test_when_custom_callbacks_passed_via_trainer_kwargs_then_trainer_receives_them():
    from pytorch_lightning.callbacks import RichModelSummary

    callback = RichModelSummary()
    model = DLinearModel(hyperparameters={"trainer_kwargs": {"callbacks": [callback]}, **DUMMY_HYPERPARAMETERS})
    received_trainer_kwargs = catch_trainer_kwargs(model)
    assert any(isinstance(cb, RichModelSummary) for cb in received_trainer_kwargs["callbacks"])


def test_when_early_stopping_patience_provided_then_early_stopping_callback_created():
    from pytorch_lightning.callbacks import EarlyStopping

    patience = 7
    model = SimpleFeedForwardModel(hyperparameters={"early_stopping_patience": patience, **DUMMY_HYPERPARAMETERS})
    received_trainer_kwargs = catch_trainer_kwargs(model)
    es_callbacks = [cb for cb in received_trainer_kwargs["callbacks"] if isinstance(cb, EarlyStopping)]
    assert len(es_callbacks) == 1
    assert es_callbacks[0].patience == patience


def test_when_early_stopping_patience_is_none_then_early_stopping_callback_not_created():
    from pytorch_lightning.callbacks import EarlyStopping

    model = SimpleFeedForwardModel(hyperparameters={"early_stopping_patience": None, **DUMMY_HYPERPARAMETERS})
    received_trainer_kwargs = catch_trainer_kwargs(model)
    es_callbacks = [cb for cb in received_trainer_kwargs["callbacks"] if isinstance(cb, EarlyStopping)]
    assert len(es_callbacks) == 0


def test_when_custom_trainer_kwargs_given_then_trainer_receives_them():
    trainer_kwargs = {"max_epochs": 5, "limit_train_batches": 100}
    model = PatchTSTModel(hyperparameters={"trainer_kwargs": trainer_kwargs, **DUMMY_HYPERPARAMETERS})
    received_trainer_kwargs = catch_trainer_kwargs(model)
    for k, v in trainer_kwargs.items():
        assert received_trainer_kwargs[k] == v
