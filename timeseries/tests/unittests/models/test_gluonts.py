from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from gluonts.model.predictor import Predictor as GluonTSPredictor
from gluonts.torch.distributions import StudentTOutput

from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.gluonts import (
    DeepARModel,
    DLinearModel,
    PatchTSTModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
)
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from ..common import (
    DATAFRAME_WITH_COVARIATES,
    DATAFRAME_WITH_STATIC,
    DUMMY_TS_DATAFRAME,
    get_data_frame_with_covariates,
)

DUMMY_HYPERPARAMETERS = {"max_epochs": 1, "num_batches_per_epoch": 1}


def test_when_context_length_is_not_set_then_default_context_length_is_used(gluonts_model_class):
    data = DUMMY_TS_DATAFRAME
    model = gluonts_model_class(freq=data.freq)
    model.fit(train_data=data)
    estimator_init_args = model._get_estimator_init_args()
    default_context_length = model._get_default_hyperparameters()["context_length"]
    assert estimator_init_args["context_length"] == default_context_length


def test_when_context_length_is_set_then_provided_context_length_is_used(gluonts_model_class):
    data = DUMMY_TS_DATAFRAME
    model = gluonts_model_class(freq=data.freq, hyperparameters={"context_length": 1337})
    model.fit(train_data=data)
    estimator_init_args = model._get_estimator_init_args()
    assert estimator_init_args["context_length"] == 1337


@pytest.mark.parametrize("time_limit", [10, None])
def test_given_time_limit_when_fit_called_then_models_train_correctly(
    gluonts_model_class, time_limit, temp_model_path
):
    model = gluonts_model_class(
        path=temp_model_path,
        freq="h",
        prediction_length=5,
        hyperparameters={"max_epochs": 2},
    )

    assert not model.gts_predictor
    model.fit(train_data=DUMMY_TS_DATAFRAME, time_limit=time_limit)
    assert isinstance(model.gts_predictor, GluonTSPredictor)


# @flaky(max_runs=3)
# @pytest.mark.timeout(4)
@pytest.mark.skip(reason="Timeout spuriously fails in CI")
def test_given_low_time_limit_when_fit_called_then_model_training_does_not_exceed_time_limit(
    gluonts_model_class, temp_model_path
):
    model = gluonts_model_class(
        path=temp_model_path,
        freq="h",
        prediction_length=5,
        hyperparameters={"max_epochs": 20000},
    )

    assert not model.gts_predictor
    model.fit(train_data=DUMMY_TS_DATAFRAME, time_limit=2)
    assert isinstance(model.gts_predictor, GluonTSPredictor)


def test_when_models_saved_then_gluonts_predictors_can_be_loaded(gluonts_model_class, temp_model_path):
    model = gluonts_model_class(
        path=temp_model_path,
        freq="h",
        quantile_levels=[0.1, 0.9],
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


def test_when_static_features_present_then_they_are_passed_to_dataset(
    gluonts_model_with_static_features_class, df_with_static
):
    df, covariate_metadata = df_with_static
    model = gluonts_model_with_static_features_class(covariate_metadata=covariate_metadata, freq=df.freq)
    with mock.patch("autogluon.timeseries.models.gluonts.dataset.SimpleGluonTSDataset.__init__") as patch_dataset:
        try:
            model.fit(train_data=df)
        except TypeError:
            pass
        finally:
            call_kwargs = patch_dataset.call_args[1]
            feat_static_cat = call_kwargs["feat_static_cat"]
            feat_static_real = call_kwargs["feat_static_real"]
            assert feat_static_cat.dtype == "int64"
            assert feat_static_real.dtype == "float32"


def test_given_fit_with_static_features_when_predicting_then_static_features_are_used(
    gluonts_model_with_static_features_class, df_with_static
):
    df, covariate_metadata = df_with_static
    model = gluonts_model_with_static_features_class(covariate_metadata=covariate_metadata, freq=df.freq)
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


def test_when_static_features_present_then_model_attributes_set_correctly(
    gluonts_model_with_static_features_class, df_with_static
):
    df, covariate_metadata = df_with_static
    model = gluonts_model_with_static_features_class(covariate_metadata=covariate_metadata, freq=df.freq)
    model.fit(train_data=df)
    assert model.num_feat_static_cat > 0
    assert model.num_feat_static_real > 0
    assert len(model.feat_static_cat_cardinality) == model.num_feat_static_cat
    assert 1 <= model.feat_static_cat_cardinality[0] <= 4


def test_when_disable_static_features_set_to_true_then_static_features_are_not_used(
    gluonts_model_with_static_features_class, df_with_static
):
    df, covariate_metadata = df_with_static
    model = gluonts_model_with_static_features_class(
        hyperparameters={"disable_static_features": True}, covariate_metadata=covariate_metadata, freq=df.freq
    )
    with mock.patch("autogluon.timeseries.models.gluonts.dataset.SimpleGluonTSDataset.__init__") as patch_dataset:
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


def test_when_known_covariates_present_then_they_are_passed_to_dataset(
    gluonts_model_with_static_features_class, df_with_covariates
):
    df, covariate_metadata = df_with_covariates
    model = gluonts_model_with_static_features_class(covariate_metadata=covariate_metadata, freq=df.freq)
    with mock.patch("autogluon.timeseries.models.gluonts.dataset.SimpleGluonTSDataset.__init__") as patch_dataset:
        try:
            model.fit(train_data=df)
        except TypeError:
            pass
        finally:
            call_kwargs = patch_dataset.call_args[1]
            feat_dynamic_real = call_kwargs["feat_dynamic_real"]
            assert feat_dynamic_real.dtype == "float32"


def test_when_known_covariates_present_then_model_attributes_set_correctly(
    gluonts_model_with_known_covariates_class, df_with_covariates
):
    df, covariate_metadata = df_with_covariates
    model = gluonts_model_with_known_covariates_class(covariate_metadata=covariate_metadata, freq=df.freq)
    model.fit(train_data=df)
    assert model.num_feat_dynamic_real > 0


def test_when_known_covariates_present_for_predict_then_covariates_have_correct_shape(
    gluonts_model_with_known_covariates_class, df_with_covariates
):
    df, covariate_metadata = df_with_covariates
    prediction_length = 5
    past_data, known_covariates = df.get_model_inputs_for_scoring(
        prediction_length, covariate_metadata.known_covariates
    )
    model = gluonts_model_with_known_covariates_class(
        covariate_metadata=covariate_metadata, freq=df.freq, prediction_length=prediction_length
    )
    model.fit(train_data=past_data)
    for ts in model._to_gluonts_dataset(past_data, known_covariates=known_covariates):
        expected_length = len(ts["target"]) + prediction_length
        if model.supports_cat_covariates:
            assert ts["feat_dynamic_cat"].shape == (len(covariate_metadata.known_covariates_cat), expected_length)
            assert ts["feat_dynamic_real"].shape == (len(covariate_metadata.known_covariates_real), expected_length)
        else:
            num_onehot_columns = past_data[covariate_metadata.known_covariates_cat].nunique().sum()
            expected_num_feat_dynamic_real = len(covariate_metadata.known_covariates_real) + num_onehot_columns
            assert ts["feat_dynamic_real"].shape == (expected_num_feat_dynamic_real, expected_length)


def test_when_disable_known_covariates_set_to_true_then_known_covariates_are_not_used(
    gluonts_model_with_known_covariates_class, df_with_covariates
):
    df, covariate_metadata = df_with_covariates
    model = gluonts_model_with_known_covariates_class(
        hyperparameters={"disable_known_covariates": True}, covariate_metadata=covariate_metadata, freq=df.freq
    )
    with mock.patch("autogluon.timeseries.models.gluonts.dataset.SimpleGluonTSDataset.__init__") as patch_dataset:
        try:
            model.fit(train_data=df)
        except TypeError:
            pass
        finally:
            call_kwargs = patch_dataset.call_args[1]
            assert call_kwargs["feat_dynamic_real"] is None
            assert call_kwargs["feat_dynamic_cat"] is None


def test_when_static_and_dynamic_covariates_present_then_model_trains_normally(
    gluonts_model_with_known_covariates_and_static_features_class,
):
    dataframe_with_static_and_covariates = DATAFRAME_WITH_STATIC.copy()
    known_covariates_names = ["cov1", "cov2"]
    for col_name in known_covariates_names:
        dataframe_with_static_and_covariates[col_name] = np.random.normal(
            size=len(dataframe_with_static_and_covariates)
        )

    gen = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    df = gen.fit_transform(dataframe_with_static_and_covariates)

    model = gluonts_model_with_known_covariates_and_static_features_class(
        covariate_metadata=gen.covariate_metadata, freq=df.freq
    )
    model.fit(train_data=df)
    model.score_and_cache_oof(df)


@pytest.mark.parametrize("predict_batch_size", [30, 200])
def test_given_custom_predict_batch_size_then_predictor_uses_correct_batch_size(predict_batch_size):
    model = PatchTSTModel(
        hyperparameters={"predict_batch_size": predict_batch_size, **DUMMY_HYPERPARAMETERS},
        freq=DUMMY_TS_DATAFRAME.freq,
    )
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    assert model.gts_predictor.batch_size == predict_batch_size  # type: ignore


def catch_trainer_kwargs(model):
    with mock.patch("lightning.pytorch.Trainer") as mock_trainer:
        try:
            model.fit(train_data=DUMMY_TS_DATAFRAME, val_data=DUMMY_TS_DATAFRAME)
        except IsADirectoryError:
            # Training fails because Trainer is a mock object
            pass
    return mock_trainer.call_args[1]


def test_when_custom_callbacks_passed_via_trainer_kwargs_then_trainer_receives_them():
    from lightning.pytorch.callbacks import RichModelSummary

    callback = RichModelSummary()
    model = DLinearModel(
        hyperparameters={"trainer_kwargs": {"callbacks": [callback]}},
        freq=DUMMY_TS_DATAFRAME.freq,
    )
    received_trainer_kwargs = catch_trainer_kwargs(model)
    assert any(isinstance(cb, RichModelSummary) for cb in received_trainer_kwargs["callbacks"])


def test_when_early_stopping_patience_provided_then_early_stopping_callback_created():
    from lightning.pytorch.callbacks import EarlyStopping

    patience = 7
    model = SimpleFeedForwardModel(
        hyperparameters={"early_stopping_patience": patience, **DUMMY_HYPERPARAMETERS},
        freq=DUMMY_TS_DATAFRAME.freq,
    )
    received_trainer_kwargs = catch_trainer_kwargs(model)
    es_callbacks = [cb for cb in received_trainer_kwargs["callbacks"] if isinstance(cb, EarlyStopping)]
    assert len(es_callbacks) == 1
    assert es_callbacks[0].patience == patience


def test_when_early_stopping_patience_is_none_then_early_stopping_callback_not_created():
    from lightning.pytorch.callbacks import EarlyStopping

    model = SimpleFeedForwardModel(
        hyperparameters={"early_stopping_patience": None, **DUMMY_HYPERPARAMETERS},
        freq=DUMMY_TS_DATAFRAME.freq,
    )
    received_trainer_kwargs = catch_trainer_kwargs(model)
    es_callbacks = [cb for cb in received_trainer_kwargs["callbacks"] if isinstance(cb, EarlyStopping)]
    assert len(es_callbacks) == 0


def test_when_custom_trainer_kwargs_given_then_trainer_receives_them():
    trainer_kwargs = {"max_epochs": 5, "limit_train_batches": 100}
    model = PatchTSTModel(
        hyperparameters={"trainer_kwargs": trainer_kwargs, **DUMMY_HYPERPARAMETERS},
        freq=DUMMY_TS_DATAFRAME.freq,
    )
    received_trainer_kwargs = catch_trainer_kwargs(model)
    for k, v in trainer_kwargs.items():
        assert received_trainer_kwargs[k] == v


def test_when_model_finishes_training_then_logs_are_removed(temp_model_path):
    model = TemporalFusionTransformerModel(
        freq=DUMMY_TS_DATAFRAME.freq, path=temp_model_path, hyperparameters=DUMMY_HYPERPARAMETERS
    )
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    assert not (Path(model.path) / "lightning_logs").exists()


@pytest.mark.parametrize("keep_lightning_logs", [True, False])
def test_when_keep_lightning_logs_set_then_logs_are_not_removed(keep_lightning_logs, temp_model_path):
    model = DeepARModel(
        freq=DUMMY_TS_DATAFRAME.freq,
        path=temp_model_path,
        hyperparameters={"keep_lightning_logs": keep_lightning_logs, **DUMMY_HYPERPARAMETERS},
    )
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    assert (Path(model.path) / "lightning_logs").exists() == keep_lightning_logs


@pytest.mark.parametrize("known_covariates_real", [["known_real_1", "known_real_2"], []])
@pytest.mark.parametrize("past_covariates_real", [["past_real_1"], []])
@pytest.mark.parametrize("static_features_real", [["static_real_1", "static_real_2"], []])
def test_given_features_present_when_model_is_fit_then_feature_transformer_is_present(
    gluonts_model_class, temp_model_path, known_covariates_real, past_covariates_real, static_features_real
):
    known_covariates_names = known_covariates_real + ["known_cat_1"]
    feat_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = get_data_frame_with_covariates(
        covariates_cat=["known_cat_1"],
        covariates_real=known_covariates_real + past_covariates_real,
        static_features_real=static_features_real,
        static_features_cat=["static_cat_1"],
    )
    data = feat_generator.fit_transform(data)
    model = gluonts_model_class(
        freq=data.freq,
        path=temp_model_path,
        covariate_metadata=feat_generator.covariate_metadata,
    )
    model.fit(train_data=data, val_data=data)
    covariate_scaler = model.covariate_scaler

    if len(known_covariates_real) > 0 and model.supports_known_covariates:
        assert len(covariate_scaler._column_transformers["known"].feature_names_in_) > 0
    else:
        assert "known" not in covariate_scaler._column_transformers

    if len(past_covariates_real) > 0 and model.supports_past_covariates:
        assert len(covariate_scaler._column_transformers["past"].feature_names_in_) > 0
    else:
        assert "past" not in covariate_scaler._column_transformers

    if len(static_features_real) > 0 and model.supports_static_features:
        assert len(covariate_scaler._column_transformers["static"].feature_names_in_) > 0
    else:
        assert "static" not in covariate_scaler._column_transformers


def test_when_model_is_initialized_then_covariate_scaler_is_created(gluonts_model_class, df_with_covariates):
    df, covariate_metadata = df_with_covariates
    model = gluonts_model_class(freq=df.freq, covariate_metadata=covariate_metadata)
    model.fit(train_data=df, time_limit=1)
    assert model.covariate_scaler is not None


def test_when_distr_output_passed_to_tft_then_model_can_fit_and_predict():
    data = DUMMY_TS_DATAFRAME.copy()
    quantile_levels = [0.15, 0.4, 0.94]
    model = TemporalFusionTransformerModel(
        freq=data.freq,
        prediction_length=4,
        quantile_levels=quantile_levels,
        hyperparameters={"distr_output": StudentTOutput(), **DUMMY_HYPERPARAMETERS},
    )
    model.fit(train_data=data)
    predictions = model.predict(data)
    assert isinstance(predictions, TimeSeriesDataFrame)
    assert set(predictions.columns) == set(["mean"] + [str(q) for q in quantile_levels])
