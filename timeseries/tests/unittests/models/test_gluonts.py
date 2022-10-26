from functools import partial
from unittest import mock

import pytest
from gluonts.model.predictor import Predictor as GluonTSPredictor
from gluonts.model.prophet import PROPHET_IS_INSTALLED
from gluonts.mx.model.seq2seq import MQRNNEstimator
from gluonts.mx.model.transformer import TransformerEstimator

import autogluon.core as ag
from autogluon.timeseries.models.gluonts import (  # MQRNNModel,; TransformerModel,
    DeepARModel,
    GenericGluonTSModel,
    MQCNNModel,
    ProphetModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
)
from autogluon.timeseries.models.gluonts.models import GenericGluonTSModelFactory
from autogluon.timeseries.utils.features import ContinuousAndCategoricalFeatureGenerator

from ..common import DUMMY_TS_DATAFRAME, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME_WITH_STATIC

TESTABLE_MODELS = [
    DeepARModel,
    MQCNNModel,
    # MQRNNModel,
    SimpleFeedForwardModel,
    # TransformerModel,
    partial(GenericGluonTSModel, gluonts_estimator_class=MQRNNEstimator),  # partial constructor for generic model
    GenericGluonTSModelFactory(TransformerEstimator),
    TemporalFusionTransformerModel,
]

MODELS_WITH_STATIC_FEATURES = [
    DeepARModel,
    MQCNNModel,
]

# if PROPHET_IS_INSTALLED:
#     TESTABLE_MODELS += [ProphetModel]


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
        hyperparameters={
            "epochs": 1,
        },
    )
    model.fit(
        train_data=DUMMY_TS_DATAFRAME,
    )
    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert model.gluonts_estimator_class is loaded_model.gluonts_estimator_class
    assert loaded_model.gts_predictor == model.gts_predictor


@pytest.mark.skipif(
    not PROPHET_IS_INSTALLED,
    reason="Prophet is not installed. Run `pip install prophet`",
)
@pytest.mark.parametrize("growth", ["linear", "logistic"])
@pytest.mark.parametrize("n_changepoints", [3, 5])
def test_when_fit_called_on_prophet_then_hyperparameters_are_passed_to_underlying_model(
    growth, n_changepoints, temp_model_path
):
    model = ProphetModel(
        path=temp_model_path,
        freq="H",
        prediction_length=4,
        hyperparameters={"growth": growth, "n_changepoints": n_changepoints},
    )

    model.fit(train_data=DUMMY_TS_DATAFRAME)

    assert model.gts_predictor.prophet_params.get("growth") == growth  # noqa
    assert model.gts_predictor.prophet_params.get("n_changepoints") == n_changepoints  # noqa  # noqa  # noqa


@pytest.mark.skipif(
    not PROPHET_IS_INSTALLED,
    reason="Prophet is not installed. Run `pip install prophet`",
)
@pytest.mark.parametrize("growth", ["linear", "logistic"])
@pytest.mark.parametrize("n_changepoints", [3, 5])
def test_when_prophet_model_saved_then_prophet_parameters_are_loaded(growth, n_changepoints, temp_model_path):
    model = ProphetModel(
        path=temp_model_path,
        freq="H",
        quantile_levels=[0.1, 0.9],
        hyperparameters={"growth": growth, "n_changepoints": n_changepoints},
    )
    model.fit(
        train_data=DUMMY_TS_DATAFRAME,
    )
    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert loaded_model.gts_predictor.prophet_params.get("growth") == growth  # noqa
    assert loaded_model.gts_predictor.prophet_params.get("n_changepoints") == n_changepoints  # noqa  # noqa


@pytest.mark.skipif(
    not PROPHET_IS_INSTALLED,
    reason="Prophet is not installed. Run `pip install prophet`",
)
def test_when_hyperparameter_tune_called_on_prophet_then_hyperparameters_are_passed_to_underlying_model(
    temp_model_path,
):
    model = ProphetModel(
        path=temp_model_path,
        freq="H",
        prediction_length=4,
        hyperparameters={"growth": "linear", "n_changepoints": ag.Int(3, 4)},
    )
    hyperparameter_tune_kwargs = "auto"

    models, results = model.hyperparameter_tune(
        time_limit=100,
        train_data=DUMMY_TS_DATAFRAME,
        val_data=DUMMY_TS_DATAFRAME,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )

    assert len(results["config_history"]) == 2
    assert results["config_history"][0]["n_changepoints"] == 3
    assert results["config_history"][1]["n_changepoints"] == 4

    assert all(c["growth"] == "linear" for c in results["config_history"].values())


@pytest.mark.parametrize(
    "quantiles, should_fail",
    [
        ([0.1, 0.5, 0.3, 0.9], False),
        ([0.9], False),
        ([0.1, 0.5, 0.55], True),
    ],
)
def test_when_tft_quantiles_are_not_deciles_then_value_error_is_raised(temp_model_path, quantiles, should_fail):
    model = TemporalFusionTransformerModel(
        path=temp_model_path,
        freq=DUMMY_TS_DATAFRAME.freq,
        prediction_length=4,
        quantile_levels=quantiles,
        hyperparameters={"epochs": 1},
    )
    if should_fail:
        with pytest.raises(ValueError, match="quantile_levels are a subset of"):
            model.fit(train_data=DUMMY_TS_DATAFRAME)
            model.predict(DUMMY_TS_DATAFRAME)
    else:
        model.fit(train_data=DUMMY_TS_DATAFRAME)
        model.predict(DUMMY_TS_DATAFRAME)


@pytest.mark.parametrize("quantiles", [[0.1, 0.5, 0.9], [0.2, 0.3, 0.7]])
def test_when_tft_quantiles_are_deciles_then_forecast_contains_correct_quantiles(temp_model_path, quantiles):
    # TFT is not covered by the quantiles test in test_models.py
    model = TemporalFusionTransformerModel(
        path=temp_model_path,
        freq=DUMMY_TS_DATAFRAME.freq,
        prediction_length=4,
        quantile_levels=quantiles,
        hyperparameters={"epochs": 1},
    )
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    predictions = model.predict(data=DUMMY_TS_DATAFRAME)
    assert "mean" in predictions.columns
    assert all(str(q) in predictions.columns for q in quantiles)


@pytest.mark.parametrize("model_class", MODELS_WITH_STATIC_FEATURES)
def test_when_static_features_are_available_then_they_are_used_by_model(model_class):
    model = model_class(hyperparameters={"epochs": 1})
    model.fit(train_data=DUMMY_VARIABLE_LENGTH_TS_DATAFRAME_WITH_STATIC)
    assert model.use_feat_static_cat == True
    assert model.use_feat_static_real == True
    assert len(model.feat_static_cat_cardinality) == 1


@pytest.fixture(scope="module")
def df_with_static():
    feature_pipeline = ContinuousAndCategoricalFeatureGenerator(verbosity=0)
    df = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME_WITH_STATIC.copy(deep=False)
    df.static_features = feature_pipeline.fit_transform(df.static_features)
    return df


@pytest.mark.parametrize("model_class", MODELS_WITH_STATIC_FEATURES)
def test_when_static_features_present_then_they_are_passed_to_dataset(model_class, df_with_static):
    model = model_class()
    with mock.patch(
        "autogluon.timeseries.models.gluonts.abstract_gluonts.SimpleGluonTSDataset.__init__"
    ) as patch_dataset:
        try:
            model.fit(train_data=df_with_static)
        except TypeError:
            call_kwargs = patch_dataset.call_args[1]
            feat_static_cat = call_kwargs["feat_static_cat"]
            feat_static_real = call_kwargs["feat_static_real"]
            assert (feat_static_cat.dtypes == "category").all()
            assert (feat_static_real.dtypes == "float").all()


@pytest.mark.parametrize("model_class", MODELS_WITH_STATIC_FEATURES)
def test_when_static_features_present_then_model_attributes_set_correctly(model_class, df_with_static):
    model = model_class(hyperparameters={"epochs": 1, "num_batches_per_epoch": 1})
    model.fit(train_data=df_with_static)
    assert model.use_feat_static_cat
    assert model.use_feat_static_real
    assert len(model.feat_static_cat_cardinality) == 1


@pytest.mark.parametrize("model_class", MODELS_WITH_STATIC_FEATURES)
def test_when_disable_static_features_set_to_true_then_static_features_are_not_used(model_class, df_with_static):
    model = model_class(hyperparameters={"disable_static_features": True})
    with mock.patch(
        "autogluon.timeseries.models.gluonts.abstract_gluonts.SimpleGluonTSDataset.__init__"
    ) as patch_dataset:
        try:
            model.fit(train_data=df_with_static)
        except TypeError:
            call_kwargs = patch_dataset.call_args[1]
            feat_static_cat = call_kwargs["feat_static_cat"]
            feat_static_real = call_kwargs["feat_static_real"]
            assert feat_static_cat is None
            assert feat_static_real is None
