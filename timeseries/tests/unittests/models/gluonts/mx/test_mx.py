from functools import partial

import pytest

from autogluon.timeseries import MXNET_INSTALLED
from autogluon.timeseries.models.gluonts.mx.models import GenericGluonTSMXNetModelFactory

from ....common import DUMMY_TS_DATAFRAME

if not MXNET_INSTALLED:
    TESTABLE_MX_MODELS = []
    TESTABLE_MX_MODELS_WITH_STATIC_FEATURES = []
else:
    from gluonts.mx.model.seq2seq import MQRNNEstimator
    from gluonts.mx.model.transformer import TransformerEstimator

    from autogluon.timeseries.models.gluonts.mx import(
        DeepARMXNetModel,
        GenericGluonTSMXNetModel,
        MQCNNMXNetModel,
        SimpleFeedForwardMXNetModel,
        TemporalFusionTransformerMXNetModel,
    )
    TESTABLE_MX_MODELS = [
        DeepARMXNetModel,
        MQCNNMXNetModel,
        # MQRNNModel,
        SimpleFeedForwardMXNetModel,
        # TransformerModel,
        partial(GenericGluonTSMXNetModel, gluonts_estimator_class=MQRNNEstimator),  # partial constructor for generic model
        GenericGluonTSMXNetModelFactory(TransformerEstimator),
        TemporalFusionTransformerMXNetModel,
    ]
    TESTABLE_MX_MODELS_WITH_STATIC_FEATURES = [
        DeepARMXNetModel,
        MQCNNMXNetModel,
    ]


@pytest.mark.parametrize(
    "quantiles, should_fail",
    [
        ([0.1, 0.5, 0.3, 0.9], False),
        ([0.9], False),
        ([0.1, 0.5, 0.55], True),
    ],
)
def test_when_tft_quantiles_are_not_deciles_then_value_error_is_raised(temp_model_path, quantiles, should_fail):
    model = TemporalFusionTransformerMXNetModel(
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
    model = TemporalFusionTransformerMXNetModel(
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
