import pytest

from autogluon.timeseries.models import ETSModel
from autogluon.timeseries.models.ensemble.greedy_ensemble import TimeSeriesGreedyEnsemble

from ..common import DUMMY_TS_DATAFRAME


def test_when_some_base_models_fail_during_prediction_then_ensemble_can_still_predict():
    ets = ETSModel(prediction_length=1, freq=DUMMY_TS_DATAFRAME.freq)
    ets.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters={"maxiter": 1, "seasonal": None})
    ensemble = TimeSeriesGreedyEnsemble(name="WeightedEnsemble")
    ensemble.model_to_weight = {"ARIMA": 0.5, "ETS": 0.5}
    ets_preds = ets.predict(DUMMY_TS_DATAFRAME)
    ensemble_preds = ensemble.predict(data={"ARIMA": None, "ETS": ets_preds})
    assert (ets_preds.values == ensemble_preds.values).all()


def test_when_all_base_models_fail_during_prediction_then_ensemble_raises_runtime_error():
    ensemble = TimeSeriesGreedyEnsemble(name="WeightedEnsemble")
    ensemble.model_to_weight = {"ARIMA": 0.5, "ETS": 0.5}
    with pytest.raises(RuntimeError):
        ensemble.predict(data={"ARIMA": None, "ETS": None})
