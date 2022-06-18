import numpy as np
import pytest

from gluonts.evaluation import Evaluator as GluonTSEvaluator
from gluonts.model.forecast import SampleForecast

from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.evaluator import TimeSeriesEvaluator
from autogluon.timeseries.utils.warning_filters import evaluator_warning_filter

from .common import DUMMY_TS_DATAFRAME


GLUONTS_PARITY_METRICS = ["mean_wQuantileLoss", "MAPE", "sMAPE", "MSE", "RMSE"]


@pytest.fixture(scope="module")
def deepar_trained():
    pred = TimeSeriesPredictor(prediction_length=2, verbosity=4)
    pred.fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={
            "DeepAR": dict(epochs=2),
        },
    )
    return pred._trainer.load_model("DeepAR")


@pytest.fixture(scope="module")
def deepar_trained_zero_data():
    pred = TimeSeriesPredictor(prediction_length=2, verbosity=4)

    data = DUMMY_TS_DATAFRAME.copy() * 0

    pred.fit(
        data,
        hyperparameters={
            "DeepAR": dict(epochs=2),
        },
    )
    return pred._trainer.load_model("DeepAR")


@pytest.mark.parametrize("metric_name", GLUONTS_PARITY_METRICS)
def test_when_given_learned_model_when_evaluator_called_then_output_equal_to_gluonts(
    metric_name, deepar_trained
):
    with evaluator_warning_filter():
        forecast_iter, timeseries_iter = deepar_trained._predict_for_scoring(
            DUMMY_TS_DATAFRAME
        )
        forecast_df = deepar_trained._gluonts_forecasts_to_data_frame(
            forecast_iter, quantile_levels=deepar_trained.quantile_levels
        )

        ag_evaluator = TimeSeriesEvaluator(eval_metric=metric_name, prediction_length=2)
        ag_value = ag_evaluator(DUMMY_TS_DATAFRAME, forecast_df)

        gts_evaluator = GluonTSEvaluator()
        gts_results, _ = gts_evaluator(
            ts_iterator=timeseries_iter, fcst_iterator=forecast_iter
        )

        assert np.isclose(gts_results[metric_name], ag_value, atol=1e-5)


@pytest.mark.parametrize("metric_name", GLUONTS_PARITY_METRICS)
def test_when_given_all_zero_data_when_evaluator_called_then_output_equal_to_gluonts(
    metric_name, deepar_trained_zero_data
):
    model = deepar_trained_zero_data
    data = DUMMY_TS_DATAFRAME.copy() * 0

    with evaluator_warning_filter():
        forecast_iter, timeseries_iter = model._predict_for_scoring(data)
        forecast_df = model._gluonts_forecasts_to_data_frame(
            forecast_iter, quantile_levels=model.quantile_levels
        )

        ag_evaluator = TimeSeriesEvaluator(eval_metric=metric_name, prediction_length=2)
        ag_value = ag_evaluator(data, forecast_df)

        gts_evaluator = GluonTSEvaluator()
        gts_results, _ = gts_evaluator(
            ts_iterator=timeseries_iter, fcst_iterator=forecast_iter
        )

        assert np.isclose(gts_results[metric_name], ag_value, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("metric_name", GLUONTS_PARITY_METRICS)
def test_when_given_zero_forecasts_when_evaluator_called_then_output_equal_to_gluonts(
    metric_name, deepar_trained
):
    with evaluator_warning_filter():
        forecast_iter, timeseries_iter = deepar_trained._predict_for_scoring(
            DUMMY_TS_DATAFRAME
        )

        zero_forecast_list = []
        for s in forecast_iter:
            zero_forecast_list.append(
                SampleForecast(
                    samples=np.zeros_like(s.samples),
                    start_date=s.start_date,
                    freq=s.freq,
                    item_id=s.item_id,
                )
            )

        forecast_df = deepar_trained._gluonts_forecasts_to_data_frame(
            zero_forecast_list, quantile_levels=deepar_trained.quantile_levels
        )

        ag_evaluator = TimeSeriesEvaluator(eval_metric=metric_name, prediction_length=2)
        ag_value = ag_evaluator(DUMMY_TS_DATAFRAME, forecast_df)

        gts_evaluator = GluonTSEvaluator()
        gts_results, _ = gts_evaluator(
            ts_iterator=timeseries_iter, fcst_iterator=zero_forecast_list
        )

        assert np.isclose(gts_results[metric_name], ag_value, atol=1e-5)
