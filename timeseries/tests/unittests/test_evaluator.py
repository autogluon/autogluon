import numpy as np
import pytest

from gluonts.evaluation import Evaluator as GluonTSEvaluator

from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.evaluator import TimeSeriesEvaluator

from .common import DUMMY_TS_DATAFRAME


@pytest.mark.parametrize("metric_name", ["mean_wQuantileLoss", "MAPE", "sMAPE"])
def test_when_given_forecasts_then_equal_to_gluonts(metric_name):
    pred = TimeSeriesPredictor(prediction_length=2, verbosity=4)

    pred.fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={
            "DeepAR": dict(epochs=2),
        },
    )

    model = pred._trainer.load_model("DeepAR")
    forecast_iter, timeseries_iter = model._predict_for_scoring(DUMMY_TS_DATAFRAME)
    forecast_df = model._gluonts_forecasts_to_data_frame(
        forecast_iter, quantile_levels=model.quantile_levels
    )

    ag_evaluator = TimeSeriesEvaluator(eval_metric=metric_name, prediction_length=2)
    ag_value = ag_evaluator(DUMMY_TS_DATAFRAME, forecast_df)

    gts_evaluator = GluonTSEvaluator()
    gts_results, _ = gts_evaluator(
        ts_iterator=timeseries_iter, fcst_iterator=forecast_iter
    )

    assert np.isclose(gts_results[metric_name], ag_value, atol=1e-5)
