import numpy as np
import pandas as pd
import pytest
from gluonts.evaluation import Evaluator as GluonTSEvaluator
from gluonts.model.forecast import QuantileForecast

from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.evaluator import TimeSeriesEvaluator, in_sample_abs_seasonal_error
from autogluon.timeseries.models.gluonts.abstract_gluonts import AbstractGluonTSModel

from .common import DUMMY_TS_DATAFRAME, get_data_frame_with_item_index

GLUONTS_PARITY_METRICS = ["WQL", "MAPE", "sMAPE", "MSE", "RMSE", "MASE", "WAPE"]
AG_TO_GLUONTS_METRIC = {"WAPE": "ND", "WQL": "mean_wQuantileLoss"}


pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def deepar_trained() -> AbstractGluonTSModel:
    pred = TimeSeriesPredictor(prediction_length=2, verbosity=4)
    pred.fit(
        DUMMY_TS_DATAFRAME,
        tuning_data=DUMMY_TS_DATAFRAME,
        hyperparameters={
            "DeepAR": dict(epochs=2),
        },
    )
    return pred._trainer.load_model("DeepAR")


@pytest.fixture(scope="module")
def deepar_trained_zero_data() -> AbstractGluonTSModel:
    pred = TimeSeriesPredictor(prediction_length=2, verbosity=4)

    data = DUMMY_TS_DATAFRAME.copy() * 0

    pred.fit(
        data,
        tuning_data=data,
        hyperparameters={
            "DeepAR": dict(epochs=2),
        },
    )
    return pred._trainer.load_model("DeepAR")


def to_gluonts_forecast(forecast_df, freq):
    forecast_list = []
    for item_id, fcast in forecast_df.groupby(level="item_id", sort=False):
        start_date = fcast.index[0][1].to_period(freq=freq)
        qf = QuantileForecast(
            forecast_arrays=fcast.values.T,
            start_date=start_date,
            forecast_keys=fcast.columns,
            item_id=item_id,
        )
        forecast_list.append(qf)
    return forecast_list


def to_gluonts_test_set(data):
    ts_list = []
    for item_id, ts in data.groupby(level="item_id", sort=False):
        ts = ts.loc[item_id]["target"]
        ts.index = ts.index.to_period(freq=data.freq)
        ts_list.append(ts)
    return ts_list


@pytest.mark.parametrize("metric_name", GLUONTS_PARITY_METRICS)
def test_when_given_learned_model_when_evaluator_called_then_output_equal_to_gluonts(metric_name, deepar_trained):
    model = deepar_trained
    data_train, data_test = DUMMY_TS_DATAFRAME.train_test_split(model.prediction_length)

    forecast_df = model.predict(data_train)
    seasonal_period = 3
    ag_evaluator = TimeSeriesEvaluator(
        eval_metric=metric_name, prediction_length=model.prediction_length, eval_metric_seasonal_period=seasonal_period
    )
    ag_value = ag_evaluator(DUMMY_TS_DATAFRAME, forecast_df)

    forecast_list = to_gluonts_forecast(forecast_df, freq=data_train.freq)
    ts_list = to_gluonts_test_set(data_test)
    gts_evaluator = GluonTSEvaluator(seasonality=seasonal_period)
    gts_results, _ = gts_evaluator(ts_iterator=ts_list, fcst_iterator=forecast_list)
    gts_metric_name = AG_TO_GLUONTS_METRIC.get(metric_name, metric_name)
    assert np.isclose(gts_results[gts_metric_name], ag_value, atol=1e-5)


@pytest.mark.parametrize("metric_name", GLUONTS_PARITY_METRICS)
def test_when_given_all_zero_data_when_evaluator_called_then_output_equal_to_gluonts(
    metric_name, deepar_trained_zero_data
):
    if metric_name == "MASE":
        pytest.skip("MASE is undefined if all data is constant")

    model = deepar_trained_zero_data
    data = DUMMY_TS_DATAFRAME.copy() * 0
    data_train, data_test = data.train_test_split(model.prediction_length)

    forecast_df = model.predict(data_train)
    seasonal_period = 3
    ag_evaluator = TimeSeriesEvaluator(
        eval_metric=metric_name, prediction_length=model.prediction_length, eval_metric_seasonal_period=seasonal_period
    )
    ag_value = ag_evaluator(data, forecast_df)

    forecast_list = to_gluonts_forecast(forecast_df, freq=data_train.freq)
    ts_list = to_gluonts_test_set(data_test)
    gts_evaluator = GluonTSEvaluator(seasonality=seasonal_period)
    gts_results, _ = gts_evaluator(ts_iterator=ts_list, fcst_iterator=forecast_list)

    gts_metric_name = AG_TO_GLUONTS_METRIC.get(metric_name, metric_name)
    assert np.isclose(gts_results[gts_metric_name], ag_value, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("metric_name", GLUONTS_PARITY_METRICS)
def test_when_given_zero_forecasts_when_evaluator_called_then_output_equal_to_gluonts(metric_name, deepar_trained):
    model = deepar_trained
    data_train, data_test = DUMMY_TS_DATAFRAME.train_test_split(model.prediction_length)

    forecast_df = model.predict(data_train) * 0
    seasonal_period = 3
    ag_evaluator = TimeSeriesEvaluator(
        eval_metric=metric_name, prediction_length=model.prediction_length, eval_metric_seasonal_period=seasonal_period
    )
    ag_value = ag_evaluator(DUMMY_TS_DATAFRAME, forecast_df)

    forecast_list = to_gluonts_forecast(forecast_df, freq=data_train.freq)
    ts_list = to_gluonts_test_set(data_test)
    gts_evaluator = GluonTSEvaluator(seasonality=seasonal_period)
    gts_results, _ = gts_evaluator(ts_iterator=ts_list, fcst_iterator=forecast_list)
    gts_metric_name = AG_TO_GLUONTS_METRIC.get(metric_name, metric_name)
    assert np.isclose(gts_results[gts_metric_name], ag_value, atol=1e-5)


def test_available_metrics_have_coefficients():
    for m in TimeSeriesEvaluator.AVAILABLE_METRICS:
        assert TimeSeriesEvaluator.METRIC_COEFFICIENTS[m]


@pytest.mark.parametrize(
    "check_input, expected_output",
    [
        (None, TimeSeriesEvaluator.DEFAULT_METRIC),
    ]
    + [(k, k) for k in TimeSeriesEvaluator.AVAILABLE_METRICS],
)
@pytest.mark.parametrize("raise_errors", [True, False])
def test_given_correct_input_check_get_eval_metric_output_correct(check_input, expected_output, raise_errors):
    assert expected_output == TimeSeriesEvaluator.check_get_evaluation_metric(
        check_input, raise_if_not_available=raise_errors
    )


@pytest.mark.parametrize("raise_errors", [True, False])
def test_given_no_input_check_get_eval_metric_output_default(raise_errors):
    assert TimeSeriesEvaluator.DEFAULT_METRIC == TimeSeriesEvaluator.check_get_evaluation_metric(
        raise_if_not_available=raise_errors
    )


def test_given_unavailable_input_and_raise_check_get_eval_metric_raises():
    with pytest.raises(ValueError):
        TimeSeriesEvaluator.check_get_evaluation_metric("some_nonsense_eval_metric", raise_if_not_available=True)


def test_given_unavailable_input_and_no_raise_check_get_eval_metric_output_default():
    assert TimeSeriesEvaluator.DEFAULT_METRIC == TimeSeriesEvaluator.check_get_evaluation_metric(
        "some_nonsense_eval_metric", raise_if_not_available=False
    )


@pytest.mark.parametrize("eval_metric", ["MASE", "RMSSE"])
def test_given_historic_data_not_cached_when_scoring_then_exception_is_raised(eval_metric):
    prediction_length = 3
    evaluator = TimeSeriesEvaluator(eval_metric=eval_metric, prediction_length=prediction_length)
    data_future = DUMMY_TS_DATAFRAME.slice_by_timestep(-prediction_length, None)
    predictions = data_future.rename({"target": "mean"}, axis=1)
    with pytest.raises(AssertionError, match="Call save_past_metrics before"):
        evaluator.score_with_saved_past_metrics(data_future=data_future, predictions=predictions)


def test_when_eval_metric_seasonal_period_is_longer_than_ts_then_scale_is_set_to_1():
    seasonal_period = max(DUMMY_TS_DATAFRAME.num_timesteps_per_item())
    naive_error_per_item = in_sample_abs_seasonal_error(
        y_past=DUMMY_TS_DATAFRAME["target"], seasonal_period=seasonal_period
    )
    assert (naive_error_per_item == 1.0).all()


@pytest.mark.parametrize("prediction_length, eval_metric_seasonal_period, expected_result", [(3, 1, 3), (6, 3, 2)])
def test_RMSSE(prediction_length, eval_metric_seasonal_period, expected_result):
    data = get_data_frame_with_item_index(
        ["1"],
        start_date="2022-01-01 00:00:00",
        data_length=2 * prediction_length,
        columns=["target"],
        data_generation="sequential",
    )
    predictions = get_data_frame_with_item_index(
        ["1"],
        start_date=str(pd.Timestamp("2022-01-01 00:00:00") + pd.to_timedelta(prediction_length, unit="H")),
        data_length=prediction_length,
        columns=["mean"],
        data_generation="sequential",
    )
    ag_evaluator = TimeSeriesEvaluator(
        eval_metric="RMSSE",
        prediction_length=prediction_length,
        eval_metric_seasonal_period=eval_metric_seasonal_period,
    )
    ag_value = ag_evaluator(data, predictions)
    assert ag_value == expected_result
