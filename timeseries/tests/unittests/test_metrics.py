import numpy as np
import pandas as pd
import pytest
from gluonts.evaluation import Evaluator as GluonTSEvaluator
from gluonts.model.forecast import QuantileForecast

from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.metrics import AVAILABLE_METRICS, DEFAULT_METRIC_NAME, check_get_evaluation_metric
from autogluon.timeseries.metrics.utils import _in_sample_abs_seasonal_error, _in_sample_squared_seasonal_error
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


def check_gluonts_parity(metric_name, data, model, zero_forecast=False, equal_nan=False):
    data_train, data_test = data.train_test_split(model.prediction_length)
    forecast_df = model.predict(data_train)
    forecast_df["mean"] = forecast_df["0.5"]
    if zero_forecast:
        forecast_df = forecast_df * 0
    seasonal_period = 3
    ag_metric = check_get_evaluation_metric(metric_name)

    ag_value = ag_metric.sign * ag_metric(
        data_test,
        forecast_df,
        prediction_length=model.prediction_length,
        seasonal_period=seasonal_period,
    )

    forecast_list = to_gluonts_forecast(forecast_df, freq=data_train.freq)
    ts_list = to_gluonts_test_set(data_test)
    gts_evaluator = GluonTSEvaluator(seasonality=seasonal_period)
    gts_results, _ = gts_evaluator(ts_iterator=ts_list, fcst_iterator=forecast_list)
    gts_metric_name = AG_TO_GLUONTS_METRIC.get(metric_name, metric_name)
    assert np.isclose(gts_results[gts_metric_name], ag_value, atol=1e-5, equal_nan=equal_nan)


@pytest.mark.parametrize("metric_name", GLUONTS_PARITY_METRICS)
def test_when_given_learned_model_when_evaluator_called_then_output_equal_to_gluonts(metric_name, deepar_trained):
    check_gluonts_parity(metric_name, DUMMY_TS_DATAFRAME, deepar_trained)


@pytest.mark.parametrize("metric_name", GLUONTS_PARITY_METRICS)
def test_when_given_all_zero_data_when_evaluator_called_then_output_equal_to_gluonts(
    metric_name, deepar_trained_zero_data
):
    if metric_name == "MASE":
        pytest.skip("MASE is undefined if all data is constant")

    check_gluonts_parity(metric_name, DUMMY_TS_DATAFRAME.copy() * 0, deepar_trained_zero_data, equal_nan=True)


@pytest.mark.parametrize("metric_name", GLUONTS_PARITY_METRICS)
def test_when_given_zero_forecasts_when_evaluator_called_then_output_equal_to_gluonts(metric_name, deepar_trained):
    check_gluonts_parity(metric_name, DUMMY_TS_DATAFRAME, deepar_trained, zero_forecast=True)


def test_available_metrics_have_coefficients():
    for metric_cls in AVAILABLE_METRICS.values():
        metric = metric_cls()
        assert metric.sign in [-1, 1]


@pytest.mark.parametrize(
    "check_input, expected_output",
    [(None, AVAILABLE_METRICS[DEFAULT_METRIC_NAME]())]
    + [(metric_name, metric_cls()) for metric_name, metric_cls in AVAILABLE_METRICS.items()]
    + [(metric_cls, metric_cls()) for metric_name, metric_cls in AVAILABLE_METRICS.items()]
    + [(metric_cls(), metric_cls()) for metric_name, metric_cls in AVAILABLE_METRICS.items()],
)
def test_given_correct_input_check_get_eval_metric_output_correct(check_input, expected_output):
    assert expected_output.name == check_get_evaluation_metric(check_input).name


def test_given_unavailable_input_and_raise_check_get_eval_metric_raises():
    with pytest.raises(ValueError):
        check_get_evaluation_metric("some_nonsense_eval_metric")


@pytest.mark.parametrize("eval_metric", ["MASE", "RMSSE", "SQL"])
def test_given_historic_data_not_cached_when_scoring_then_exception_is_raised(eval_metric):
    prediction_length = 3
    evaluator = check_get_evaluation_metric(eval_metric)
    data_future = DUMMY_TS_DATAFRAME.slice_by_timestep(-prediction_length, None)
    predictions = data_future.rename({"target": "mean"}, axis=1)
    with pytest.raises(AssertionError, match="Call `save_past_metrics` before"):
        evaluator.compute_metric(data_future=data_future, predictions=predictions)


def test_when_eval_metric_seasonal_period_is_longer_than_ts_then_abs_seasonal_error_is_set_to_1():
    seasonal_period = max(DUMMY_TS_DATAFRAME.num_timesteps_per_item())
    naive_error_per_item = _in_sample_abs_seasonal_error(
        y_past=DUMMY_TS_DATAFRAME["target"], seasonal_period=seasonal_period
    )
    assert (naive_error_per_item == 1.0).all()


def test_when_eval_metric_seasonal_period_is_longer_than_ts_then_squared_seasonal_error_is_set_to_1():
    seasonal_period = max(DUMMY_TS_DATAFRAME.num_timesteps_per_item())
    naive_error_per_item = _in_sample_squared_seasonal_error(
        y_past=DUMMY_TS_DATAFRAME["target"], seasonal_period=seasonal_period
    )
    assert (naive_error_per_item == 1.0).all()


@pytest.mark.parametrize("prediction_length, seasonal_period, expected_result", [(3, 1, 3), (6, 3, 2)])
def test_RMSSE(prediction_length, seasonal_period, expected_result):
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
    metric = check_get_evaluation_metric("RMSSE")
    ag_value = metric.sign * metric(
        data, predictions, prediction_length=prediction_length, seasonal_period=seasonal_period
    )
    assert ag_value == expected_result


@pytest.mark.parametrize("metric_name", AVAILABLE_METRICS)
def test_given_metric_is_optimized_by_median_when_model_predicts_then_median_is_pasted_to_mean_forecast(metric_name):
    eval_metric = check_get_evaluation_metric(metric_name)
    pred = TimeSeriesPredictor(prediction_length=5, eval_metric=eval_metric)
    pred.fit(DUMMY_TS_DATAFRAME, hyperparameters={"DeepAR": {"epochs": 1, "num_batches_per_epoch": 1}})
    predictions = pred.predict(DUMMY_TS_DATAFRAME)
    if eval_metric.optimized_by_median:
        assert (predictions["mean"] == predictions["0.5"]).all()
    else:
        assert (predictions["mean"] != predictions["0.5"]).any()


@pytest.mark.parametrize("metric_name", AVAILABLE_METRICS)
def test_when_perfect_predictions_passed_to_metric_then_score_equals_optimum(metric_name):
    prediction_length = 5
    eval_metric = check_get_evaluation_metric(metric_name)
    data = DUMMY_TS_DATAFRAME.copy()
    predictions = data.slice_by_timestep(-prediction_length, None).rename(columns={"target": "mean"})
    for q in ["0.1", "0.4", "0.9"]:
        predictions[q] = predictions["mean"]
    score = eval_metric.score(data, predictions, prediction_length=prediction_length)
    assert score == eval_metric.optimum


@pytest.mark.parametrize("metric_name", AVAILABLE_METRICS)
def test_when_better_predictions_passed_to_metric_then_score_improves(metric_name):
    prediction_length = 5
    eval_metric = check_get_evaluation_metric(metric_name)
    data = DUMMY_TS_DATAFRAME.copy()
    predictions = data.slice_by_timestep(-prediction_length, None).rename(columns={"target": "mean"})
    for q in ["0.1", "0.4", "0.9"]:
        predictions[q] = predictions["mean"]
    good_score = eval_metric.score(data, predictions + 1, prediction_length=prediction_length)
    bad_score = eval_metric.score(data, predictions + 50, prediction_length=prediction_length)
    assert good_score > bad_score
