from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from gluonts.dataset.split import split
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    ND,
    RMSE,
    SMAPE,
    AverageMeanScaledQuantileLoss,
    MeanWeightedSumQuantileLoss,
)
from gluonts.ev.metrics import Metric as GluonTSMetric
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast

from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.metrics import (
    AVAILABLE_METRICS,
    DEFAULT_METRIC_NAME,
    TimeSeriesScorer,
    check_get_evaluation_metric,
)
from autogluon.timeseries.metrics.utils import in_sample_abs_seasonal_error, in_sample_squared_seasonal_error
from autogluon.timeseries.models.gluonts.abstract import AbstractGluonTSModel

from .common import DUMMY_TS_DATAFRAME, get_data_frame_with_item_index, get_prediction_for_df

pytestmark = pytest.mark.filterwarnings("ignore")


def get_ag_and_gts_metrics() -> List[Tuple[str, GluonTSMetric]]:
    # Each entry is a tuple (ag_metric_name, gts_metric_object)
    default_quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Metric that have different names in AutoGluon and GluonTS
    ag_and_gts_metrics = [
        ("WQL", MeanWeightedSumQuantileLoss(default_quantile_levels)),
        ("SQL", AverageMeanScaledQuantileLoss(default_quantile_levels)),
        ("WAPE", ND("mean")),
    ]
    # Metric that have same names in AutoGluon and GluonTS
    for point_metric_cls in [MAPE, SMAPE, MSE, RMSE, MASE, MAE]:
        name = str(point_metric_cls.__name__)
        ag_and_gts_metrics.append((name, point_metric_cls("mean")))
    return ag_and_gts_metrics


AG_AND_GTS_METRICS = get_ag_and_gts_metrics()


@pytest.fixture(scope="module")
def deepar_trained() -> AbstractGluonTSModel:
    pred = TimeSeriesPredictor(prediction_length=2, verbosity=4)
    pred.fit(
        DUMMY_TS_DATAFRAME,
        tuning_data=DUMMY_TS_DATAFRAME,
        hyperparameters={
            "DeepAR": {"max_epochs": 2},
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
            "DeepAR": {"max_epochs": 2},
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


def to_gluonts_test_set(data, prediction_length):
    ts_list = []
    for item_id, ts in data.groupby(level="item_id", sort=False):
        entry = {"target": ts.loc[item_id]["target"], "start": pd.Period(ts.loc[item_id].index[0], freq=data.freq)}
        ts_list.append(entry)
    _, test_template = split(dataset=ts_list, offset=-prediction_length)
    return test_template.generate_instances(prediction_length, windows=1)


def check_gluonts_parity(ag_metric_name, gts_metric, data, model, zero_forecast=False, equal_nan=False):
    data_train, data_test = data.train_test_split(model.prediction_length)
    forecast_df = model.predict(data_train)
    forecast_df["mean"] = forecast_df["0.5"]
    if zero_forecast:
        forecast_df = forecast_df * 0
    ag_metric = check_get_evaluation_metric(
        ag_metric_name,
        prediction_length=model.prediction_length,
        seasonal_period=3,
    )

    ag_value = ag_metric.sign * ag_metric(data_test, forecast_df)

    gts_forecast = to_gluonts_forecast(forecast_df, freq=data_train.freq)
    gts_test_set = to_gluonts_test_set(data_test, model.prediction_length)
    gts_value = evaluate_forecasts(
        gts_forecast, test_data=gts_test_set, seasonality=ag_metric.seasonal_period, metrics=[gts_metric]
    ).values.item()
    assert np.isclose(gts_value, ag_value, atol=1e-5, equal_nan=equal_nan)


@pytest.mark.parametrize("ag_metric_name, gts_metric", AG_AND_GTS_METRICS)
def test_when_metric_evaluated_then_output_equal_to_gluonts(ag_metric_name, gts_metric, deepar_trained):
    check_gluonts_parity(
        ag_metric_name,
        gts_metric,
        data=DUMMY_TS_DATAFRAME,
        model=deepar_trained,
    )


@pytest.mark.parametrize("ag_metric_name, gts_metric", AG_AND_GTS_METRICS)
def test_given_all_zero_data_when_metric_evaluated_then_output_equal_to_gluonts(
    ag_metric_name, gts_metric, deepar_trained_zero_data
):
    check_gluonts_parity(
        ag_metric_name,
        gts_metric,
        data=DUMMY_TS_DATAFRAME.copy() * 0,
        model=deepar_trained_zero_data,
        equal_nan=True,
    )


@pytest.mark.parametrize("ag_metric_name, gts_metric", AG_AND_GTS_METRICS)
def test_given_zero_forecasts_when_metric_evaluated_then_output_equal_to_gluonts(
    ag_metric_name, gts_metric, deepar_trained
):
    check_gluonts_parity(
        ag_metric_name,
        gts_metric,
        data=DUMMY_TS_DATAFRAME,
        model=deepar_trained,
        zero_forecast=True,
    )


@pytest.mark.parametrize("ag_metric_name, gts_metric", AG_AND_GTS_METRICS)
def test_given_missing_target_values_when_metric_evaluated_then_output_equal_to_gluonts(
    ag_metric_name, gts_metric, deepar_trained
):
    check_gluonts_parity(
        ag_metric_name,
        gts_metric,
        data=DUMMY_TS_DATAFRAME,
        model=deepar_trained,
    )


@pytest.mark.parametrize("metric_cls", AVAILABLE_METRICS.values())
def test_given_missing_target_values_when_metric_evaluated_then_metric_is_not_nan(metric_cls):
    prediction_length = 5
    train, test = DUMMY_TS_DATAFRAME.train_test_split(prediction_length)
    predictions = get_prediction_for_df(train, prediction_length)
    score = metric_cls(prediction_length=prediction_length)(data=test, predictions=predictions)
    assert not pd.isna(score)


@pytest.mark.parametrize("metric_cls", AVAILABLE_METRICS.values())
def test_given_predictions_contain_nan_when_metric_evaluated_then_exception_is_raised(metric_cls):
    prediction_length = 5
    train, test = DUMMY_TS_DATAFRAME.train_test_split(prediction_length)
    predictions = get_prediction_for_df(train, prediction_length)
    predictions.iloc[[3, 5]] = float("nan")
    with pytest.raises(AssertionError, match="Predictions contain NaN values"):
        metric_cls(prediction_length=prediction_length)(data=test, predictions=predictions)


def test_available_metrics_have_coefficients():
    for metric_cls in AVAILABLE_METRICS.values():
        metric = metric_cls()
        assert metric.sign in [-1, 1]


@pytest.mark.parametrize(
    "check_input, expected_output",
    [(None, AVAILABLE_METRICS[DEFAULT_METRIC_NAME]())]
    + [(metric_name, metric_cls()) for metric_name, metric_cls in AVAILABLE_METRICS.items()]
    + [(metric_cls, metric_cls()) for metric_cls in AVAILABLE_METRICS.values()]
    + [(metric_cls(), metric_cls()) for metric_cls in AVAILABLE_METRICS.values()],
)
def test_given_correct_input_check_get_eval_metric_output_correct(check_input, expected_output):
    assert expected_output.name == check_get_evaluation_metric(check_input, prediction_length=1).name


def test_given_unavailable_input_and_raise_check_get_eval_metric_raises():
    with pytest.raises(ValueError):
        check_get_evaluation_metric("some_nonsense_eval_metric", prediction_length=1)


@pytest.mark.parametrize("eval_metric", ["MASE", "RMSSE", "SQL"])
def test_given_historic_data_not_cached_when_scoring_then_exception_is_raised(eval_metric):
    prediction_length = 3
    evaluator = check_get_evaluation_metric(eval_metric, prediction_length=prediction_length)
    data_future = DUMMY_TS_DATAFRAME.slice_by_timestep(-prediction_length, None)
    predictions = data_future.rename({"target": "mean"}, axis=1)
    with pytest.raises(AssertionError, match="Call `save_past_metrics` before"):
        evaluator.compute_metric(data_future=data_future, predictions=predictions)


def test_when_eval_metric_seasonal_period_is_longer_than_ts_then_abs_seasonal_error_is_set_to_1():
    seasonal_period = max(DUMMY_TS_DATAFRAME.num_timesteps_per_item())
    naive_error_per_item = in_sample_abs_seasonal_error(
        y_past=DUMMY_TS_DATAFRAME["target"], seasonal_period=seasonal_period
    )
    assert (naive_error_per_item == 1.0).all()


def test_when_eval_metric_seasonal_period_is_longer_than_ts_then_squared_seasonal_error_is_set_to_1():
    seasonal_period = max(DUMMY_TS_DATAFRAME.num_timesteps_per_item())
    naive_error_per_item = in_sample_squared_seasonal_error(
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
        start_date=str(pd.Timestamp("2022-01-01 00:00:00") + pd.to_timedelta(prediction_length, unit="h")),
        data_length=prediction_length,
        columns=["mean"],
        data_generation="sequential",
    )
    metric = check_get_evaluation_metric("RMSSE", prediction_length=prediction_length, seasonal_period=seasonal_period)
    ag_value = metric.sign * metric(data, predictions)
    assert ag_value == expected_result


@pytest.mark.parametrize(
    "prediction_length, expected_result",
    [
        (3, 1.03952774131806),
        (4, 1.11754262032011),
        (5, 1.17302207173233),
        (6, 1.21497832991862),
    ],
)
def test_RMSLE(prediction_length, expected_result):
    data = get_data_frame_with_item_index(
        ["1"],
        start_date="2022-01-01 00:00:00",
        data_length=2 * prediction_length,
        columns=["target"],
        data_generation="sequential",
    )
    predictions = get_data_frame_with_item_index(
        ["1"],
        start_date=str(pd.Timestamp("2022-01-01 00:00:00") + pd.to_timedelta(prediction_length, unit="h")),
        data_length=prediction_length,
        columns=["mean"],
        data_generation="sequential",
    )
    metric = check_get_evaluation_metric("RMSLE", prediction_length=prediction_length)
    ag_value = metric.sign * metric(data, predictions)
    assert np.isclose(ag_value, expected_result, atol=1e-5)


@pytest.mark.parametrize("metric_name", AVAILABLE_METRICS)
def test_given_metric_is_optimized_by_median_when_model_predicts_then_median_is_pasted_to_mean_forecast(metric_name):
    pred = TimeSeriesPredictor(prediction_length=5, eval_metric=metric_name)
    pred.fit(DUMMY_TS_DATAFRAME, hyperparameters={"DeepAR": {"max_epochs": 1, "num_batches_per_epoch": 1}})
    predictions = pred.predict(DUMMY_TS_DATAFRAME)
    if pred.eval_metric.optimized_by_median:
        assert (predictions["mean"] == predictions["0.5"]).all()
    else:
        assert (predictions["mean"] != predictions["0.5"]).any()


@pytest.mark.parametrize("metric_name", AVAILABLE_METRICS)
def test_when_perfect_predictions_passed_to_metric_then_score_equals_optimum(metric_name):
    prediction_length = 5
    eval_metric = check_get_evaluation_metric(metric_name, prediction_length=prediction_length)
    data = DUMMY_TS_DATAFRAME.copy()
    predictions = data.slice_by_timestep(-prediction_length, None).rename(columns={"target": "mean"}).fillna(0.0)
    for q in ["0.1", "0.4", "0.9"]:
        predictions[q] = predictions["mean"]
    score = eval_metric.score(data, predictions)
    assert score == eval_metric.optimum


@pytest.mark.parametrize("metric_name", AVAILABLE_METRICS)
def test_when_better_predictions_passed_to_metric_then_score_improves(metric_name):
    prediction_length = 5
    eval_metric = check_get_evaluation_metric(metric_name, prediction_length=prediction_length)
    data = DUMMY_TS_DATAFRAME.copy()
    predictions = data.slice_by_timestep(-prediction_length, None).rename(columns={"target": "mean"}).fillna(0.0)
    for q in ["0.1", "0.4", "0.9"]:
        predictions[q] = predictions["mean"]
    good_score = eval_metric.score(data, predictions + 1)
    bad_score = eval_metric.score(data, predictions + 50)
    assert good_score > bad_score


@pytest.mark.parametrize("metric_name", ["WCD", "wcd"])
def test_when_experimental_metric_name_used_then_predictor_can_score(metric_name):
    predictor = TimeSeriesPredictor(prediction_length=3, eval_metric=metric_name)
    predictor.fit(DUMMY_TS_DATAFRAME, hyperparameters={"DeepAR": {"max_epochs": 1, "num_batches_per_epoch": 1}})
    evaluation_results = predictor.evaluate(DUMMY_TS_DATAFRAME)
    assert np.isfinite(evaluation_results["WCD"])


@pytest.mark.parametrize(
    "horizon_weight, error_match",
    [
        ([1, 1], "must have length equal to"),
        ([3, 3, 4, 2], "must have length equal to"),
        ([float("inf"), 1, 1], "values must be finite"),
        ([1, 1, float("nan")], "All values"),
        ([0, 0, 0], "At least some values"),
        ([-0.5, 1, 1], "All values"),
    ],
)
def test_when_horizon_weight_contains_invalid_values_then_exception_is_raised(horizon_weight, error_match):
    with pytest.raises(ValueError, match=error_match):
        TimeSeriesScorer.check_get_horizon_weight(horizon_weight, prediction_length=3)


@pytest.mark.parametrize("metric_cls", AVAILABLE_METRICS.values())
def test_when_horizon_weight_is_all_ones_then_metric_value_does_not_change(metric_cls):
    prediction_length = 5
    train, test = DUMMY_TS_DATAFRAME.train_test_split(prediction_length)
    predictions = get_prediction_for_df(train, prediction_length)
    orig_score = metric_cls(prediction_length=prediction_length)(data=test, predictions=predictions)
    weighted_score = metric_cls(prediction_length=prediction_length, horizon_weight=np.ones(prediction_length))(
        data=test, predictions=predictions
    )
    assert np.isclose(orig_score, weighted_score)


@pytest.mark.parametrize("metric_cls", AVAILABLE_METRICS.values())
def test_when_horizon_weight_is_non_uniform_then_metric_value_changes(metric_cls):
    prediction_length = 5
    train, test = DUMMY_TS_DATAFRAME.train_test_split(prediction_length)
    predictions = get_prediction_for_df(train, prediction_length)
    orig_score = metric_cls(prediction_length=prediction_length)(data=test, predictions=predictions)
    weighted_score = metric_cls(prediction_length=prediction_length, horizon_weight=np.array([1, 1, 0, 3, 0]))(
        data=test, predictions=predictions
    )
    assert orig_score != weighted_score


@pytest.mark.parametrize(
    "input_horizon_weight, normalized_horizon_weight",
    [
        [[1], [1]],
        [[1, 3], [0.5, 1.5]],
        [[0, 0, 1], [0, 0, 3]],
    ],
)
def test_when_horizon_weight_is_checked_then_values_are_normalized(input_horizon_weight, normalized_horizon_weight):
    checked_horizon_weight = TimeSeriesScorer.check_get_horizon_weight(
        input_horizon_weight, prediction_length=len(input_horizon_weight)
    )
    assert isinstance(checked_horizon_weight, np.ndarray)
    assert np.allclose(checked_horizon_weight.sum(), len(input_horizon_weight))
    assert np.allclose(checked_horizon_weight, normalized_horizon_weight)


@pytest.mark.parametrize(
    "horizon_weight",
    [[1, 1, 1], [[4, 5, 6, 7]], np.array([1, 2, 3]), (3, 2), np.array([[1, 4]])],
)
def test_when_horizon_weight_is_checked_then_horizon_weight_has_correct_shape(horizon_weight):
    prediction_length = len(np.ravel(horizon_weight))
    scorer = TimeSeriesScorer(prediction_length=prediction_length, horizon_weight=horizon_weight)
    assert isinstance(scorer.horizon_weight, np.ndarray)
    assert scorer.horizon_weight.shape == (1, prediction_length)


@pytest.fixture(scope="module")
def partially_matching_predictions():
    # For each item, the error equals zero for the first two time steps, and the error is positive for the remainder
    prediction_length = 4
    data = DUMMY_TS_DATAFRAME.copy()
    past = data.slice_by_timestep(None, -prediction_length)
    predictions = get_prediction_for_df(past, prediction_length=prediction_length)

    # Set the predictions for the first two time steps to exactly match the ground truth
    future_start = data.slice_by_timestep(-prediction_length, -prediction_length + 2)
    predictions.loc[future_start.index] = future_start.fillna(0.0)
    return data, predictions


@pytest.mark.parametrize("metric_cls", AVAILABLE_METRICS.values())
def test_when_horizon_weight_is_zero_for_wrong_predictions_then_metric_value_is_zero(
    metric_cls, partially_matching_predictions
):
    data, predictions = partially_matching_predictions
    score = metric_cls(prediction_length=4, horizon_weight=np.array([2, 2, 0, 0]))(data=data, predictions=predictions)
    assert np.allclose(score, 0.0)


@pytest.mark.parametrize("metric_cls", AVAILABLE_METRICS.values())
def test_when_horizon_weight_is_zero_for_correct_predictions_then_error_increases(
    metric_cls, partially_matching_predictions
):
    data, predictions = partially_matching_predictions
    prediction_length = 4
    orig_score = metric_cls(prediction_length)(data=data, predictions=predictions)
    weighted_score = metric_cls(prediction_length, horizon_weight=np.array([0, 0, 2, 2]))(
        data=data, predictions=predictions
    )
    assert weighted_score < orig_score
