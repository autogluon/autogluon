"""Unit tests and utils common to all models"""

import itertools
import shutil
import sys
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from flaky import flaky

from autogluon.common import space
from autogluon.core.hpo.constants import RAY_BACKEND
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP
from autogluon.timeseries.metrics import AVAILABLE_METRICS
from autogluon.timeseries.models import DeepARModel, ETSModel
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.models.multi_window import MultiWindowBacktestingModel
from autogluon.timeseries.regressor import CovariateRegressor

from ..common import (
    DUMMY_TS_DATAFRAME,
    CustomMetric,
    dict_equal_primitive,
    get_data_frame_with_item_index,
    to_supported_pandas_freq,
)
from .chronos import TESTABLE_MODELS as CHRONOS_TESTABLE_MODELS
from .chronos import ZERO_SHOT_MODELS as CHRONOS_ZERO_SHOT_MODELS
from .test_gluonts import TESTABLE_MODELS as GLUONTS_TESTABLE_MODELS
from .test_local import TESTABLE_MODELS as LOCAL_TESTABLE_MODELS
from .test_mlforecast import TESTABLE_MODELS as MLFORECAST_TESTABLE_MODELS
from .test_multi_window_model import get_multi_window_deepar

TESTABLE_MODELS = (
    CHRONOS_TESTABLE_MODELS
    + GLUONTS_TESTABLE_MODELS
    + LOCAL_TESTABLE_MODELS
    + MLFORECAST_TESTABLE_MODELS
    + [get_multi_window_deepar]
)

TESTABLE_PREDICTION_LENGTHS = [1, 5]


@pytest.fixture(scope="module")
def trained_models(dummy_hyperparameters):
    models = {}
    model_paths = []
    for model_class, prediction_length in itertools.product(TESTABLE_MODELS, TESTABLE_PREDICTION_LENGTHS):
        temp_model_path = tempfile.mkdtemp()
        model = model_class(
            path=temp_model_path,
            freq="h",
            prediction_length=prediction_length,
            hyperparameters=dummy_hyperparameters,
        )

        model.fit(train_data=DUMMY_TS_DATAFRAME)
        model.score_and_cache_oof(DUMMY_TS_DATAFRAME, store_val_score=True, store_predict_time=True)
        models[(prediction_length, repr(model_class))] = model
        model_paths.append(temp_model_path)

    yield models

    for td in model_paths:
        shutil.rmtree(td)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_models_can_be_initialized(model_class, temp_model_path):
    model = model_class(path=temp_model_path, freq="h", prediction_length=24)
    assert isinstance(model, AbstractTimeSeriesModel)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
@pytest.mark.parametrize("metric", AVAILABLE_METRICS)
def test_when_fit_called_then_models_train_and_all_scores_can_be_computed(
    model_class, prediction_length, metric, trained_models
):
    model = trained_models[(prediction_length, repr(model_class))]
    score = model.score(DUMMY_TS_DATAFRAME, metric)

    assert isinstance(score, float)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
def test_when_score_and_cache_oof_called_then_attributes_are_saved(model_class, prediction_length, trained_models):
    model = trained_models[(prediction_length, repr(model_class))]
    assert isinstance(model.val_score, float)
    assert isinstance(model.predict_time, float)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
def test_when_score_and_cache_oof_called_then_oof_predictions_are_saved(
    model_class, prediction_length, trained_models
):
    model = trained_models[(prediction_length, repr(model_class))]
    if isinstance(model, MultiWindowBacktestingModel):
        pytest.skip()

    oof_predictions = model.get_oof_predictions()[0]
    assert isinstance(oof_predictions, TimeSeriesDataFrame)
    oof_score = model._score_with_predictions(DUMMY_TS_DATAFRAME, oof_predictions)
    assert isinstance(oof_score, float)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
def test_when_score_called_then_model_receives_truncated_data(model_class, prediction_length, trained_models):
    model = trained_models[(prediction_length, repr(model_class))]

    with mock.patch.object(model, "predict") as patch_method:
        # Mock breaks the internals of the `score` method
        try:
            model.score(DUMMY_TS_DATAFRAME)
        except AssertionError:
            pass

        (call_df,) = patch_method.call_args[0]

        for j in DUMMY_TS_DATAFRAME.item_ids:
            assert np.allclose(call_df.loc[j], DUMMY_TS_DATAFRAME.loc[j][:-prediction_length], equal_nan=True)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", TESTABLE_PREDICTION_LENGTHS)
def test_when_models_saved_then_they_can_be_loaded(model_class, trained_models, prediction_length):
    model = trained_models[(prediction_length, repr(model_class))]

    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert dict_equal_primitive(model.params, loaded_model.params)
    assert dict_equal_primitive(model.params_aux, loaded_model.params_aux)
    assert model.metadata == loaded_model.metadata
    for orig_oof_pred, loaded_oof_pred in zip(model.get_oof_predictions(), loaded_model.get_oof_predictions()):
        assert orig_oof_pred.equals(loaded_oof_pred)


@flaky
@pytest.mark.skipif(sys.platform.startswith("win"), reason="HPO tests lead to known issues in Windows platform tests")
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_hyperparameter_spaces_when_tune_called_then_tuning_output_correct(
    model_class, temp_model_path, dummy_hyperparameters
):
    model = model_class(
        path=temp_model_path,
        freq="h",
        quantile_levels=[0.1, 0.9],
        hyperparameters={**dummy_hyperparameters, "max_epochs": space.Int(1, 3)},
    )
    if isinstance(model, MultiWindowBacktestingModel):
        val_data = None
    else:
        val_data = DUMMY_TS_DATAFRAME

    num_trials = 2

    hpo_results, _ = model.hyperparameter_tune(
        hyperparameter_tune_kwargs={"num_trials": num_trials, "scheduler": "local", "searcher": "random"},
        time_limit=300,
        train_data=DUMMY_TS_DATAFRAME,
        val_data=val_data,
    )
    assert len(hpo_results) == num_trials
    for result in hpo_results.values():
        assert 1 <= result["hyperparameters"]["max_epochs"] <= 3


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_hyperparameter_spaces_to_init_when_fit_called_then_error_is_raised(model_class, temp_model_path):
    model = model_class(
        path=temp_model_path,
        freq="h",
        quantile_levels=[0.1, 0.9],
        hyperparameters={
            "max_epochs": space.Int(3, 4),
        },
    )
    with pytest.raises(ValueError, match=".*hyperparameter_tune.*"):
        model.fit(
            train_data=DUMMY_TS_DATAFRAME,
        )


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize(
    "quantile_levels",
    [
        [0.1, 0.44, 0.72],
        [0.1, 0.5, 0.9],
    ],
)
def test_when_fit_called_then_models_train_and_returned_predictor_inference_has_mean_and_correct_quantiles(
    model_class, quantile_levels, temp_model_path, dummy_hyperparameters
):
    model = model_class(
        path=temp_model_path,
        freq="h",
        prediction_length=3,
        quantile_levels=quantile_levels,
        hyperparameters=dummy_hyperparameters,
    )
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    predictions = model.predict(DUMMY_TS_DATAFRAME, quantile_levels=quantile_levels)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    expected_columns = ["mean"] + [str(q) for q in quantile_levels]
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert (predictions.columns == expected_columns).all()


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
def test_when_fit_called_then_models_train_and_returned_predictor_inference_correct(
    model_class, prediction_length, trained_models
):
    train_data = DUMMY_TS_DATAFRAME
    model = trained_models[(prediction_length, repr(model_class))]

    predictions = model.predict(train_data)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == train_data.item_ids)
    assert all(len(predictions.loc[i]) == prediction_length for i in predicted_item_index)
    assert all(predictions.loc[i].index[0].hour > 0 for i in predicted_item_index)


@pytest.mark.parametrize(
    "model_class",
    [DeepARModel, ETSModel],
)
@pytest.mark.parametrize("test_data_index", [["A", "B"], ["C", "D"], ["A"]])
def test_when_fit_called_then_models_train_and_returned_predictor_inference_aligns_with_time(
    model_class, test_data_index, temp_model_path, dummy_hyperparameters
):
    prediction_length = 3
    train_data = get_data_frame_with_item_index(["A", "B"], data_length=10)
    test_data = get_data_frame_with_item_index(test_data_index, data_length=15)

    model = model_class(
        path=temp_model_path,
        freq="h",
        prediction_length=prediction_length,
        hyperparameters=dummy_hyperparameters,
    )

    model.fit(train_data=train_data)

    max_hour_in_test = test_data.index.levels[1].max().hour
    predictions = model.predict(test_data)
    min_hour_in_pred = predictions.index.levels[1].min().hour

    assert min_hour_in_pred == max_hour_in_test + 1


@pytest.mark.parametrize("freq", ["D", "h", "s", "ME"])
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_predict_called_then_predicted_timestamps_align_with_time(
    model_class, freq, temp_model_path, dummy_hyperparameters
):
    freq = to_supported_pandas_freq(freq)
    prediction_length = 4
    train_length = 20
    item_id = "A"
    timestamps = pd.date_range(start=pd.Timestamp("2020-01-05 12:05:01"), freq=freq, periods=train_length)
    index = pd.MultiIndex.from_product([(item_id,), timestamps], names=[ITEMID, TIMESTAMP])
    train_data = TimeSeriesDataFrame(pd.DataFrame({"target": np.random.rand(train_length)}, index=index))

    model = model_class(
        path=temp_model_path,
        freq=train_data.freq,
        prediction_length=prediction_length,
        hyperparameters=dummy_hyperparameters,
    )

    model.fit(train_data=train_data)
    predictions = model.predict(train_data)

    offset = pd.tseries.frequencies.to_offset(freq)
    preds_first_item = predictions.loc[item_id]
    for i in range(prediction_length):
        assert preds_first_item.index[i] == timestamps[-1] + offset * (i + 1)


@pytest.mark.parametrize(
    "model_class",
    [DeepARModel, ETSModel],
)
@pytest.mark.parametrize(
    "train_data, test_data",
    [
        (get_data_frame_with_item_index([0, 1, 2, 3]), get_data_frame_with_item_index([0, 1, 2, 3])),
        (
            get_data_frame_with_item_index(["A", "B", "C"]),
            get_data_frame_with_item_index(["A", "B", "C"]),
        ),
        (
            get_data_frame_with_item_index(["A", "B", "C"]),
            get_data_frame_with_item_index(["A", "B", "D"]),
        ),
        (
            get_data_frame_with_item_index(["A", "B", "C"]),
            get_data_frame_with_item_index(["A", "B"]),
        ),
        (
            get_data_frame_with_item_index(["A", "B"]),
            get_data_frame_with_item_index(["A", "B", "C"]),
        ),
    ],
)
def test_when_predict_called_with_test_data_then_predictor_inference_correct(
    model_class, temp_model_path, train_data, test_data, dummy_hyperparameters
):
    prediction_length = 5
    model = model_class(
        path=temp_model_path,
        freq="h",
        prediction_length=prediction_length,
        hyperparameters=dummy_hyperparameters,
    )

    model.fit(train_data=train_data)

    predictions = model.predict(test_data)

    assert isinstance(predictions, TimeSeriesDataFrame)
    assert len(predictions) == test_data.num_items * prediction_length

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == test_data.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == prediction_length for i in predicted_item_index)
    assert all(predictions.loc[i].index[0].hour > 0 for i in predicted_item_index)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
def test_when_get_info_is_called_then_all_keys_are_present(model_class, prediction_length, trained_models):
    model = trained_models[(prediction_length, repr(model_class))]
    info = model.get_info()
    expected_keys = [
        "name",
        "model_type",
        "eval_metric",
        "fit_time",
        "predict_time",
        "freq",
        "prediction_length",
        "quantile_levels",
        "val_score",
        "hyperparameters",
    ]
    for key in expected_keys:
        assert key in info


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_median_not_in_quantile_levels_then_median_is_present_in_raw_predictions(
    model_class, dummy_hyperparameters
):
    data = get_data_frame_with_item_index(["B", "A", "X", "C"])
    model = model_class(
        prediction_length=3,
        quantile_levels=[0.1, 0.15],
        freq=data.freq,
        hyperparameters=dummy_hyperparameters,
    )
    if isinstance(model, MultiWindowBacktestingModel):
        # Median is present in the predictions of the base model, but not in the MultiWindowBacktestingModel wrapper
        pytest.skip()
    model.fit(train_data=data)

    raw_predictions = model._predict(data)
    assert "0.5" in raw_predictions.columns


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_median_not_in_quantile_levels_then_median_is_dropped_at_prediction_time(
    model_class, dummy_hyperparameters
):
    model = model_class(
        prediction_length=3,
        quantile_levels=[0.1, 0.15],
        freq=DUMMY_TS_DATAFRAME.freq,
        hyperparameters=dummy_hyperparameters,
    )
    assert model.must_drop_median
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    final_predictions = model.predict(DUMMY_TS_DATAFRAME)
    assert "0.5" not in final_predictions.columns


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_custom_metric_passed_to_model_then_model_can_score(model_class, dummy_hyperparameters):
    model = model_class(
        prediction_length=3,
        freq=DUMMY_TS_DATAFRAME.freq,
        quantile_levels=[0.1, 0.15],
        hyperparameters=dummy_hyperparameters,
        eval_metric=CustomMetric(),
    )
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    score = model.score(DUMMY_TS_DATAFRAME.sort_index())
    assert isinstance(score, float)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_custom_metric_passed_to_model_then_model_can_hyperparameter_tune(model_class, dummy_hyperparameters):
    model = model_class(
        prediction_length=3,
        freq=DUMMY_TS_DATAFRAME.freq,
        hyperparameters={**dummy_hyperparameters, "max_epochs": space.Int(1, 3)},
        eval_metric=CustomMetric(),
    )
    backend = model._get_hpo_backend()
    if backend is RAY_BACKEND:
        # Ray has trouble keeping references to the custom metric in the test namespace. We therefore
        # skip this test.
        pytest.skip()

    if isinstance(model, MultiWindowBacktestingModel):
        val_data = None
    else:
        val_data = DUMMY_TS_DATAFRAME.sort_index()

    num_trials = 2

    hpo_results, _ = model.hyperparameter_tune(
        hyperparameter_tune_kwargs={"num_trials": num_trials, "scheduler": "local", "searcher": "random"},
        time_limit=300,
        train_data=DUMMY_TS_DATAFRAME,
        val_data=val_data,
    )
    assert len(hpo_results) == num_trials
    for result in hpo_results.values():
        assert 1 <= result["hyperparameters"]["max_epochs"] <= 3
        assert np.isfinite(result["val_score"])


@pytest.mark.parametrize("searcher", ["random", "bayes"])
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_searcher_when_ray_backend_used_in_hpo_then_correct_searcher_used(model_class, searcher):
    model = model_class(
        prediction_length=3,
        freq=DUMMY_TS_DATAFRAME.freq,
        hyperparameters={
            "max_epochs": space.Int(1, 3),
            "num_batches_per_epoch": 1,
            "use_fallback_model": False,
        },
        eval_metric="MASE",
    )
    backend = model._get_hpo_backend()
    if backend is not RAY_BACKEND:
        pytest.skip()

    val_data = None if isinstance(model, MultiWindowBacktestingModel) else DUMMY_TS_DATAFRAME
    num_trials = 2

    with mock.patch("ray.tune.Tuner") as mock_tuner:
        try:
            _ = model.hyperparameter_tune(
                hyperparameter_tune_kwargs={"num_trials": num_trials, "scheduler": "FIFO", "searcher": searcher},
                time_limit=300,
                train_data=DUMMY_TS_DATAFRAME,
                val_data=val_data,
            )
        except:
            pass

        ray_searcher_class_name = mock_tuner.call_args[1]["tune_config"].search_alg.__class__.__name__
        assert {
            "bayes": "HyperOpt",
            "random": "BasicVariant",
        }.get(searcher) in ray_searcher_class_name


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_data_contains_missing_values_then_model_can_fit_and_predict(
    temp_model_path, model_class, dummy_hyperparameters
):
    data = DUMMY_TS_DATAFRAME
    prediction_length = 5
    model = model_class(
        freq=data.freq,
        path=temp_model_path,
        prediction_length=prediction_length,
        hyperparameters=dummy_hyperparameters,
    )
    model.fit(
        train_data=data,
        val_data=None if isinstance(model, MultiWindowBacktestingModel) else data,
    )
    predictions = model.predict(data)
    assert not predictions.isna().any(axis=None) and all(predictions.item_ids == data.item_ids)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_fit_and_predict_called_then_train_val_and_test_data_is_preprocessed(
    temp_model_path, model_class, dummy_hyperparameters
):
    train_data = DUMMY_TS_DATAFRAME.copy()
    model = model_class(freq=train_data.freq, path=temp_model_path, hyperparameters=dummy_hyperparameters)
    model.initialize()
    preprocessed_data = train_data + 5.0
    model_tags = model._get_tags()
    expected_train_data = preprocessed_data if model_tags["can_use_train_data"] else train_data
    expected_val_data = preprocessed_data if model_tags["can_use_val_data"] else train_data
    # We need the ugly line break because Python <3.10 does not support parentheses for context managers
    with (
        mock.patch.object(model, "preprocess") as mock_preprocess,
        mock.patch.object(model, "_fit") as mock_fit,
        mock.patch.object(model, "_predict") as mock_predict,
    ):
        mock_preprocess.return_value = preprocessed_data, None
        model.fit(train_data=train_data, val_data=train_data)
        fit_kwargs = mock_fit.call_args[1]
        model_train_data = fit_kwargs["train_data"]
        model_val_data = fit_kwargs["val_data"]
        assert model_train_data.equals(expected_train_data)
        assert model_val_data.equals(expected_val_data)

        model.predict(train_data)
        model_predict_data = mock_predict.call_args[1]["data"]
        assert model_predict_data.equals(preprocessed_data)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_given_model_doesnt_support_nan_when_model_fits_then_nans_are_filled(
    temp_model_path, model_class, dummy_hyperparameters
):
    data = get_data_frame_with_item_index(["B", "A", "C", "X"])
    data.iloc[[0, 1, 5, 10, 23, 26, 33, 60]] = float("nan")
    prediction_length = 5
    model = model_class(
        freq=data.freq,
        path=temp_model_path,
        prediction_length=prediction_length,
        hyperparameters=dummy_hyperparameters,
    )

    with mock.patch.object(model, "_fit") as mock_fit:
        model.fit(
            train_data=data,
            val_data=None if isinstance(model, MultiWindowBacktestingModel) else data,
        )
        fit_kwargs = mock_fit.call_args[1]

    model_allows_nan = model._get_tags()["allow_nan"]
    input_contains_nan = fit_kwargs["train_data"].isna().any(axis=None)
    assert model_allows_nan == input_contains_nan


EXPECTED_MODEL_TAGS = [
    "allow_nan",
    "can_refit_full",
    "can_use_train_data",
    "can_use_val_data",
    # Tabular tags - not used by time series models
    "valid_oof",
    "handles_text",
]


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_model_created_then_model_has_all_required_tags(temp_model_path, model_class):
    model = model_class(path=temp_model_path)
    model_tags = model._get_tags()
    for tag in EXPECTED_MODEL_TAGS:
        assert tag in model_tags
    assert len(model_tags) == len(EXPECTED_MODEL_TAGS)


@pytest.mark.parametrize("model_class", CHRONOS_ZERO_SHOT_MODELS + LOCAL_TESTABLE_MODELS)
def test_when_inference_only_model_scores_oof_then_time_limit_is_passed_to_predict(model_class, dummy_hyperparameters):
    data = DUMMY_TS_DATAFRAME
    model = model_class(freq=data.freq, hyperparameters=dummy_hyperparameters)
    time_limit = 94.4
    model.fit(train_data=data, time_limit=time_limit)
    with mock.patch.object(model, "_predict") as mock_predict:
        model.score_and_cache_oof(data)
        assert abs(mock_predict.call_args[1]["time_limit"] - time_limit) < 0.5


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
def test_given_context_has_1_observation_when_model_predicts_then_model_can_predict(
    model_class, prediction_length, trained_models
):
    from autogluon.timeseries.models.local.statsforecast import AbstractProbabilisticStatsForecastModel

    if isinstance(model_class, type) and issubclass(model_class, AbstractProbabilisticStatsForecastModel):
        pytest.skip("StatsForecast models will use fallback model if history has 1 observation")

    model = trained_models[(prediction_length, repr(model_class))]
    data = TimeSeriesDataFrame.from_iterable_dataset(
        [{"target": [1], "start": pd.Period("2020-01-01", freq="D")} for _ in range(5)]
    )
    predictions = model.predict(data)
    assert len(predictions) == data.num_items * prediction_length


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_itemid_has_string_dtype_then_model_can_predict(model_class, trained_models):
    model = trained_models[(5, repr(model_class))]
    data = DUMMY_TS_DATAFRAME.copy()
    # Convert item_id level to pd.StringDtype()
    data.index = data.index.set_levels(data.index.levels[0].astype(pd.StringDtype()), level="item_id")
    predictions = model.predict(data)
    assert isinstance(predictions, TimeSeriesDataFrame)
    assert len(predictions) == predictions.num_items * model.prediction_length


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_target_scaler_is_used_then_model_can_fit_and_predict(
    model_class, dummy_hyperparameters, df_with_covariates_and_metadata
):
    data, covariate_metadata = df_with_covariates_and_metadata
    model = model_class(freq=data.freq, hyperparameters={"target_scaler": "min_max", **dummy_hyperparameters})
    model.fit(train_data=data)
    predictions = model.predict(data)
    assert isinstance(predictions, TimeSeriesDataFrame)
    assert not predictions.isna().any(axis=None)
    assert len(predictions) == predictions.num_items * model.prediction_length


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("target_scaler", [None, "standard"])
def test_when_covariate_regressor_is_used_then_model_can_fit_and_predict(
    model_class, dummy_hyperparameters, target_scaler, df_with_covariates_and_metadata
):
    prediction_length = 3
    data, covariate_metadata = df_with_covariates_and_metadata
    train_data, test_data = data.train_test_split(prediction_length)
    model = model_class(
        freq=train_data.freq,
        prediction_length=prediction_length,
        hyperparameters={"covariate_regressor": "LR", "target_scaler": target_scaler, **dummy_hyperparameters},
        metadata=covariate_metadata,
    )
    model.fit(train_data=train_data)
    if isinstance(model, MultiWindowBacktestingModel):
        regressor = model.most_recent_model.covariate_regressor
    else:
        regressor = model.covariate_regressor
    assert isinstance(regressor, CovariateRegressor)
    assert regressor.is_fit()

    predictions = model.predict(
        train_data,
        known_covariates=test_data.slice_by_timestep(-prediction_length, None),
    )
    assert isinstance(predictions, TimeSeriesDataFrame)
    assert not predictions.isna().any(axis=None)
    assert len(predictions) == predictions.num_items * model.prediction_length
    assert set(predictions.columns) == set(["mean"] + [str(q) for q in model.quantile_levels])
