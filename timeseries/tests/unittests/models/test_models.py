"""Unit tests and utils common to all models"""

import sys
import uuid
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from flaky import flaky

from autogluon.common import space
from autogluon.core.hpo.constants import RAY_BACKEND
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.models.multi_window import MultiWindowBacktestingModel
from autogluon.timeseries.regressor import CovariateRegressor

from ..common import (
    DUMMY_TS_DATAFRAME,
    CustomMetric,
    dict_equal_primitive,
    get_data_frame_with_item_index,
    mask_entries,
    to_supported_pandas_freq,
)

TESTABLE_PREDICTION_LENGTHS = [1, 5]


class TestAllModelsInitialization:
    EXPECTED_MODEL_TAGS = [
        "allow_nan",
        "can_refit_full",
        "can_use_train_data",
        "can_use_val_data",
        # Tabular tags - not used by time series models
        "valid_oof",
        "handles_text",
        "can_estimate_memory_usage_static",
    ]

    def test_models_can_be_initialized(self, model_class, temp_model_path):
        model = model_class(path=temp_model_path, freq="h", prediction_length=24)
        assert isinstance(model, AbstractTimeSeriesModel)

    def test_when_model_created_then_model_has_all_required_tags(self, model_class, temp_model_path):
        model = model_class(path=temp_model_path)
        model_tags = model._get_tags()
        for tag in self.EXPECTED_MODEL_TAGS:
            assert tag in model_tags
        assert len(model_tags) == len(self.EXPECTED_MODEL_TAGS)

    def test_when_get_hyperparameters_called_then_copy_is_returned(self, model_class, temp_model_path):
        hp = {}
        model = model_class(path=temp_model_path, freq="h", prediction_length=24, hyperparameters=hp)
        assert model.get_hyperparameters() is not hp


class TestAllModelsPostTraining:
    @pytest.fixture(scope="class", params=TESTABLE_PREDICTION_LENGTHS)
    def trained_model(self, request, model_class, tmp_path_factory):
        model = model_class(
            path=str(tmp_path_factory.mktemp(str(uuid.uuid4())[:6])),
            freq="h",
            prediction_length=request.param,
        )

        model.fit(train_data=DUMMY_TS_DATAFRAME)
        model.score_and_cache_oof(DUMMY_TS_DATAFRAME, store_val_score=True, store_predict_time=True)

        yield model

    def test_when_score_called_then_scores_can_be_computed(self, trained_model):
        score = trained_model.score(DUMMY_TS_DATAFRAME)
        assert isinstance(score, float)

    def test_when_val_score_accessed_then_value_is_returned(self, trained_model):
        assert isinstance(trained_model.val_score, float)

    def test_when_predict_time_accessed_then_value_is_returned(self, trained_model):
        assert isinstance(trained_model.predict_time, float)

    def test_given_score_and_cache_oof_called_when_get_oof_predictions_called_then_oof_predictions_are_available(
        self,
        trained_model,
    ):
        if isinstance(trained_model, MultiWindowBacktestingModel):
            pytest.skip()

        oof_predictions = trained_model.get_oof_predictions()[0]
        assert isinstance(oof_predictions, TimeSeriesDataFrame)
        oof_score = trained_model._score_with_predictions(DUMMY_TS_DATAFRAME, oof_predictions)
        assert isinstance(oof_score, float)

    def test_when_score_called_then_model_receives_truncated_data(self, trained_model):
        with mock.patch.object(trained_model, "predict") as patch_method:
            # Mock breaks the internals of the `score` method
            try:
                trained_model.score(DUMMY_TS_DATAFRAME)
            except AssertionError:
                pass

            (call_df,) = patch_method.call_args[0]

            for j in DUMMY_TS_DATAFRAME.item_ids:
                truncated_data = DUMMY_TS_DATAFRAME.loc[j][: -trained_model.prediction_length]
                assert np.allclose(call_df.loc[j], truncated_data, equal_nan=True)

    def test_when_models_saved_then_they_can_be_loaded(self, trained_model):
        trained_model.save()

        loaded_model = trained_model.__class__.load(path=trained_model.path)

        assert dict_equal_primitive(trained_model.get_hyperparameters(), loaded_model.get_hyperparameters())
        assert trained_model.covariate_metadata == loaded_model.covariate_metadata
        for orig_oof_pred, loaded_oof_pred in zip(
            trained_model.get_oof_predictions(), loaded_model.get_oof_predictions()
        ):
            assert orig_oof_pred.equals(loaded_oof_pred)

    @pytest.mark.parametrize(
        "test_data",
        [
            get_data_frame_with_item_index(["A", "B"], data_length=15),
            mask_entries(get_data_frame_with_item_index(["C", "D"], data_length=60)),
            get_data_frame_with_item_index(["A"], data_length=10),
            get_data_frame_with_item_index([0, 1, 2, 3], data_length=15),
        ],
    )
    def test_when_predict_called_then_predictions_align_index_aligns_with_expected_index(
        self, trained_model, test_data
    ):
        max_hour_in_test = test_data.index.levels[1].max().hour

        predictions = trained_model.predict(test_data)
        predicted_item_index = predictions.item_ids
        min_hour_in_pred = predictions.index.levels[1].min().hour

        assert isinstance(predictions, TimeSeriesDataFrame)
        assert all(predicted_item_index == test_data.item_ids)
        assert all(len(predictions.loc[i]) == trained_model.prediction_length for i in predicted_item_index)

        assert min_hour_in_pred == max_hour_in_test + 1

    def test_when_context_has_one_observation_then_model_can_predict(self, trained_model):
        from autogluon.timeseries.models.local.statsforecast import AbstractProbabilisticStatsForecastModel

        if isinstance(trained_model, AbstractProbabilisticStatsForecastModel):
            pytest.skip("StatsForecast models will use fallback model if history has 1 observation")

        data = TimeSeriesDataFrame.from_iterable_dataset(
            [{"target": [1], "start": pd.Period("2020-01-01", freq="D")} for _ in range(5)]
        )
        predictions = trained_model.predict(data)
        assert len(predictions) == data.num_items * trained_model.prediction_length

    def test_when_itemid_has_arrow_string_dtype_then_model_can_predict(self, trained_model):
        data = DUMMY_TS_DATAFRAME.copy()

        # Convert item_id level to pd.StringDtype()
        data.index = data.index.set_levels(data.index.levels[0].astype(pd.StringDtype()), level="item_id")
        predictions = trained_model.predict(data)
        assert isinstance(predictions, TimeSeriesDataFrame)
        assert len(predictions) == predictions.num_items * trained_model.prediction_length

    def test_when_get_info_is_called_then_all_keys_are_present(self, trained_model):
        info = trained_model.get_info()
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


class TestAllModelsWhenHyperparameterTuning:
    @flaky
    @pytest.mark.skipif(
        sys.platform.startswith("win"), reason="HPO tests lead to known issues in Windows platform tests"
    )
    def test_when_hyperparameter_tune_called_then_tuning_output_correct(self, gluonts_model_class, temp_model_path):
        # TODO: add hyperparameter tuning tests for other model classes

        model = gluonts_model_class(
            path=temp_model_path,
            freq="h",
            quantile_levels=[0.1, 0.9],
            hyperparameters={"max_epochs": space.Int(1, 3)},
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

    @pytest.mark.parametrize("searcher", ["random", "bayes"])
    @pytest.mark.skipif(
        sys.platform == "win32" and sys.version_info >= (3, 13), reason="No ray support on Windows with Python 3.13"
    )
    def test_given_searcher_when_ray_backend_used_in_hpo_then_correct_searcher_used(
        self, gluonts_model_class, searcher
    ):
        model = gluonts_model_class(
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

    def test_when_custom_metric_passed_to_model_then_model_can_hyperparameter_tune(self, gluonts_model_class):
        model = gluonts_model_class(
            prediction_length=3,
            freq=DUMMY_TS_DATAFRAME.freq,
            hyperparameters={"max_epochs": space.Int(1, 3)},
            eval_metric=CustomMetric(),
        )
        backend = model._get_hpo_backend()
        if backend is RAY_BACKEND:
            pytest.skip(reason="Ray has trouble keeping references to the custom metric in the test namespace")

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

    def test_when_hyperparameter_spaces_provided_to_init_and_fit_called_then_error_is_raised(
        self, model_class, temp_model_path
    ):
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


class TestAllModelsWhenCustomProblemSpecificationsProvided:
    """Test all models with varying forecast problem specifications such as frequency of the
    time series, quantile levels, item index, etc.
    """

    @pytest.mark.parametrize(
        "quantile_levels",
        [
            [0.1, 0.44, 0.72],
            [0.1, 0.5, 0.9],
        ],
    )
    def test_when_fit_called_then_models_train_and_returned_predictions_have_mean_and_correct_quantiles(
        self, model_class, quantile_levels, temp_model_path
    ):
        model = model_class(
            path=temp_model_path,
            freq="h",
            prediction_length=3,
            quantile_levels=quantile_levels,
        )
        model.fit(train_data=DUMMY_TS_DATAFRAME)
        predictions = model.predict(DUMMY_TS_DATAFRAME, quantile_levels=quantile_levels)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.item_ids
        expected_columns = ["mean"] + [str(q) for q in quantile_levels]
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
        assert (predictions.columns == expected_columns).all()

    @pytest.mark.parametrize("freq", ["D", "h", "s", "ME"])
    def test_when_predict_called_with_custom_frequency_then_predicted_timestamps_align_with_time(
        self, model_class, freq, temp_model_path
    ):
        freq = to_supported_pandas_freq(freq)
        prediction_length = 4
        train_length = 20
        item_id = "A"
        timestamps = pd.date_range(start=pd.Timestamp("2020-01-05 12:05:01"), freq=freq, periods=train_length)
        index = pd.MultiIndex.from_product(
            [(item_id,), timestamps], names=[TimeSeriesDataFrame.ITEMID, TimeSeriesDataFrame.TIMESTAMP]
        )
        train_data = TimeSeriesDataFrame(pd.DataFrame({"target": np.random.rand(train_length)}, index=index))

        model = model_class(
            path=temp_model_path,
            freq=train_data.freq,
            prediction_length=prediction_length,
        )

        model.fit(train_data=train_data)
        predictions = model.predict(train_data)

        offset = pd.tseries.frequencies.to_offset(freq)
        preds_first_item = predictions.loc[item_id]
        for i in range(prediction_length):
            assert offset is not None
            assert preds_first_item.index[i] == timestamps[-1] + offset * (i + 1)

    def test_when_custom_metric_passed_to_model_then_model_can_score(self, model_class):
        model = model_class(
            prediction_length=3,
            freq=DUMMY_TS_DATAFRAME.freq,
            quantile_levels=[0.1, 0.15],
            eval_metric=CustomMetric(),
        )
        model.fit(train_data=DUMMY_TS_DATAFRAME)
        score = model.score(DUMMY_TS_DATAFRAME.sort_index())
        assert isinstance(score, float)

    def test_when_median_not_in_quantile_levels_then_median_is_present_in_raw_predictions(self, model_class):
        data = get_data_frame_with_item_index(["B", "A", "X", "C"])
        model = model_class(
            prediction_length=3,
            quantile_levels=[0.1, 0.15],
            freq=data.freq,
        )
        if isinstance(model, MultiWindowBacktestingModel):
            # Median is present in the predictions of the base model, but not in the MultiWindowBacktestingModel wrapper
            pytest.skip()
        model.fit(train_data=data)

        raw_predictions = model._predict(data)
        assert "0.5" in raw_predictions.columns

    def test_when_median_not_in_quantile_levels_then_median_is_dropped_at_prediction_time(self, model_class):
        model = model_class(
            prediction_length=3,
            quantile_levels=[0.1, 0.15],
            freq=DUMMY_TS_DATAFRAME.freq,
        )
        assert model.must_drop_median
        model.fit(train_data=DUMMY_TS_DATAFRAME)
        final_predictions = model.predict(DUMMY_TS_DATAFRAME)
        assert "0.5" not in final_predictions.columns


class TestAllModelsWhenPreprocessingAndTransformsRequested:
    def test_when_fit_and_predict_called_then_train_val_and_test_data_is_preprocessed(
        self, model_class, temp_model_path
    ):
        train_data = DUMMY_TS_DATAFRAME.copy()
        model = model_class(freq=train_data.freq, path=temp_model_path)
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

    def test_given_model_doesnt_support_nan_when_model_fits_then_nans_are_filled(self, model_class, temp_model_path):
        data = get_data_frame_with_item_index(["B", "A", "C", "X"])
        data.iloc[[0, 1, 5, 10, 23, 26, 33, 60]] = float("nan")
        prediction_length = 5
        model = model_class(
            freq=data.freq,
            path=temp_model_path,
            prediction_length=prediction_length,
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

    def test_when_target_scaler_is_used_then_model_can_fit_and_predict(
        self, model_class, df_with_covariates_and_metadata
    ):
        data, _ = df_with_covariates_and_metadata
        model = model_class(freq=data.freq, hyperparameters={"target_scaler": "min_max"})
        model.fit(train_data=data)
        predictions = model.predict(data)
        assert isinstance(predictions, TimeSeriesDataFrame)
        assert not predictions.isna().any(axis=None)
        assert len(predictions) == predictions.num_items * model.prediction_length

    @pytest.mark.parametrize("target_scaler", [None, "standard"])
    def test_when_covariate_regressor_is_used_then_model_can_fit_and_predict(
        self, model_class, target_scaler, df_with_covariates_and_metadata
    ):
        prediction_length = 3
        data, covariate_metadata = df_with_covariates_and_metadata
        train_data, test_data = data.train_test_split(prediction_length)
        model = model_class(
            freq=train_data.freq,
            prediction_length=prediction_length,
            hyperparameters={"covariate_regressor": "LR", "target_scaler": target_scaler},
            covariate_metadata=covariate_metadata,
        )
        model.fit(train_data=train_data)
        if isinstance(model, MultiWindowBacktestingModel):
            assert model.most_recent_model is not None
            regressor = model.most_recent_model.covariate_regressor
        else:
            regressor = model.covariate_regressor
        assert isinstance(regressor, CovariateRegressor)
        assert regressor.is_fit()

        known_covariates = test_data.slice_by_timestep(-prediction_length, None).drop(columns=["target"])
        predictions = model.predict(
            train_data,
            known_covariates=known_covariates,
        )
        assert isinstance(predictions, TimeSeriesDataFrame)
        assert not predictions.isna().any(axis=None)
        assert len(predictions) == predictions.num_items * model.prediction_length
        assert set(predictions.columns) == set(["mean"] + [str(q) for q in model.quantile_levels])


class TestInferenceOnlyModels:
    def test_when_inference_only_model_scores_oof_then_time_limit_is_passed_to_predict(
        self, inference_only_model_class
    ):
        data = DUMMY_TS_DATAFRAME
        model_kwargs = dict(freq=data.freq)
        base_model = inference_only_model_class(**model_kwargs)
        mw_model = MultiWindowBacktestingModel(model_base=base_model, **model_kwargs)  # type: ignore
        time_limit = 94.4
        with mock.patch.object(type(base_model), "_predict") as mock_predict:
            mock_predict.side_effect = RuntimeError
            try:
                mw_model.fit(train_data=data, time_limit=time_limit)
            except RuntimeError:
                pass
            assert abs(mock_predict.call_args[1]["time_limit"] - time_limit) < 5.0
