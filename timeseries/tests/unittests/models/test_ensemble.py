import itertools
from unittest import mock

import pytest

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.models import SeasonalNaiveModel
from autogluon.timeseries.models.ensemble.abstract_timeseries_ensemble import AbstractTimeSeriesEnsembleModel
from autogluon.timeseries.models.ensemble.greedy_ensemble import TimeSeriesGreedyEnsemble

from ..common import DUMMY_TS_DATAFRAME, PREDICTIONS_FOR_DUMMY_TS_DATAFRAME, get_data_frame_with_item_index


class DummyEnsembleModel(AbstractTimeSeriesEnsembleModel):
    def _fit_ensemble(self, predictions_per_window, data_per_window, time_limit=None, **kwargs):
        return self

    def predict(self, data, **kwargs):
        return PREDICTIONS_FOR_DUMMY_TS_DATAFRAME


class TestAbstractTimeSeriesEnsembleModel:
    """Test the methods that are common to all ensemble models."""

    @pytest.fixture()
    def model(self):
        yield DummyEnsembleModel()

    @pytest.fixture()
    def ensemble_data(self):
        yield {
            "predictions_per_window": {
                "dummy_model": [PREDICTIONS_FOR_DUMMY_TS_DATAFRAME],
                "dummy_model_2": [PREDICTIONS_FOR_DUMMY_TS_DATAFRAME],
            },
            "data_per_window": [DUMMY_TS_DATAFRAME],
        }

    def test_given_model_when_fit_called_with_small_time_limit_then_exception_raised(self, model, ensemble_data):
        """Test that a nonpositive time limit causes a TimeLimitExceeded exception."""
        with pytest.raises(TimeLimitExceeded):
            model.fit_ensemble(**ensemble_data, time_limit=0)

    def test_given_model_when_fit_called_with_missing_data_windows_then_value_error_raised(self, model, ensemble_data):
        window_1 = get_data_frame_with_item_index(["A", "B", "C"], start_date="2022-01-01", freq="D")
        window_2 = get_data_frame_with_item_index(["A", "B", "C"], start_date="2022-01-04", freq="D")
        ensemble_data["data_per_window"] = [window_1, window_2]
        with pytest.raises(ValueError, match="predictions are unavailable for some validation windows"):
            _ = model.fit_ensemble(**ensemble_data, time_limit=10)

    def test_given_model_when_fit_ensemble_called_with_single_data_frame_then_value_error_raised(
        self, model, ensemble_data
    ):
        ensemble_data["data_per_window"] = DUMMY_TS_DATAFRAME
        with pytest.raises(ValueError, match="should contain ground truth for each validation window"):
            _ = model.fit_ensemble(**ensemble_data, time_limit=10)

    def test_given_model_when_fit_ensemble_called_then_internal_fit_method_called_correctly(
        self, model, ensemble_data
    ):
        with mock.patch.object(model, "_fit_ensemble") as mock_fit_ensemble:
            _ = model.fit_ensemble(**ensemble_data, time_limit=10)
            mock_fit_ensemble.assert_called_once()
            assert mock_fit_ensemble.call_args.kwargs["time_limit"] == 10
            assert (
                mock_fit_ensemble.call_args.kwargs["predictions_per_window"] is ensemble_data["predictions_per_window"]
            )
            assert mock_fit_ensemble.call_args.kwargs["data_per_window"] is ensemble_data["data_per_window"]


class TestAllTimeSeriesEnsembleModels:
    """Test that all ensemble models can be instantiated."""

    @pytest.fixture(params=[TimeSeriesGreedyEnsemble])
    def model_constructor(self, request):
        yield request.param

    @pytest.fixture(params=itertools.product([1, 3], [1, 3], [1, 3]))
    def predictions_data_and_prediction_length(self, request):
        num_windows, num_models, prediction_length = request.param
        data = get_data_frame_with_item_index(["A", "B", "C"], start_date="2022-01-01", freq="D", data_length=120)
        data_per_window = [data.slice_by_timestep(end_index=-i * 10) for i in range(num_windows, 0, -1)]

        preds_per_window = {}

        for s in range(1, num_models + 1):
            preds_per_window[f"SNaive{s}"] = [
                SeasonalNaiveModel(
                    prediction_length=prediction_length,
                    hyperparameters={
                        "seasonal_period": s,
                        "n_jobs": 1,
                    },
                )
                .fit(d)
                .predict(d)
                for d in data_per_window
            ]

        yield (
            preds_per_window,
            data_per_window,
            prediction_length,
        )

    def test_ensemble_models_can_be_initialized(self, model_constructor):
        try:
            model_constructor()
        except:
            pytest.fail(f"Could not initialize {model_constructor}")

    def test_ensemble_models_can_fit_and_predict(self, model_constructor, predictions_data_and_prediction_length):
        predictions_per_window, data_per_window, prediction_length = predictions_data_and_prediction_length

        model = model_constructor(prediction_length=prediction_length)
        try:
            model.fit_ensemble(predictions_per_window=predictions_per_window, data_per_window=data_per_window)
            model.predict({k: v[0] for k, v in predictions_per_window.items()})
        except:
            pytest.fail(f"Could not fit and predict with {model_constructor}")

    def test_when_ensemble_models_predict_then_prediction_horizon_aligns_with_input(
        self, model_constructor, predictions_data_and_prediction_length
    ):
        predictions_per_window, data_per_window, prediction_length = predictions_data_and_prediction_length

        model = model_constructor(prediction_length=prediction_length)
        model.fit_ensemble(predictions_per_window=predictions_per_window, data_per_window=data_per_window)
        predictions = model.predict({k: v[0] for k, v in predictions_per_window.items()})

        first_model_prediction = next(iter(predictions_per_window.values()))[0]
        assert all(predictions.index == first_model_prediction.index)

    def test_when_ensemble_models_predict_then_prediction_contains_no_nans(
        self, model_constructor, predictions_data_and_prediction_length
    ):
        predictions_per_window, data_per_window, prediction_length = predictions_data_and_prediction_length

        model = model_constructor(prediction_length=prediction_length)
        model.fit_ensemble(predictions_per_window=predictions_per_window, data_per_window=data_per_window)
        predictions = model.predict({k: v[0] for k, v in predictions_per_window.items()})

        assert not predictions.isna().any(axis=None)

    def test_given_model_when_fit_ensemble_called_then_internal_fit_method_called_correctly(
        self, model_constructor, predictions_data_and_prediction_length
    ):
        predictions_per_window, data_per_window, prediction_length = predictions_data_and_prediction_length
        model = model_constructor(prediction_length=prediction_length)

        with mock.patch.object(model, "_fit_ensemble") as mock_fit_ensemble:
            _ = model.fit_ensemble(
                predictions_per_window=predictions_per_window, data_per_window=data_per_window, time_limit=10
            )
            mock_fit_ensemble.assert_called_once()
            assert mock_fit_ensemble.call_args.kwargs["time_limit"] == 10
            assert mock_fit_ensemble.call_args.kwargs["predictions_per_window"] is predictions_per_window
            assert mock_fit_ensemble.call_args.kwargs["data_per_window"] is data_per_window


class TestTimeSeriesGreedyEnsemble:
    def test_when_some_base_models_fail_during_prediction_then_ensemble_raises_runtime_error(self):
        base_model = SeasonalNaiveModel(prediction_length=1, freq=DUMMY_TS_DATAFRAME.freq)
        base_model.fit(train_data=DUMMY_TS_DATAFRAME)
        base_model_preds = base_model.predict(DUMMY_TS_DATAFRAME)

        ensemble = TimeSeriesGreedyEnsemble()
        ensemble.model_to_weight = {"ARIMA": 0.5, "SeasonalNaive": 0.5}

        with pytest.raises(RuntimeError):
            ensemble.predict(data={"ARIMA": None, "SeasonalNaive": base_model_preds})
