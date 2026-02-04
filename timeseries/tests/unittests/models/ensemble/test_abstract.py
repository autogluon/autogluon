from unittest import mock

import pytest

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel

from ...common import DUMMY_TS_DATAFRAME, PREDICTIONS_FOR_DUMMY_TS_DATAFRAME, get_data_frame_with_item_index


class DummyEnsembleModel(AbstractTimeSeriesEnsembleModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_names = []

    def _fit(self, predictions_per_window, data_per_window, model_scores=None, time_limit=None, **kwargs):
        self._model_names = list(predictions_per_window.keys())

    def _predict(self, data, **kwargs):
        return PREDICTIONS_FOR_DUMMY_TS_DATAFRAME

    def remap_base_models(self, model_refit_map: dict[str, str]) -> None:
        pass

    @property
    def model_names(self) -> list[str]:
        return self._model_names


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
            "model_scores": {"dummy_model": -25.0, "dummy_model_2": -15.0},
        }

    def test_given_model_when_fit_called_with_small_time_limit_then_exception_raised(self, model, ensemble_data):
        """Test that a nonpositive time limit causes a TimeLimitExceeded exception."""
        with pytest.raises(TimeLimitExceeded):
            model.fit(**ensemble_data, time_limit=0)

    def test_given_model_when_fit_called_with_missing_data_windows_then_value_error_raised(self, model, ensemble_data):
        window_1 = get_data_frame_with_item_index(["A", "B", "C"], start_date="2022-01-01", freq="D")
        window_2 = get_data_frame_with_item_index(["A", "B", "C"], start_date="2022-01-04", freq="D")
        ensemble_data["data_per_window"] = [window_1, window_2]
        with pytest.raises(ValueError, match="predictions are unavailable for some validation windows"):
            _ = model.fit(**ensemble_data, time_limit=10)

    def test_given_model_when_fit_called_with_single_data_frame_then_value_error_raised(self, model, ensemble_data):
        ensemble_data["data_per_window"] = DUMMY_TS_DATAFRAME
        with pytest.raises(ValueError, match="should contain ground truth for each validation window"):
            _ = model.fit(**ensemble_data, time_limit=10)

    def test_given_model_when_fit_called_then_internal_fit_method_called_correctly(self, model, ensemble_data):
        with mock.patch.object(model, "_fit") as mock_fit:
            _ = model.fit(**ensemble_data, time_limit=10)
            mock_fit.assert_called_once()
            assert mock_fit.call_args.kwargs["time_limit"] == 10
            assert mock_fit.call_args.kwargs["predictions_per_window"] is ensemble_data["predictions_per_window"]
            assert mock_fit.call_args.kwargs["data_per_window"] is ensemble_data["data_per_window"]
