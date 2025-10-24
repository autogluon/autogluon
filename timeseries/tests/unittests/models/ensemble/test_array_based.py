from unittest import mock

import numpy as np
import pytest

from autogluon.timeseries.models.ensemble.array_based.abstract import ArrayBasedTimeSeriesEnsembleModel
from autogluon.timeseries.models.ensemble.array_based.regressor import EnsembleRegressor

from ...common import get_data_frame_with_item_index, get_data_frame_with_variable_lengths, get_prediction_for_df

PREDICTIONS = get_prediction_for_df(get_data_frame_with_item_index(["1", "2", "A", "B"]))


class DummyEnsembleRegressor(EnsembleRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitted = False

    def fit(self, base_model_predictions: np.ndarray, labels: np.ndarray, **kwargs):
        self.fitted = True
        return self

    def predict(self, base_model_predictions: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Regressor not fitted")
        return np.mean(base_model_predictions, axis=-1)


class DummyArrayBasedEnsembleModel(ArrayBasedTimeSeriesEnsembleModel):
    _regressor_type = DummyEnsembleRegressor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_names = []

    @property
    def model_names(self) -> list[str]:
        return self._model_names

    def _fit(self, predictions_per_window, data_per_window, model_scores=None, time_limit=None, **kwargs):
        self._model_names = list(predictions_per_window.keys())
        super()._fit(predictions_per_window, data_per_window, model_scores, time_limit, **kwargs)


class TestArrayBasedTimeSeriesEnsembleModel:
    @pytest.fixture()
    def model(self):
        yield DummyArrayBasedEnsembleModel()

    @pytest.fixture(params=["variable", "fixed"])
    def ensemble_data(self, request):
        index = ["1", "2", "A", "B"]
        if request.param == "variable":
            df = get_data_frame_with_variable_lengths(dict(zip(index, range(20, 20 + 20 * 4, 20))))
        else:
            df = get_data_frame_with_item_index(index)  # type: ignore
        preds = get_prediction_for_df(df)

        yield {
            "predictions_per_window": {
                "dummy_model": [preds],
                "dummy_model_2": [preds * 2],
            },
            "data_per_window": [df],
            "model_scores": {"dummy_model": -2.5, "dummy_model_2": -1.0},
        }

    def test_given_model_when_initialized_then_default_hyperparameters_set(self, model):
        expected_defaults = {
            "isotonization": "sort",
            "detect_and_ignore_failures": True,
        }
        assert model._get_default_hyperparameters() == expected_defaults

    def test_given_model_when_initialized_then_ensemble_regressor_is_none(self, model):
        assert model.ensemble_regressor is None

    def test_given_dataframe_when_to_array_called_then_array_has_correct_shape(self):
        df = get_data_frame_with_item_index(["A", "B"], data_length=3, freq="D")
        array = DummyArrayBasedEnsembleModel.to_array(df)

        expected_shape = (2, 3, len(df.columns))  # (items, prediction_length, outputs)
        assert array.shape == expected_shape

    def test_given_model_when_fit_called_then_ensemble_regressor_created_and_fitted(self, model, ensemble_data):
        model.prediction_length = 5  # Match the prediction data
        model.fit(**ensemble_data)

        assert model.ensemble_regressor is not None
        assert isinstance(model.ensemble_regressor, DummyEnsembleRegressor)
        assert model.ensemble_regressor.fitted

    def test_given_model_when_get_base_model_predictions_array_called_with_empty_dict_then_error_raised(self, model):
        with pytest.raises(ValueError, match="No base model predictions are provided"):
            model._get_base_model_predictions_array({})

    def test_given_model_when_get_base_model_predictions_array_called_then_correct_array_shape_returned(self, model):
        model.prediction_length = 5  # Match the prediction data
        predictions = {
            "model1": [PREDICTIONS],
            "model2": [PREDICTIONS * 2],
        }

        array = model._get_base_model_predictions_array(predictions)

        assert array.shape == (1, 4, 5, 10, 2)  # (windows, items, prediction_length, outputs, models)

        # Check content
        model1_array = model.to_array(PREDICTIONS)
        assert np.allclose(array[0, :, :, :, 0], model1_array)
        assert np.allclose(array[0, :, :, :, 1], 2 * model1_array)

    def test_given_model_when_get_base_model_predictions_array_called_with_single_window_then_correct_array_shape_returned(
        self, model
    ):
        model.prediction_length = 5  # Match the prediction data
        predictions = {
            "model1": PREDICTIONS,
            "model2": PREDICTIONS * 2,
        }

        array = model._get_base_model_predictions_array(predictions)

        assert array.ndim == 5
        assert array.shape[-1] == 2  # 2 models

    def test_given_unfitted_model_when_predict_called_then_error_raised(self, model):
        data = {"model1": PREDICTIONS}

        with pytest.raises(ValueError, match="Ensemble model has not been fitted yet"):
            model._predict(data)

    def test_given_model_when_fit_called_with_hyperparameters_then_regressor_receives_hyperparameters(
        self, model, ensemble_data
    ):
        model.prediction_length = 5  # Match the prediction data
        model.hyperparameters = {"isotonization": False}

        with mock.patch.object(DummyEnsembleRegressor, "__init__", return_value=None) as mock_init:
            model.fit(**ensemble_data)

            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args.kwargs
            assert "isotonization" in call_kwargs

    def test_given_model_when_fit_called_then_regressor_receives_correct_array(self, model, ensemble_data):
        model.prediction_length = 5  # match the prediction data
        with mock.patch.object(DummyEnsembleRegressor, "fit") as mock_fit:
            model.fit(**ensemble_data)

            mock_fit.assert_called_once()
            call_args = mock_fit.call_args
            base_model_predictions = call_args[1]["base_model_predictions"]
            labels = call_args[1]["labels"]

            assert base_model_predictions.shape == (1, 4, 5, 10, 2)
            assert labels.shape == (1, 4, 5, 1)  # window, items, prediction_length, 1 (target)

    def test_given_model_when_isotonize_called_with_sort_then_quantiles_sorted(self, model, ensemble_data):
        model.prediction_length = 1
        model.hyperparameters = {"isotonization": "sort"}
        model.fit(**ensemble_data)

        unsorted_array = np.array([[[[3.0, 1.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0]]]])
        with mock.patch.object(model.ensemble_regressor, "predict", return_value=unsorted_array):
            data = {
                "dummy_model": PREDICTIONS.iloc[:1],
                "dummy_model_2": PREDICTIONS.iloc[:1],
            }

            result = model._predict(data)

            expected_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            assert np.allclose(result.iloc[0].values, expected_sorted)

    def test_given_model_when_remap_base_models_called_then_model_names_updated(self, model, ensemble_data):
        model.fit(**ensemble_data)

        original_names = model.model_names.copy()
        model_refit_map = {original_names[0]: f"{original_names[0]}_v2"}

        model.remap_base_models(model_refit_map)

        expected_names = [f"{original_names[0]}_v2"] + original_names[1:]
        assert model.model_names == expected_names

    def test_given_model_when_detect_and_ignore_failures_enabled_then_nan_models_filtered(self, model):
        predictions_per_window = {
            "good_model": [PREDICTIONS],
            "failed_model": [PREDICTIONS * 2],
            "another_good_model": [PREDICTIONS * 3],
        }
        model_scores = {
            "good_model": -0.1,
            "failed_model": float("nan"),  # Failed model
            "another_good_model": -0.2,
        }

        filtered = model._filter_failed_models(predictions_per_window, model_scores)

        assert set(filtered.keys()) == {"good_model", "another_good_model"}
        assert "failed_model" not in filtered

    def test_given_model_when_detect_and_ignore_failures_disabled_then_all_models_kept(self, model):
        model_disabled = DummyArrayBasedEnsembleModel(hyperparameters={"detect_and_ignore_failures": False})
        predictions_per_window = {
            "good_model": [PREDICTIONS],
            "failed_model": [PREDICTIONS * 2],
        }
        model_scores = {
            "good_model": -0.1,
            "failed_model": float("nan"),
        }

        filtered = model_disabled._filter_failed_models(predictions_per_window, model_scores)

        assert set(filtered.keys()) == {"good_model", "failed_model"}

    def test_given_model_when_all_models_failed_then_error_raised(self, model):
        predictions_per_window = {
            "failed_model1": [PREDICTIONS],
            "failed_model2": [PREDICTIONS * 2],
        }
        model_scores = {
            "failed_model1": float("nan"),
            "failed_model2": float("inf"),
        }

        with pytest.raises(ValueError, match="All models have NaN scores"):
            model._filter_failed_models(predictions_per_window, model_scores)

    def test_given_model_when_fit_with_failed_models_then_only_good_models_used(self, model):
        ensemble_data = {
            "data_per_window": [PREDICTIONS],
            "predictions_per_window": {
                "good_model": [PREDICTIONS],
                "failed_model": [PREDICTIONS * 2],
                "another_good_model": [PREDICTIONS * 3],
            },
            "model_scores": {
                "good_model": -0.1,
                "failed_model": float("nan"),
                "another_good_model": -0.2,
            },
        }

        model.fit(**ensemble_data)

        assert set(model.model_names) == {"good_model", "another_good_model"}
        assert "failed_model" not in model.model_names

    def test_given_model_when_model_has_loss_10x_median_then_filtered_out(self, model):
        predictions_per_window = {
            "good_model1": [PREDICTIONS],
            "good_model2": [PREDICTIONS * 2],
            "failed_model": [PREDICTIONS * 3],
        }
        model_scores = {
            "good_model1": -0.1,
            "good_model2": -0.2,
            "failed_model": -5.0,  # (> 10x median)
        }

        filtered = model._filter_failed_models(predictions_per_window, model_scores)

        assert set(filtered.keys()) == {"good_model1", "good_model2"}
        assert "failed_model" not in filtered
