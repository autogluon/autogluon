import math
import os
from itertools import chain

import numpy as np
import pytest

from autogluon.timeseries.models.ensemble.array_based.models import MedianEnsemble, TabularEnsemble
from autogluon.timeseries.models.ensemble.array_based.regressor import (
    MedianEnsembleRegressor,
    TabularEnsembleRegressor,
)

from ....common import get_data_frame_with_item_index, get_prediction_for_df

PREDICTIONS = get_prediction_for_df(get_data_frame_with_item_index(["1", "2", "A", "B"]))


class TestEnsembleModels:
    @pytest.fixture(params=[MedianEnsemble, TabularEnsemble])
    def model(self, request):
        yield request.param()

    @pytest.fixture()
    def ensemble_data(self):
        index = ["1", "2", "A", "B"]
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

    def test_given_model_when_get_ensemble_regressor_called_then_correct_regressor_returned(self, model):
        regressor = model._get_ensemble_regressor()

        if isinstance(model, MedianEnsemble):
            assert isinstance(regressor, MedianEnsembleRegressor)
        elif isinstance(model, TabularEnsemble):
            assert isinstance(regressor, TabularEnsembleRegressor)

    def test_given_model_when_initialized_then_ensemble_regressor_is_none(self, model):
        assert model.ensemble_regressor is None

    def test_given_model_when_fit_called_then_ensemble_regressor_created_and_fitted(self, model, ensemble_data):
        model.prediction_length = 5
        model.fit(**ensemble_data)

        assert model.ensemble_regressor is not None
        if isinstance(model, MedianEnsemble):
            assert isinstance(model.ensemble_regressor, MedianEnsembleRegressor)
        elif isinstance(model, TabularEnsemble):
            assert isinstance(model.ensemble_regressor, TabularEnsembleRegressor)

    def test_given_fitted_model_when_predict_called_then_prediction_returned(self, model, ensemble_data):
        model.prediction_length = 5
        model.fit(**ensemble_data)

        data = {
            "dummy_model": PREDICTIONS,
            "dummy_model_2": PREDICTIONS * 2,
        }

        result = model._predict(data)
        assert result is not None
        assert len(result) > 0


class TestTabularEnsemble:
    @pytest.fixture()
    def ensemble_data(self):
        index = ["1", "2", "A", "B"]
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

    @pytest.mark.parametrize(
        "tabular_hyperparameters",
        [
            {"GBM": {}},
            {"GBM": {}, "RF": {}},
        ],
    )
    def test_given_tabular_hyperparameters_when_fit_called_then_hyperparameters_passed_to_predictor(
        self, tabular_hyperparameters, ensemble_data
    ):
        model = TabularEnsemble(hyperparameters={"tabular_hyperparameters": tabular_hyperparameters})
        model.prediction_length = 5
        model.fit(**ensemble_data)

        regressor = model.ensemble_regressor
        assert isinstance(regressor, TabularEnsembleRegressor)
        assert regressor.tabular_hyperparameters == tabular_hyperparameters

    def test_given_quantile_levels_when_fit_called_then_correct_quantile_levels_used(self, ensemble_data):
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # test data has fixed quantile levels
        model = TabularEnsemble()
        model.quantile_levels = quantile_levels
        model.prediction_length = 5
        model.fit(**ensemble_data)

        regressor = model.ensemble_regressor
        assert isinstance(regressor, TabularEnsembleRegressor)
        assert regressor.quantile_levels == quantile_levels
        assert np.array_equal(regressor.predictor.quantile_levels, quantile_levels)  # type: ignore

    @pytest.mark.parametrize(
        "quantile_levels,expected_median_idx",
        [
            ([0.5], 0),
            ([0.1, 0.5, 0.9], 1),
            ([0.1, 0.3, 0.7, 0.9], 2),  # closest to 0.5 is 0.7 at index 2
            ([0.6, 0.7, 0.8], 0),  # closest to 0.5 is 0.6 at index 0
        ],
    )
    def test_given_quantile_levels_when_get_median_quantile_index_called_then_correct_index_returned(
        self, quantile_levels, expected_median_idx, tmp_path
    ):
        regressor = TabularEnsembleRegressor(path=str(tmp_path), quantile_levels=quantile_levels)
        median_idx = regressor._get_median_quantile_index()
        assert median_idx == expected_median_idx

    @pytest.mark.parametrize("num_windows", [1, 2])
    @pytest.mark.parametrize("num_items", [1, 5])
    @pytest.mark.parametrize("prediction_length", [1, 7])
    @pytest.mark.parametrize("num_models", [1, 15])
    def test_when_get_feature_df_called_then_columns_in_correct_order(
        self, num_windows, num_items, prediction_length, num_models, tmp_path
    ):
        quantile_levels = [0.1, 0.5, 0.9]
        regressor = TabularEnsembleRegressor(path=str(tmp_path), quantile_levels=quantile_levels)

        leading_dims = (num_windows, num_items, prediction_length)
        num_tabular_items = math.prod(leading_dims)

        # initialize mean and quantile predictions to known base cases
        mean_preds = np.full((*leading_dims, 1, num_models), 5.0)
        quantile_preds = np.zeros((*leading_dims, 3, num_models))
        for i, q in enumerate(quantile_levels):
            quantile_preds[:, :, :, i, :] = q

        # multiply base cases with tabular item indices
        factor = np.arange(num_tabular_items).reshape(leading_dims + (1, 1))
        mean_preds *= factor
        quantile_preds *= factor

        # convert to feature data frame
        feature_df = regressor._get_feature_df(mean_preds, quantile_preds)

        assert feature_df.shape == (num_tabular_items, num_models * (1 + 3))

        # get expected column names and base_Values
        label_and_expected_per_model = [
            [(f"model_{i}_mean", 5.0)] + [(f"model_{i}_q{q}", q) for q in quantile_levels] for i in range(num_models)
        ]
        label_and_expected_per_output = list(zip(*label_and_expected_per_model))  # transpose
        columns_and_expected = list(chain.from_iterable(label_and_expected_per_output))

        # check
        assert list(feature_df.columns) == [col for col, _ in columns_and_expected]

        for i in range(num_tabular_items):
            row = feature_df.iloc[i]
            for col, expected in columns_and_expected:
                assert row[col] == expected * factor.ravel()[i]

    def test_given_tabular_ensemble_when_path_provided_then_regressor_gets_correct_path(self, tmp_path):
        model_path = os.path.join(str(tmp_path), "test_model")
        model = TabularEnsemble(path=model_path)

        regressor = model._get_ensemble_regressor()
        expected_path = os.path.join(model.path, "ensemble_regressor")

        assert regressor.path == expected_path

    def test_given_tabular_ensemble_when_fitted_then_regressor_path_exists(self, ensemble_data, tmp_path):
        model = TabularEnsemble(path=str(tmp_path), prediction_length=5)
        model.fit(**ensemble_data, time_limit=10)

        regressor_path = os.path.join(model.path, "ensemble_regressor")
        assert os.path.exists(regressor_path)
        assert model.ensemble_regressor.path == regressor_path
        assert os.path.exists(os.path.join(regressor_path, "predictor.pkl"))

    @pytest.mark.parametrize("save", [True, False])
    def test_given_regressor_when_predict_without_fit_then_error_raised(self, save, tmp_path):
        model = TabularEnsemble(path=str(tmp_path), prediction_length=5)
        if save:
            model.save()

        test_predictions = {
            "model1": PREDICTIONS,
            "model2": PREDICTIONS * 1.1,
        }

        with pytest.raises(ValueError, match="Ensemble model has not been fitted yet"):
            model.predict(test_predictions)

    def test_given_fitted_tabular_ensemble_when_deleted_and_loaded_then_can_predict(self, ensemble_data, tmp_path):
        model = TabularEnsemble(path=str(tmp_path), prediction_length=5)
        model.fit(**ensemble_data, time_limit=10)

        test_predictions = {
            "dummy_model": PREDICTIONS,
            "dummy_model_2": PREDICTIONS * 2,
        }
        original_result = model.predict(test_predictions)

        saved_path = model.save()
        del model

        loaded_model = TabularEnsemble.load(saved_path)
        loaded_result = loaded_model.predict(test_predictions)

        np.testing.assert_array_almost_equal(original_result.values, loaded_result.values)
