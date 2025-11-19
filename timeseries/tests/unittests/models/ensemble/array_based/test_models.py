import math
import shutil
from itertools import chain
from unittest.mock import Mock

import numpy as np
import pytest

from autogluon.timeseries.models.ensemble.array_based.models import (
    MedianEnsemble,
    PerQuantileTabularEnsemble,
    TabularEnsemble,
)
from autogluon.timeseries.models.ensemble.array_based.regressor import (
    MedianEnsembleRegressor,
    PerQuantileTabularEnsembleRegressor,
    TabularEnsembleRegressor,
)

from ....common import get_data_frame_with_item_index, get_prediction_for_df

PREDICTIONS = get_prediction_for_df(get_data_frame_with_item_index(["1", "2", "A", "B"]))


@pytest.fixture()
def ensemble_data():
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


class TestEnsembleModels:
    @pytest.fixture(params=[MedianEnsemble, TabularEnsemble, PerQuantileTabularEnsemble])
    def model(self, request):
        yield request.param()

    def test_given_model_when_get_ensemble_regressor_called_then_correct_regressor_returned(self, model):
        regressor = model._get_ensemble_regressor()

        if isinstance(model, MedianEnsemble):
            assert isinstance(regressor, MedianEnsembleRegressor)
        elif isinstance(model, TabularEnsemble):
            assert isinstance(regressor, TabularEnsembleRegressor)
        elif isinstance(model, PerQuantileTabularEnsemble):
            assert isinstance(regressor, PerQuantileTabularEnsembleRegressor)

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
        elif isinstance(model, PerQuantileTabularEnsemble):
            assert isinstance(model.ensemble_regressor, PerQuantileTabularEnsembleRegressor)

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


class TestTabularEnsembleCommon:
    """Common tests for both TabularEnsemble and PerQuantileTabularEnsemble."""

    @pytest.fixture(params=[TabularEnsemble, PerQuantileTabularEnsemble])
    def ensemble_model_class(self, request):
        return request.param

    @pytest.mark.parametrize("model_hyperparameters", [{}, {"max_depth": 5}])
    def test_given_model_hyperparameters_when_fit_called_then_correct_hyperparameters_are_used(
        self, ensemble_model_class, model_hyperparameters, ensemble_data
    ):
        model = ensemble_model_class(hyperparameters={"model_hyperparameters": model_hyperparameters})
        model.prediction_length = 5
        model.fit(**ensemble_data)

        if ensemble_model_class == TabularEnsemble:
            tabular_model = model.ensemble_regressor.model
        else:
            tabular_model = model.ensemble_regressor.mean_model
        assert model_hyperparameters.items() <= tabular_model.get_params()["hyperparameters"].items()

    def test_given_fitted_ensemble_when_deleted_and_loaded_then_can_predict(
        self, ensemble_model_class, ensemble_data, tmp_path
    ):
        model = ensemble_model_class(path=str(tmp_path), prediction_length=5)
        model.fit(**ensemble_data, time_limit=10)

        test_predictions = {"dummy_model": PREDICTIONS, "dummy_model_2": PREDICTIONS * 2}
        original_result = model.predict(test_predictions)

        saved_path = model.save()
        del model

        loaded_model = ensemble_model_class.load(saved_path)
        loaded_result = loaded_model.predict(test_predictions)

        np.testing.assert_array_almost_equal(original_result.values, loaded_result.values)

    def test_given_fitted_ensemble_saved_when_moved_and_loaded_then_can_predict(
        self, ensemble_model_class, ensemble_data, tmp_path_factory
    ):
        original_dir = tmp_path_factory.mktemp("original")
        moved_dir = tmp_path_factory.mktemp("moved")

        model = ensemble_model_class(path=str(original_dir), prediction_length=5)
        model.fit(**ensemble_data, time_limit=10)

        test_predictions = {"dummy_model": PREDICTIONS, "dummy_model_2": PREDICTIONS * 2}
        original_result = model.predict(test_predictions)

        saved_path = model.save()
        del model

        # Move the entire saved model directory to new location
        moved_path = moved_dir / "moved_model"
        shutil.move(saved_path, str(moved_path))

        # Load from the new location
        loaded_model = ensemble_model_class.load(str(moved_path))
        assert loaded_model.path == str(moved_path)

        loaded_result = loaded_model.predict(test_predictions)

        np.testing.assert_array_almost_equal(original_result.values, loaded_result.values)

    @pytest.mark.parametrize("save", [True, False])
    def test_given_ensemble_when_predict_without_fit_then_error_raised(self, ensemble_model_class, save, tmp_path):
        model = ensemble_model_class(path=str(tmp_path), prediction_length=5)
        if save:
            model.save()

        test_predictions = {"model1": PREDICTIONS, "model2": PREDICTIONS * 1.1}

        with pytest.raises(ValueError, match="Ensemble model has not been fitted yet"):
            model.predict(test_predictions)


class TestTabularEnsemble:
    def test_given_quantile_levels_when_fit_called_then_correct_quantile_levels_used(self, ensemble_data):
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # test data has fixed quantile levels
        model = TabularEnsemble()
        model.quantile_levels = quantile_levels
        model.prediction_length = 5
        model.fit(**ensemble_data)

        regressor = model.ensemble_regressor
        assert isinstance(regressor, TabularEnsembleRegressor)
        assert regressor.quantile_levels == quantile_levels
        assert regressor.model.is_fit()

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
        self, quantile_levels, expected_median_idx
    ):
        regressor = TabularEnsembleRegressor(quantile_levels=quantile_levels, model_name="GBM")
        median_idx = regressor._get_median_quantile_index()
        assert median_idx == expected_median_idx

    @pytest.mark.parametrize("num_windows", [1, 2])
    @pytest.mark.parametrize("num_items", [1, 5])
    @pytest.mark.parametrize("prediction_length", [1, 7])
    @pytest.mark.parametrize("num_models", [1, 15])
    def test_when_get_feature_df_called_then_columns_in_correct_order(
        self, num_windows, num_items, prediction_length, num_models
    ):
        quantile_levels = [0.1, 0.5, 0.9]
        regressor = TabularEnsembleRegressor(quantile_levels=quantile_levels, model_name="GBM")

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

    def test_given_tabular_ensemble_when_fitted_then_model_is_fit(self, ensemble_data, tmp_path):
        model = TabularEnsemble(path=str(tmp_path), prediction_length=5)
        model.fit(**ensemble_data, time_limit=10)

        assert isinstance(model.ensemble_regressor, TabularEnsembleRegressor)
        assert model.ensemble_regressor.model.is_fit()


class TestPerQuantileTabularEnsemble:
    def test_given_quantile_levels_when_fit_called_then_correct_number_of_models_created(self, ensemble_data):
        quantile_levels = [0.1, 0.5, 0.9]
        model = PerQuantileTabularEnsemble(quantile_levels=quantile_levels, prediction_length=5)
        model.fit(**ensemble_data)

        regressor = model.ensemble_regressor
        assert isinstance(regressor, PerQuantileTabularEnsembleRegressor)
        assert len(regressor.quantile_models) == len(quantile_levels)
        assert regressor.mean_model is not None
        assert regressor.mean_model.is_fit()
        assert all(m.is_fit() for m in regressor.quantile_models)

    def test_given_per_quantile_ensemble_when_fitted_then_separate_models_created(self, ensemble_data, tmp_path):
        """Test that separate model instances are created for each quantile and mean."""
        model = PerQuantileTabularEnsemble(path=str(tmp_path), prediction_length=5)
        model.quantile_levels = [0.1, 0.5, 0.9]
        model.fit(**ensemble_data, time_limit=10)

        regressor = model.ensemble_regressor
        assert isinstance(regressor, PerQuantileTabularEnsembleRegressor)

        # Verify separate models exist
        assert len(regressor.quantile_models) == 3
        assert regressor.mean_model is not None

    def test_given_per_quantile_ensemble_when_predict_called_then_correct_features_passed_to_models(
        self, ensemble_data, tmp_path
    ):
        model = PerQuantileTabularEnsemble(path=str(tmp_path), prediction_length=5)
        model.quantile_levels = [0.1, 0.5, 0.9]
        model.fit(**ensemble_data, time_limit=10)

        regressor = model.ensemble_regressor
        assert isinstance(regressor, PerQuantileTabularEnsembleRegressor)

        regressor.mean_model = Mock()
        regressor.mean_model.is_fit.return_value = True
        regressor.quantile_models = [Mock() for _ in range(len(model.quantile_levels))]

        # Create test predictions with known values
        test_mean_preds = np.array([[[[[10.0, 20.0]]]]])  # shape: (1, 1, 1, 1, 2)
        test_quantile_preds = np.array(
            [
                [1.0, 2.0],  # quantile 0.1
                [5.0, 6.0],  # quantile 0.5
                [9.0, 10.0],  # quantile 0.9
            ]
        )[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape: (1, 1, 1, 3, 2)

        regressor.predict(test_mean_preds, test_quantile_preds)

        expected_call_args = np.concatenate([test_mean_preds[0, 0, 0], test_quantile_preds[0, 0, 0]], axis=-2)
        for mock_model, expected in zip(
            chain([regressor.mean_model], regressor.quantile_models),
            expected_call_args,
        ):
            call_args = mock_model.predict.call_args[0][0]  # type: ignore
            np.testing.assert_array_equal(call_args.values, expected[np.newaxis, ...])
