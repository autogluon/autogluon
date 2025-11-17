import numpy as np
import pytest

from autogluon.timeseries.models.ensemble.array_based.regressor import LinearStackerEnsembleRegressor


class TestLinearStackerEnsembleRegressor:
    @pytest.fixture(scope="class")
    def sample_data(self):
        rng = np.random.default_rng(42)
        num_windows, num_items, prediction_length, num_models = 3, 5, 7, 3
        quantile_levels = [0.1, 0.5, 0.9]

        mean_predictions = rng.standard_normal((num_windows, num_items, prediction_length, 1, num_models))
        quantile_predictions = rng.standard_normal(
            (num_windows, num_items, prediction_length, len(quantile_levels), num_models)
        )
        labels = rng.standard_normal((num_windows, num_items, prediction_length, 1))

        return {
            "mean_predictions": mean_predictions,
            "quantile_predictions": quantile_predictions,
            "labels": labels,
            "quantile_levels": quantile_levels,
        }

    @pytest.mark.parametrize("weights_per", ["m", "mt", "mq", "mtq"])
    def test_given_weights_per_when_ensemble_regressor_fit_then_can_predict_correct_shape(
        self, sample_data, weights_per
    ):
        _, num_items, prediction_length, num_quantiles, _ = sample_data["quantile_predictions"].shape
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data["quantile_levels"],
            weights_per=weights_per,
            max_epochs=10,
        )
        regressor.fit(
            base_model_mean_predictions=sample_data["mean_predictions"],
            base_model_quantile_predictions=sample_data["quantile_predictions"],
            labels=sample_data["labels"],
        )
        mean_pred, quantile_pred = regressor.predict(
            base_model_mean_predictions=sample_data["mean_predictions"][:1],
            base_model_quantile_predictions=sample_data["quantile_predictions"][:1],
        )
        assert mean_pred.shape == (1, num_items, prediction_length, 1)
        assert quantile_pred.shape == (1, num_items, prediction_length, num_quantiles)

    @pytest.mark.parametrize("weights_per", ["m", "mt", "mq", "mtq"])
    def test_given_weights_per_when_regressor_fit_then_weights_have_correct_shape(self, sample_data, weights_per):
        _, _, prediction_length, num_quantiles, num_models = sample_data["quantile_predictions"].shape
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data["quantile_levels"],
            weights_per=weights_per,
            max_epochs=5,
        )

        regressor.fit(
            base_model_mean_predictions=sample_data["mean_predictions"],
            base_model_quantile_predictions=sample_data["quantile_predictions"],
            labels=sample_data["labels"],
        )

        expected_shapes = {
            "m": (1, 1, 1, 1, num_models),
            "mt": (1, 1, prediction_length, 1, num_models),
            "mq": (1, 1, 1, num_quantiles + 1, num_models),
            "mtq": (1, 1, prediction_length, num_quantiles + 1, num_models),
        }
        assert regressor.weights is not None
        assert regressor.weights.shape == expected_shapes[weights_per]

    @pytest.mark.parametrize("weights_per", ["m", "mt", "mq", "mtq"])
    def test_given_weights_per_when_regressor_fit_then_weights_sum_to_one_per_model(self, sample_data, weights_per):
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data["quantile_levels"],
            weights_per=weights_per,
            max_epochs=5,
        )

        regressor.fit(
            base_model_mean_predictions=sample_data["mean_predictions"],
            base_model_quantile_predictions=sample_data["quantile_predictions"],
            labels=sample_data["labels"],
        )

        assert regressor.weights is not None
        assert np.isclose(regressor.weights.sum(-1), 1.0).all()
