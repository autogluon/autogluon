import numpy as np
import pytest

from autogluon.timeseries.models.ensemble.array_based import LinearStackerEnsemble
from autogluon.timeseries.models.ensemble.array_based.regressor import LinearStackerEnsembleRegressor


class TestLinearStackerEnsembleRegressor:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        num_windows, num_items, prediction_length, num_models = 3, 5, 4, 3
        quantile_levels = [0.1, 0.5, 0.9]

        # Create mock base model predictions
        mean_predictions = np.random.randn(num_windows, num_items, prediction_length, 1, num_models)
        quantile_predictions = np.random.randn(
            num_windows, num_items, prediction_length, len(quantile_levels), num_models
        )

        # Create mock labels
        labels = np.random.randn(num_windows, num_items, prediction_length, 1)

        return {
            "mean_predictions": mean_predictions,
            "quantile_predictions": quantile_predictions,
            "labels": labels,
            "quantile_levels": quantile_levels,
        }

    def test_regressor_fit_predict_basic(self, sample_data):
        """Test basic fit and predict functionality."""
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data["quantile_levels"],
            weights_per="m",
            max_epochs=10,  # Small for testing
        )

        # Fit the regressor
        regressor.fit(
            base_model_mean_predictions=sample_data["mean_predictions"],
            base_model_quantile_predictions=sample_data["quantile_predictions"],
            labels=sample_data["labels"],
        )

        # Check that weights were learned
        assert regressor.weights is not None
        assert regressor.weights.shape == (1, 1, 1, 1, 3)  # Full broadcasting shape for "m" weights_per

        # Test prediction
        mean_pred, quantile_pred = regressor.predict(
            base_model_mean_predictions=sample_data["mean_predictions"][:1],  # Single window
            base_model_quantile_predictions=sample_data["quantile_predictions"][:1],
        )

        # Check output shapes
        assert mean_pred.shape == (1, 5, 4, 1)  # (windows, items, time, 1)
        assert quantile_pred.shape == (1, 5, 4, 3)  # (windows, items, time, quantiles)

    @pytest.mark.parametrize("weights_per", ["m", "mt", "mq", "mtq"])
    def test_regressor_different_weights_per(self, sample_data, weights_per):
        """Test different weights_per configurations."""
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data["quantile_levels"],
            weights_per=weights_per,
            max_epochs=5,  # Small for testing
        )

        regressor.fit(
            base_model_mean_predictions=sample_data["mean_predictions"],
            base_model_quantile_predictions=sample_data["quantile_predictions"],
            labels=sample_data["labels"],
        )

        # Check weight shapes
        expected_shapes = {
            "m": (1, 1, 1, 1, 3),  # Full broadcasting shape
            "mt": (1, 1, 4, 1, 3),  # prediction_length, num_models
            "mq": (1, 1, 1, 4, 3),  # (len(quantile_levels) + 1), num_models
            "mtq": (1, 1, 4, 4, 3),  # prediction_length, (len(quantile_levels) + 1), num_models
        }

        assert regressor.weights.shape == expected_shapes[weights_per]

        # Test prediction works
        mean_pred, quantile_pred = regressor.predict(
            base_model_mean_predictions=sample_data["mean_predictions"][:1],
            base_model_quantile_predictions=sample_data["quantile_predictions"][:1],
        )

        assert mean_pred.shape == (1, 5, 4, 1)
        assert quantile_pred.shape == (1, 5, 4, 3)

    def test_regressor_serialization(self, sample_data):
        """Test that regressor can be pickled and unpickled."""
        import pickle

        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data["quantile_levels"],
            weights_per="m",
            max_epochs=5,
        )

        regressor.fit(
            base_model_mean_predictions=sample_data["mean_predictions"],
            base_model_quantile_predictions=sample_data["quantile_predictions"],
            labels=sample_data["labels"],
        )

        # Serialize and deserialize
        serialized = pickle.dumps(regressor)
        deserialized = pickle.loads(serialized)

        # Check that weights are preserved
        np.testing.assert_array_equal(regressor.weights, deserialized.weights)

        # Check that prediction still works
        mean_pred1, quantile_pred1 = regressor.predict(
            base_model_mean_predictions=sample_data["mean_predictions"][:1],
            base_model_quantile_predictions=sample_data["quantile_predictions"][:1],
        )

        mean_pred2, quantile_pred2 = deserialized.predict(
            base_model_mean_predictions=sample_data["mean_predictions"][:1],
            base_model_quantile_predictions=sample_data["quantile_predictions"][:1],
        )

        np.testing.assert_array_equal(mean_pred1, mean_pred2)
        np.testing.assert_array_equal(quantile_pred1, quantile_pred2)

    def test_regressor_timeout(self, sample_data):
        """Test that regressor respects time limit."""
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data["quantile_levels"],
            weights_per="m",
            max_epochs=10000,  # Large number to ensure timeout hits first
        )

        import time

        start_time = time.time()

        regressor.fit(
            base_model_mean_predictions=sample_data["mean_predictions"],
            base_model_quantile_predictions=sample_data["quantile_predictions"],
            labels=sample_data["labels"],
            time_limit=0.1,  # Very short time limit
        )

        elapsed_time = time.time() - start_time

        # Should finish within reasonable time of the limit (allow some overhead)
        assert elapsed_time < 0.5, f"Training took {elapsed_time:.3f}s, expected < 0.5s"

        # Should still have learned weights
        assert regressor.weights is not None


class TestLinearStackerEnsemble:
    def test_ensemble_model_creation(self):
        """Test that the ensemble model can be created with default hyperparameters."""
        model = LinearStackerEnsemble()

        # Check default hyperparameters
        hps = model.get_hyperparameters()
        assert hps["weights_per"] == "m"
        assert hps["lr"] == 0.1
        assert hps["max_epochs"] == 10000

    def test_ensemble_model_custom_hyperparameters(self):
        """Test ensemble model with custom hyperparameters."""
        custom_hps = {
            "weights_per": "mt",
            "lr": 0.05,
            "max_epochs": 1000,
        }

        model = LinearStackerEnsemble(hyperparameters=custom_hps)
        hps = model.get_hyperparameters()

        assert hps["weights_per"] == "mt"
        assert hps["lr"] == 0.05
        assert hps["max_epochs"] == 1000
