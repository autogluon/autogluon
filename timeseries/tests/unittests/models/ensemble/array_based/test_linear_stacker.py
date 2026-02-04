from unittest import mock

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

    def test_per_when_regressor_initialized_with_weights_then_predictions_correct(self, sample_data):
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data["quantile_levels"],
            weights_per="m",
            max_epochs=5,
        )
        weights = [0.2, 0.25, 0.55]
        regressor.weights = np.array(weights).reshape((1, 1, 1, 1, -1))

        mean_predictions, quantile_predictions = regressor.predict(
            base_model_mean_predictions=sample_data["mean_predictions"][:1],
            base_model_quantile_predictions=sample_data["quantile_predictions"][:1],
        )

        expected_mean = np.average(sample_data["mean_predictions"][:1], axis=-1, weights=weights)
        expected_quantile = np.average(sample_data["quantile_predictions"][:1], axis=-1, weights=weights)

        assert np.allclose(mean_predictions, expected_mean)
        assert np.allclose(quantile_predictions, expected_quantile)


class TestLinearStackerEnsembleRegressionSparsification:
    @pytest.fixture(scope="class")
    def sample_data_with_low_weight_model(self):
        rng = np.random.default_rng(42)

        num_windows, num_items, prediction_length, num_models = 3, 5, 7, 3
        quantile_levels = [0.1, 0.5, 0.9]

        mean_predictions = rng.standard_normal((num_windows, num_items, prediction_length, 1, num_models))
        quantile_predictions = rng.standard_normal(
            (num_windows, num_items, prediction_length, len(quantile_levels), num_models)
        )
        labels = rng.standard_normal((num_windows, num_items, prediction_length, 1))
        model_names = [f"model_{i}" for i in range(num_models)]

        # model 1 will not have large weight
        mean_predictions[..., 1] *= 0.01
        quantile_predictions[..., 1] *= 0.01

        return {
            "mean_predictions": mean_predictions,
            "quantile_predictions": quantile_predictions,
            "labels": labels,
            "quantile_levels": quantile_levels,
            "model_names": model_names,
        }

    @pytest.mark.parametrize("weights_per", ["m", "mt", "mq", "mtq"])
    def test_when_prune_below_zero_then_no_sparsification(self, sample_data_with_low_weight_model, weights_per):
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data_with_low_weight_model["quantile_levels"],
            weights_per=weights_per,
            prune_below=0.0,
            max_epochs=10,
        )
        regressor.fit(
            base_model_mean_predictions=sample_data_with_low_weight_model["mean_predictions"],
            base_model_quantile_predictions=sample_data_with_low_weight_model["quantile_predictions"],
            labels=sample_data_with_low_weight_model["labels"],
        )

        assert regressor.kept_indices is None
        assert regressor.weights.shape[-1] == 3  # type: ignore

    @pytest.mark.parametrize("weights_per", ["m", "mt", "mq", "mtq"])
    def test_when_prune_below_set_then_low_weight_models_dropped(self, sample_data_with_low_weight_model, weights_per):
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data_with_low_weight_model["quantile_levels"],
            weights_per=weights_per,
            prune_below=0.3,
            max_epochs=10,
        )
        regressor.fit(
            base_model_mean_predictions=sample_data_with_low_weight_model["mean_predictions"],
            base_model_quantile_predictions=sample_data_with_low_weight_model["quantile_predictions"],
            labels=sample_data_with_low_weight_model["labels"],
        )

        assert regressor.kept_indices is not None
        assert len(regressor.kept_indices) < 3
        assert regressor.weights.shape[-1] == len(regressor.kept_indices)  # type: ignore

    @pytest.mark.parametrize("weights_per", ["m", "mt", "mq", "mtq"])
    def test_when_sparsified_then_weights_sum_to_one(self, sample_data_with_low_weight_model, weights_per):
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data_with_low_weight_model["quantile_levels"],
            weights_per=weights_per,
            prune_below=0.3,
            max_epochs=10,
        )
        regressor.fit(
            base_model_mean_predictions=sample_data_with_low_weight_model["mean_predictions"],
            base_model_quantile_predictions=sample_data_with_low_weight_model["quantile_predictions"],
            labels=sample_data_with_low_weight_model["labels"],
        )

        assert np.isclose(regressor.weights.sum(-1), 1.0).all()  # type: ignore

    @pytest.mark.parametrize("weights_per", ["m", "mt", "mq", "mtq"])
    def test_when_all_models_below_threshold_then_keeps_highest(self, sample_data_with_low_weight_model, weights_per):
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data_with_low_weight_model["quantile_levels"],
            weights_per=weights_per,
            prune_below=0.99,
            max_epochs=10,
        )
        regressor.fit(
            base_model_mean_predictions=sample_data_with_low_weight_model["mean_predictions"],
            base_model_quantile_predictions=sample_data_with_low_weight_model["quantile_predictions"],
            labels=sample_data_with_low_weight_model["labels"],
        )

        assert regressor.kept_indices is not None
        assert len(regressor.kept_indices) == 1
        assert regressor.weights.shape[-1] == 1  # type: ignore

    @pytest.mark.parametrize("weights_per", ["m", "mt", "mq"])
    def test_when_sparsified_then_correct_model_dropped(self, sample_data_with_low_weight_model, weights_per):
        regressor_full = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data_with_low_weight_model["quantile_levels"],
            weights_per=weights_per,
            prune_below=0.0,
            max_epochs=100,
        )
        regressor_full.fit(
            base_model_mean_predictions=sample_data_with_low_weight_model["mean_predictions"],
            base_model_quantile_predictions=sample_data_with_low_weight_model["quantile_predictions"],
            labels=sample_data_with_low_weight_model["labels"],
        )

        assert regressor_full.weights is not None
        importances_full = regressor_full.weights.squeeze()
        if importances_full.ndim > 1:
            importances_full = importances_full.mean(axis=tuple(range(importances_full.ndim - 1)))
        lowest_weight_model = int(importances_full.argmin())

        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data_with_low_weight_model["quantile_levels"],
            weights_per=weights_per,
            prune_below=0.15,
            max_epochs=100,
        )
        regressor.fit(
            base_model_mean_predictions=sample_data_with_low_weight_model["mean_predictions"],
            base_model_quantile_predictions=sample_data_with_low_weight_model["quantile_predictions"],
            labels=sample_data_with_low_weight_model["labels"],
        )

        assert regressor.kept_indices is not None
        assert lowest_weight_model not in regressor.kept_indices

    @pytest.mark.parametrize("weights_per", ["m", "mt", "mq"])
    def test_when_sparsified_then_predictions_correct_with_kept_models_only(
        self, sample_data_with_low_weight_model, weights_per
    ):
        regressor = LinearStackerEnsembleRegressor(
            quantile_levels=sample_data_with_low_weight_model["quantile_levels"],
            weights_per=weights_per,
            prune_below=0.15,
            max_epochs=50,
        )
        regressor.fit(
            base_model_mean_predictions=sample_data_with_low_weight_model["mean_predictions"],
            base_model_quantile_predictions=sample_data_with_low_weight_model["quantile_predictions"],
            labels=sample_data_with_low_weight_model["labels"],
        )

        kept_mean = sample_data_with_low_weight_model["mean_predictions"][:1, ..., regressor.kept_indices]
        kept_quantile = sample_data_with_low_weight_model["quantile_predictions"][:1, ..., regressor.kept_indices]

        mean_pred, quantile_pred = regressor.predict(
            base_model_mean_predictions=kept_mean,
            base_model_quantile_predictions=kept_quantile,
        )

        all_kept = np.concatenate([kept_mean, kept_quantile], axis=3)
        expected = np.sum(regressor.weights * all_kept, axis=-1)
        expected_mean = expected[:, :, :, :1]
        expected_quantile = expected[:, :, :, 1:]

        assert np.allclose(mean_pred, expected_mean)
        assert np.allclose(quantile_pred, expected_quantile)


class TestLinearStackerEnsembleModelSparsification:
    @pytest.fixture
    def sparsification_data(self, ensemble_data):
        preds = ensemble_data["predictions_per_window"]["dummy_model"][0]
        full_data = ensemble_data["data_per_window"][0]
        return {
            "predictions_per_window": {
                "model_0": [preds],
                "model_1": [preds * 0.01 - 10_000],  # model to be dropped
                "model_2": [preds * 2],
            },
            "data_per_window": [full_data],
        }

    def test_when_sparsified_then_model_names_updated(self, sparsification_data):
        from autogluon.timeseries.models.ensemble.array_based import LinearStackerEnsemble

        model = LinearStackerEnsemble(
            name="test_ensemble",
            prediction_length=5,
            hyperparameters={"prune_below": 0.15, "max_epochs": 100},
        )

        model._fit(**sparsification_data)

        assert isinstance(model.ensemble_regressor, LinearStackerEnsembleRegressor)
        assert model.ensemble_regressor.kept_indices is not None
        assert len(model.model_names) == len(model.ensemble_regressor.kept_indices)
        assert len(model.model_names) < 3
        assert "model_1" not in model.model_names

    def test_when_sparsified_then_predictions_use_correct_models(self, sparsification_data):
        from autogluon.timeseries.models.ensemble.array_based import LinearStackerEnsemble

        model = LinearStackerEnsemble(
            name="test_ensemble",
            prediction_length=5,
            hyperparameters={"prune_below": 0.15, "max_epochs": 100},
        )
        model._fit(**sparsification_data)

        predict_data = {
            k: sparsification_data["predictions_per_window"][k][0] for k in ["model_0", "model_1", "model_2"]
        }

        expected_arrays = [
            model.to_array(sparsification_data["predictions_per_window"][k][0]) for k in ["model_0", "model_2"]
        ]
        expected_data = np.stack(expected_arrays, axis=-1)
        expected_mean = expected_data[np.newaxis, :, :, :1, :]

        with mock.patch.object(model.ensemble_regressor, "predict") as mock_predict:
            num_items = len(sparsification_data["data_per_window"][0].item_ids)
            mock_predict.return_value = (np.zeros((1, num_items, 5, 1)), np.zeros((1, num_items, 5, 9)))
            model._predict(predict_data)

            assert mock_predict.called
            call_args = mock_predict.call_args
            call_arg_predictions = call_args.kwargs["base_model_mean_predictions"]

            np.testing.assert_array_equal(call_arg_predictions, expected_mean)
