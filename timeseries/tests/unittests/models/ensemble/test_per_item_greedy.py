import itertools
from unittest import mock

import numpy as np
import pytest

from autogluon.timeseries.models import SeasonalNaiveModel
from autogluon.timeseries.models.ensemble import PerItemGreedyEnsemble

from ...common import get_data_frame_with_item_index


class TestPerItemGreedyEnsemble:
    @pytest.fixture(params=itertools.product([1, 2], [2, 3], [1, 3]))
    def predictions_data_and_prediction_length(self, request, temp_model_path):
        num_windows, num_models, prediction_length = request.param
        full_data = get_data_frame_with_item_index(["A", "B", "C"], start_date="2022-01-01", freq="D", data_length=120)
        data_per_window = [
            full_data.slice_by_timestep(end_index=-(w * prediction_length)) for w in range(num_windows, 0, -1)
        ]

        preds_per_window = {f"SNaive{s}": [] for s in range(1, num_models + 1)}
        for data in data_per_window:
            train_data, _ = data.train_test_split(prediction_length)
            for s in range(1, num_models + 1):
                preds_per_window[f"SNaive{s}"].append(
                    SeasonalNaiveModel(
                        prediction_length=prediction_length,
                        hyperparameters={"seasonal_period": s, "n_jobs": 1},
                        path=temp_model_path,
                    )
                    .fit(train_data)
                    .predict(train_data)
                )

        yield preds_per_window, data_per_window, prediction_length

    @pytest.fixture
    def fitted_model(self, predictions_data_and_prediction_length):
        preds_per_window, data_per_window, prediction_length = predictions_data_and_prediction_length
        model = PerItemGreedyEnsemble(prediction_length=prediction_length)
        model.fit(predictions_per_window=preds_per_window, data_per_window=data_per_window)
        yield model, preds_per_window, data_per_window, prediction_length

    def test_when_fit_called_then_weights_df_has_correct_structure(self, fitted_model):
        model, preds_per_window, _, _ = fitted_model
        assert model.weights_df.shape == (3, model.weights_df.shape[1])
        assert model.weights_df.shape[1] <= len(preds_per_window)
        assert all(model.weights_df.index == ["A", "B", "C"])

    def test_when_fit_called_then_average_weights_are_computed_correctly(self, fitted_model):
        model, _, _, _ = fitted_model
        assert len(model.average_weight) == len(model.weights_df.columns)
        assert np.allclose(model.average_weight.values, model.weights_df.mean(axis=0).values)
        assert (model.average_weight > 0).all()

    def test_when_fit_called_then_weights_sum_to_one_per_item(self, fitted_model):
        model, _, _, _ = fitted_model
        assert np.allclose(model.weights_df.sum(axis=1).values, 1.0)

    def test_when_predict_called_then_predictions_can_be_scored(self, fitted_model):
        model, preds_per_window, data_per_window, _ = fitted_model

        preds_for_predict = {k: v[0] for k, v in preds_per_window.items() if k in model.model_names}
        predictions = model.predict(preds_for_predict)

        metric_value = model.eval_metric(data_per_window[0], predictions)
        assert np.isfinite(metric_value)

    def test_when_predict_with_unseen_items_then_average_weight_is_used(self, fitted_model, temp_model_path):
        model, _, _, prediction_length = fitted_model

        new_data = get_data_frame_with_item_index(["D", "E"], start_date="2022-01-01", freq="D", data_length=120)
        new_preds = {
            model_name: SeasonalNaiveModel(
                prediction_length=prediction_length,
                hyperparameters={"seasonal_period": int(model_name.replace("SNaive", "")), "n_jobs": 1},
                path=temp_model_path,
            )
            .fit(new_data)
            .predict(new_data)
            for model_name in model.model_names
        }

        predictions = model.predict(new_preds)
        assert not predictions.isna().any(axis=None) and len(predictions.item_ids) == 2

    def test_when_remap_base_models_called_then_columns_are_renamed(self, fitted_model):
        model, _, _, _ = fitted_model
        original_columns = set(model.weights_df.columns)
        model.remap_base_models({col: f"{col}_refit" for col in original_columns})
        assert set(model.weights_df.columns) == {f"{col}_refit" for col in original_columns}

    def test_when_n_jobs_exceeds_num_items_then_n_jobs_is_reduced(self, predictions_data_and_prediction_length):
        preds_per_window, data_per_window, prediction_length = predictions_data_and_prediction_length
        model = PerItemGreedyEnsemble(prediction_length=prediction_length, hyperparameters={"n_jobs": 100})

        with mock.patch(
            "autogluon.timeseries.models.ensemble.per_item_greedy.Parallel", wraps=mock.MagicMock()
        ) as mock_parallel:
            mock_parallel.return_value = mock.MagicMock(return_value=[{} for _ in range(3)])
            model.fit(predictions_per_window=preds_per_window, data_per_window=data_per_window)
            assert mock_parallel.call_args.kwargs["n_jobs"] == 3

    def test_when_model_names_called_then_returns_non_zero_weight_models(self, fitted_model):
        model, preds_per_window, _, _ = fitted_model
        assert set(model.model_names) == set(model.weights_df.columns) <= set(preds_per_window.keys())
