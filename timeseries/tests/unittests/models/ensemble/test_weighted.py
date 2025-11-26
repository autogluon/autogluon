import itertools
from unittest import mock

import numpy as np
import pytest

from autogluon.timeseries.models import SeasonalNaiveModel
from autogluon.timeseries.models.ensemble import (
    GreedyEnsemble,
    PerformanceWeightedEnsemble,
    SimpleAverageEnsemble,
)

from ...common import DUMMY_TS_DATAFRAME, PREDICTIONS_FOR_DUMMY_TS_DATAFRAME, get_data_frame_with_item_index


@pytest.fixture(
    params=itertools.product(
        [
            {"model1": -0.2, "model2": -0.3, "model3": -1500},
            {"model1": -0.5, "model2": -0.2, "model3": 0},
            {"model1": -3.0, "model2": -1.0, "model3": -2.0},
            {"model1": -3.0, "model2": -1.0, "model3": float("nan")},
        ],
        [1, 2, 3],  # number of constituents
    )
)
def ensemble_data_with_varying_scores(request):
    model_scores, number_of_models = request.param
    model_keys = ["model1", "model2", "model3"]

    return {
        "predictions_per_window": dict(
            zip(model_keys[:number_of_models], [[PREDICTIONS_FOR_DUMMY_TS_DATAFRAME]] * number_of_models)
        ),
        "data_per_window": [DUMMY_TS_DATAFRAME],
        "model_scores": {k: v for k, v in itertools.islice(model_scores.items(), number_of_models)},
    }


class TestAllTimeSeriesWeightedEnsembleModels:
    """Test that all ensemble models can be instantiated."""

    @pytest.fixture(
        params=[
            GreedyEnsemble,
            PerformanceWeightedEnsemble,
            SimpleAverageEnsemble,
        ]
    )
    def model_constructor(self, request):
        yield request.param

    def test_ensemble_models_can_be_initialized(self, model_constructor):
        try:
            model_constructor()
        except:
            pytest.fail(f"Could not initialize {model_constructor}")

    def test_ensemble_models_can_fit_and_predict(self, model_constructor, predictions_data_and_prediction_length):
        predictions_per_window, data_per_window, model_scores, prediction_length = (
            predictions_data_and_prediction_length
        )

        model = model_constructor(prediction_length=prediction_length)
        try:
            model.fit(
                predictions_per_window=predictions_per_window,
                data_per_window=data_per_window,
                model_scores=model_scores,
            )
            model.predict({k: v[0] for k, v in predictions_per_window.items()})
        except:
            pytest.fail(f"Could not fit and predict with {model_constructor}")

    def test_when_ensemble_models_predict_then_prediction_horizon_aligns_with_input(
        self, model_constructor, predictions_data_and_prediction_length
    ):
        predictions_per_window, data_per_window, model_scores, prediction_length = (
            predictions_data_and_prediction_length
        )

        model = model_constructor(prediction_length=prediction_length)
        model.fit(
            predictions_per_window=predictions_per_window, data_per_window=data_per_window, model_scores=model_scores
        )
        predictions = model.predict({k: v[0] for k, v in predictions_per_window.items()})

        first_model_prediction = next(iter(predictions_per_window.values()))[0]
        assert all(predictions.index == first_model_prediction.index)

    def test_when_ensemble_models_predict_then_prediction_contains_no_nans(
        self, model_constructor, predictions_data_and_prediction_length
    ):
        predictions_per_window, data_per_window, model_scores, prediction_length = (
            predictions_data_and_prediction_length
        )

        model = model_constructor(prediction_length=prediction_length)
        model.fit(
            predictions_per_window=predictions_per_window, data_per_window=data_per_window, model_scores=model_scores
        )
        predictions = model.predict({k: v[0] for k, v in predictions_per_window.items()})

        assert not predictions.isna().any(axis=None)

    def test_given_model_when_fit_called_then_internal_fit_method_called_correctly(
        self, model_constructor, predictions_data_and_prediction_length
    ):
        predictions_per_window, data_per_window, model_scores, prediction_length = (
            predictions_data_and_prediction_length
        )
        model = model_constructor(prediction_length=prediction_length)

        with mock.patch.object(model, "_fit") as mock_fit:
            _ = model.fit(
                predictions_per_window=predictions_per_window,
                data_per_window=data_per_window,
                model_scores=model_scores,
                time_limit=10,
            )
            mock_fit.assert_called_once()
            assert mock_fit.call_args.kwargs["time_limit"] == 10
            assert mock_fit.call_args.kwargs["predictions_per_window"] is predictions_per_window
            assert mock_fit.call_args.kwargs["data_per_window"] is data_per_window

    def test_when_some_base_models_fail_during_prediction_then_ensemble_raises_runtime_error(self, model_constructor):
        base_model = SeasonalNaiveModel(prediction_length=1, freq=DUMMY_TS_DATAFRAME.freq)
        base_model.fit(train_data=DUMMY_TS_DATAFRAME)
        base_model_preds = base_model.predict(DUMMY_TS_DATAFRAME)

        ensemble = model_constructor()
        ensemble.model_to_weight = {"ARIMA": 0.5, "SeasonalNaive": 0.5}

        with pytest.raises(RuntimeError):
            ensemble.predict(data={"ARIMA": None, "SeasonalNaive": base_model_preds})

    def test_when_predict_called_then_predictions_can_be_scored(
        self, model_constructor, predictions_data_and_prediction_length
    ):
        predictions_per_window, data_per_window, model_scores, prediction_length = (
            predictions_data_and_prediction_length
        )

        model = model_constructor(prediction_length=prediction_length)
        model.fit(
            predictions_per_window=predictions_per_window, data_per_window=data_per_window, model_scores=model_scores
        )
        predictions = model.predict({k: v[0] for k, v in predictions_per_window.items()})

        metric_value = model.eval_metric(data_per_window[0], predictions)
        assert np.isfinite(metric_value)


class TestSimpleAverageEnsemble:
    def test_when_fit_called_then_weights_are_equal_and_correct(self, ensemble_data_with_varying_scores):
        model = SimpleAverageEnsemble()
        model.fit(**ensemble_data_with_varying_scores)

        expected_weight = 1.0 / len(ensemble_data_with_varying_scores["predictions_per_window"])
        for model_name, weight in model.model_to_weight.items():
            assert weight == pytest.approx(expected_weight)


class TestPerformanceWeightedEnsemble:
    @pytest.mark.parametrize("weight_scheme", ["sqrt", "sq", "inv"])
    def test_when_fit_called_then_scores_are_correct(self, ensemble_data_with_varying_scores, weight_scheme):
        model = PerformanceWeightedEnsemble(hyperparameters={"weight_scheme": weight_scheme})
        model.fit(**ensemble_data_with_varying_scores)

        scores = ensemble_data_with_varying_scores["model_scores"]
        scores = {k: v for k, v in scores.items() if not np.isnan(v)}

        expected_weights = {}
        if weight_scheme == "sq":
            expected_weights = {name: np.square(1 / (-score + 1e-5)) for name, score in scores.items()}
        elif weight_scheme == "inv":
            expected_weights = {name: 1 / (-score + 1e-5) for name, score in scores.items()}
        elif weight_scheme == "sqrt":
            expected_weights = {name: np.sqrt(1 / (-score + 1e-5)) for name, score in scores.items()}

        total_weight = sum(expected_weights.values())
        expected_weights = {name: weight / total_weight for name, weight in expected_weights.items()}

        for model_name, weight in model.model_to_weight.items():
            assert weight == pytest.approx(expected_weights[model_name])

    @pytest.mark.parametrize("weight_scheme", ["sqrt", "sq", "inv"])
    def test_when_fit_called_then_higher_scores_are_given_to_higher_scores(
        self, ensemble_data_with_varying_scores, weight_scheme
    ):
        ensemble = PerformanceWeightedEnsemble(hyperparameters={"weight_scheme": weight_scheme})
        ensemble.fit(**ensemble_data_with_varying_scores)

        scores = ensemble_data_with_varying_scores["model_scores"]
        scores = {k: v for k, v in scores.items() if not np.isnan(v)}

        models_ranked_by_error = sorted(list(scores.keys()), key=lambda x: -scores.get(x), reverse=True)  # type: ignore
        models_ranked_by_weight = sorted(list(scores.keys()), key=ensemble.model_to_weight.get)  # type: ignore

        assert models_ranked_by_error == models_ranked_by_weight

    def test_when_fit_called_then_raises_error_without_model_scores(self, ensemble_data_with_varying_scores):
        model = PerformanceWeightedEnsemble()
        ensemble_data_without_scores = {
            k: v for k, v in ensemble_data_with_varying_scores.items() if k != "model_scores"
        }

        with pytest.raises(AssertionError):
            model.fit(**ensemble_data_without_scores)
