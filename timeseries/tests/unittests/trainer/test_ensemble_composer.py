import random
from pathlib import Path
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.trainer import TimeSeriesTrainer
from autogluon.timeseries.trainer.ensemble_composer import EnsembleComposer, validate_ensemble_hyperparameters

from ..common import DUMMY_TS_DATAFRAME, get_data_frame_with_item_index, get_prediction_for_df


@pytest.fixture()
def patch_models():
    rng = random.Random(42)

    def mock_predict(self, data, **kwargs):
        return get_prediction_for_df(data, prediction_length=self.prediction_length)

    def mock_greedy_fit(self, predictions_per_window, *args, **kwargs):
        model_names = list(predictions_per_window.keys())
        weights = [rng.uniform(0, 1) for _ in range(len(model_names))]
        self.model_to_weight = {model_name: weights[i] / sum(weights) for i, model_name in enumerate(model_names)}
        return self

    with (
        mock.patch("autogluon.timeseries.models.local.naive.NaiveModel.predict", mock_predict),
        mock.patch("autogluon.timeseries.models.local.naive.SeasonalNaiveModel.predict", mock_predict),
        mock.patch("autogluon.timeseries.models.ensemble.weighted.greedy.GreedyEnsemble.fit", mock_greedy_fit),
        # nvutil cudaInit and cudaShutdown is triggered for each run of the trainer. we disable this here
        mock.patch("autogluon.common.utils.resource_utils.ResourceManager.get_gpu_count", return_value=0),
    ):
        yield


class TestSingleLayerEnsemble:
    @pytest.fixture()
    def trainer(self, tmp_path_factory, patch_models):
        path = str(tmp_path_factory.mktemp("agts_ensemble_composer_dummy_trainer"))
        trainer = TimeSeriesTrainer(path=path, prediction_length=3)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
        )
        yield trainer

    @pytest.mark.parametrize(
        "ensemble_hyperparameters,expected_count",
        [
            ([{"GreedyEnsemble": {"ensemble_size": 2}}], 1),
            ([{"GreedyEnsemble": {"ensemble_size": 2}, "PerformanceWeightedEnsemble": {}}], 2),
            ([{"PerformanceWeightedEnsemble": {"weight_scheme": "sq"}}], 1),
            ([{"SimpleAverageEnsemble": {}}], 1),
            ([{"GreedyEnsemble": [{"ensemble_size": 2}, {"ensemble_size": 3}]}], 2),
            (
                [
                    {
                        "GreedyEnsemble": {"ensemble_size": 2},
                        "SimpleAverageEnsemble": {},
                        "PerformanceWeightedEnsemble": {},
                    }
                ],
                3,
            ),
        ],
    )
    def test_when_ensemble_composer_created_then_can_train_single_layer_ensembles(
        self, trainer, ensemble_hyperparameters, expected_count
    ):
        """Test that the ensemble composer can create single-layer ensembles correctly."""
        ensemble_composer = EnsembleComposer(
            path=trainer.path,
            prediction_length=trainer.prediction_length,
            eval_metric=trainer.eval_metric,
            ensemble_hyperparameters=ensemble_hyperparameters,
            num_windows_per_layer=(1,),
            target=trainer.target,
            quantile_levels=trainer.quantile_levels,
            model_graph=trainer.model_graph,
        )
        data_per_window = trainer._get_validation_windows(DUMMY_TS_DATAFRAME, None)
        model_names = trainer.get_model_names(layer=0)
        predictions_per_window = trainer._get_base_model_predictions(model_names)
        ensemble_composer.fit(data_per_window=data_per_window, predictions_per_window=predictions_per_window)

        ensembles = list(ensemble_composer.iter_ensembles())
        assert len(ensembles) == expected_count

        for layer_idx, _, base_models in ensembles:
            assert layer_idx == 1
            assert len(base_models) >= 1

    def test_when_single_layer_then_ensemble_names_have_no_suffix(self, trainer):
        """Test that single-layer ensembles don't get a layer suffix."""
        ensemble_composer = EnsembleComposer(
            path=trainer.path,
            prediction_length=trainer.prediction_length,
            eval_metric=trainer.eval_metric,
            ensemble_hyperparameters=[{"GreedyEnsemble": {}, "SimpleAverageEnsemble": {}}],
            num_windows_per_layer=(1,),
            target=trainer.target,
            quantile_levels=trainer.quantile_levels,
            model_graph=trainer.model_graph,
        )
        data_per_window = trainer._get_validation_windows(DUMMY_TS_DATAFRAME, None)
        model_names = trainer.get_model_names(layer=0)
        predictions_per_window = trainer._get_base_model_predictions(model_names)
        ensemble_composer.fit(data_per_window=data_per_window, predictions_per_window=predictions_per_window)

        ensembles = list(ensemble_composer.iter_ensembles())

        for layer_idx, ensemble, _ in ensembles:
            assert not ensemble.name.endswith("_L2"), (
                f"Single-layer ensemble {ensemble.name} should not have layer suffix"
            )


class TestTwoLayerStacking:
    @pytest.fixture(
        params=[
            # ensemble_hyperparameters, expected_count_per_layer
            ([{"GreedyEnsemble": {"ensemble_size": 2}}, {"SimpleAverageEnsemble": {}}], [1, 1]),
            (
                [{"GreedyEnsemble": [{"ensemble_size": 2}, {"ensemble_size": 3}]}, {"SimpleAverageEnsemble": {}}],
                [2, 1],
            ),
            (
                [
                    {"GreedyEnsemble": [{"ensemble_size": 2}, {"ensemble_size": 3}]},
                    {"SimpleAverageEnsemble": {}, "PerformanceWeightedEnsemble": {}},
                ],
                [2, 2],
            ),
        ],
    )
    def fitted_composer_and_expected_count(self, tmp_path_factory, request, patch_models):
        path = str(tmp_path_factory.mktemp("agts_l2_trainer"))
        trainer = TimeSeriesTrainer(path=path, prediction_length=3, num_val_windows=5)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
        )

        ensemble_hyperparameters, expected_count_per_layer = request.param

        data_per_window = trainer._get_validation_windows(DUMMY_TS_DATAFRAME, None)
        model_names = trainer.get_model_names(layer=0)
        predictions_per_window = trainer._get_base_model_predictions(model_names)

        ensemble_composer = EnsembleComposer(
            path=trainer.path,
            prediction_length=trainer.prediction_length,
            eval_metric=trainer.eval_metric,
            ensemble_hyperparameters=ensemble_hyperparameters,
            num_windows_per_layer=(3, 2),
            target=trainer.target,
            quantile_levels=trainer.quantile_levels,
            model_graph=trainer.model_graph,
        ).fit(data_per_window=data_per_window, predictions_per_window=predictions_per_window)

        return ensemble_composer, expected_count_per_layer

    def test_when_two_layers_then_correct_number_of_ensembles_created(self, fitted_composer_and_expected_count):
        ensemble_composer, expected_count_per_layer = fitted_composer_and_expected_count
        ensembles = list(ensemble_composer.iter_ensembles())
        assert len(ensembles) == sum(expected_count_per_layer)

    def test_when_two_layers_then_layer_indices_correct(self, fitted_composer_and_expected_count):
        ensemble_composer, expected_count_per_layer = fitted_composer_and_expected_count

        ensembles = list(ensemble_composer.iter_ensembles())
        layer_indices = [layer_idx for layer_idx, _, _ in ensembles]
        assert layer_indices == [j for j, count in enumerate(expected_count_per_layer, start=1) for _ in range(count)]

    def test_when_two_layers_then_every_layer_has_correct_oof_predictions(self, fitted_composer_and_expected_count):
        ensemble_composer, _ = fitted_composer_and_expected_count

        ensembles = list(ensemble_composer.iter_ensembles())
        expected_oof_counts = {1: 5, 2: 2}

        for layer_idx, ensemble, _ in ensembles:
            oof_predictions = ensemble.load_oof_predictions(ensemble.path)
            assert len(oof_predictions) == expected_oof_counts[layer_idx]

    def test_when_two_layers_then_l3_uses_l2_as_base(self, fitted_composer_and_expected_count):
        ensemble_composer, _ = fitted_composer_and_expected_count
        ensembles = list(ensemble_composer.iter_ensembles())

        l2_models = [ens.name for layer_idx, ens, _ in ensembles if layer_idx == 1]
        l3_base_models = [base_models for layer_idx, _, base_models in ensembles if layer_idx == 2]

        for base_models in l3_base_models:
            assert set(base_models).issubset(set(l2_models))

    def test_when_two_layers_then_graph_structure_correct(self, fitted_composer_and_expected_count):
        ensemble_composer, expected_count_per_layer = fitted_composer_and_expected_count

        graph = ensemble_composer.model_graph
        rootset = [n for n in graph.nodes if not list(graph.predecessors(n))]
        layers = list(nx.traversal.bfs_layers(graph, rootset))

        assert len(layers) == 3  # Base models (layer 0), L2 (layer 1), L3 (layer 2)
        assert len(layers[0]) == 2  # 2 base models
        assert len(layers[1]) == expected_count_per_layer[0]
        assert len(layers[2]) == expected_count_per_layer[1]

    def test_when_two_layers_then_ensemble_names_have_layer_suffix(self, fitted_composer_and_expected_count):
        ensemble_composer, _ = fitted_composer_and_expected_count
        ensembles = list(ensemble_composer.iter_ensembles())

        for layer_idx, ensemble, _ in ensembles:
            expected_suffix = f"_L{layer_idx + 1}"
            assert ensemble.name.endswith(expected_suffix)


class TestThreeLayerStacking:
    @pytest.fixture()
    def fitted_composer(self, tmp_path_factory, patch_models):
        path = str(tmp_path_factory.mktemp("agts_l3_trainer"))
        trainer = TimeSeriesTrainer(path=path, prediction_length=3, num_val_windows=5)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
        )

        ensemble_hyperparameters = [
            {"GreedyEnsemble": {}, "SimpleAverageEnsemble": {}},
            {"GreedyEnsemble": {}, "SimpleAverageEnsemble": {}},
            {"GreedyEnsemble": {}},
        ]

        data_per_window = trainer._get_validation_windows(DUMMY_TS_DATAFRAME, None)
        model_names = trainer.get_model_names(layer=0)
        predictions_per_window = trainer._get_base_model_predictions(model_names)

        ensemble_composer = EnsembleComposer(
            path=trainer.path,
            prediction_length=trainer.prediction_length,
            eval_metric=trainer.eval_metric,
            ensemble_hyperparameters=ensemble_hyperparameters,  # type: ignore
            num_windows_per_layer=(2, 2, 1),
            target=trainer.target,
            quantile_levels=trainer.quantile_levels,
            model_graph=trainer.model_graph,
        ).fit(data_per_window=data_per_window, predictions_per_window=predictions_per_window)

        return ensemble_composer

    def test_when_three_layers_then_correct_number_of_ensembles_created(self, fitted_composer):
        ensembles = list(fitted_composer.iter_ensembles())
        assert len(ensembles) == 5

    def test_when_three_layers_then_layer_indices_correct(self, fitted_composer):
        ensembles = list(fitted_composer.iter_ensembles())
        layer_indices = [layer_idx for layer_idx, _, _ in ensembles]
        assert layer_indices == [1, 1, 2, 2, 3]

    def test_when_three_layers_then_oof_predictions_correct(self, fitted_composer):
        ensembles = list(fitted_composer.iter_ensembles())
        expected_oof_counts = {1: 5, 2: 3, 3: 1}

        for layer_idx, ensemble, _ in ensembles:
            oof_predictions = ensemble.load_oof_predictions(ensemble.path)
            assert len(oof_predictions) == expected_oof_counts[layer_idx]

    def test_when_three_layers_then_ensemble_names_have_layer_suffix(self, fitted_composer):
        ensembles = list(fitted_composer.iter_ensembles())

        for layer_idx, ensemble, _ in ensembles:
            expected_suffix = f"_L{layer_idx + 1}"
            assert ensemble.name.endswith(expected_suffix)


class TestMultilayerStackingValidationScoreComputation:
    @pytest.mark.parametrize(
        "ensemble_hyperparameters, num_windows_per_layer",
        [
            ([{"GreedyEnsemble": {"ensemble_size": 2}}, {"SimpleAverageEnsemble": {}}], (3, 2)),
            ([{"GreedyEnsemble": {"ensemble_size": 2}}, {"SimpleAverageEnsemble": {}}], (4, 1)),
            ([{"GreedyEnsemble": {"ensemble_size": 2}, "SimpleAverageEnsemble": {}}], (5,)),
        ],
    )
    def test_when_fit_called_then_all_ensembles_are_scored_on_last_layers_data(
        self, tmp_path_factory, patch_models, ensemble_hyperparameters, num_windows_per_layer
    ):
        path = str(tmp_path_factory.mktemp("agts_scoring_test"))
        trainer = TimeSeriesTrainer(path=path, prediction_length=3, num_val_windows=5)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
        )

        data_per_window = trainer._get_validation_windows(DUMMY_TS_DATAFRAME, None)
        model_names = trainer.get_model_names(layer=0)
        predictions_per_window = trainer._get_base_model_predictions(model_names)

        ensemble_composer = EnsembleComposer(
            path=trainer.path,
            prediction_length=trainer.prediction_length,
            eval_metric=trainer.eval_metric,
            ensemble_hyperparameters=ensemble_hyperparameters,
            num_windows_per_layer=num_windows_per_layer,
            target=trainer.target,
            quantile_levels=trainer.quantile_levels,
            model_graph=trainer.model_graph,
        )

        last_layer_ground_truth = data_per_window[-num_windows_per_layer[-1] :]

        ensemble_composer.fit(
            data_per_window=data_per_window,
            predictions_per_window=predictions_per_window,
        )

        # Verify all ensembles were trained and scored
        ensembles = list(ensemble_composer.iter_ensembles())
        assert len(ensembles) > 0

        # For each ensemble, verify its val_score was computed using last layer data
        # by checking that recomputing with last layer data gives the same score
        for layer_idx, ensemble, _ in ensembles:
            last_layer_oof = ensemble.get_oof_predictions()[-len(last_layer_ground_truth) :]

            # Recompute score using last layer ground truth
            recomputed_scores = [
                trainer.eval_metric(data, pred, target=trainer.target)
                for pred, data in zip(last_layer_oof, last_layer_ground_truth)
            ]
            recomputed_val_score = float(np.mean(recomputed_scores))

            # Should match the ensemble's val_score
            assert ensemble.val_score is not None
            assert abs(ensemble.val_score - recomputed_val_score) < 1e-6, (
                f"Ensemble {ensemble.name} at layer {layer_idx} was not scored on last layer data. "
                f"Expected val_score={recomputed_val_score}, got={ensemble.val_score}"
            )


class TestWindowSlicing:
    def get_trainer_and_composer(
        self,
        path: Path,
        train_data: TimeSeriesDataFrame,
        ensemble_hyperparameters: list[dict],
        num_windows_per_layer: tuple[int, ...],
        prediction_length: int,
        num_val_windows: int,
    ):
        trainer = TimeSeriesTrainer(
            path=str(path / "agts_window_slicing"),
            prediction_length=prediction_length,
            num_val_windows=num_val_windows,
        )
        trainer.fit(
            train_data=train_data,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
        )

        ensemble_composer = EnsembleComposer(
            path=trainer.path,
            prediction_length=trainer.prediction_length,
            eval_metric=trainer.eval_metric,
            ensemble_hyperparameters=ensemble_hyperparameters,
            num_windows_per_layer=num_windows_per_layer,
            target=trainer.target,
            quantile_levels=trainer.quantile_levels,
            model_graph=trainer.model_graph,
        )

        return trainer, ensemble_composer

    @pytest.mark.parametrize(
        "ensemble_hyperparameters, num_windows_per_layer, expected_num_windows, expected_first_window_offset",
        [
            (
                # ensemble_hyperparameters
                [{"GreedyEnsemble": {}}, {"GreedyEnsemble": {}}],
                # num_windows_per_layer provided to ensemble composer
                (3, 2),
                # expected_num_windows: number of windows we expect to be in the prediction_per_window and
                # data_per_window parameters in the call to _fit_single_ensemble
                (3, 2),
                # first_window_offset: number of prediction_length windows we expect the first window of this
                # call to _fit_single_ensemble to be offset from the start of the trainer's validation windows
                (0, 3),
            ),
            (
                [{"GreedyEnsemble": {}}, {"GreedyEnsemble": {}}, {"GreedyEnsemble": {}}],
                (2, 2, 1),
                (2, 2, 1),
                (0, 2, 4),
            ),
            (
                [{"GreedyEnsemble": [{"ensemble_size": 2}, {"ensemble_size": 3}]}, {"GreedyEnsemble": {}}],
                (4, 1),
                (4, 4, 1),
                (0, 0, 4),
            ),
        ],
    )
    @pytest.mark.parametrize("prediction_length", [1, 5])
    def test_when_ensemble_composer_called_then_window_indices_correct(
        self,
        tmp_path,
        patch_models,
        ensemble_hyperparameters,
        num_windows_per_layer,
        expected_num_windows,
        expected_first_window_offset,
        prediction_length,
    ):
        num_val_windows = 5
        train_df = get_data_frame_with_item_index(["10", "A", "2", "1"], data_length=50)
        trainer, ensemble_composer = self.get_trainer_and_composer(
            path=tmp_path,
            train_data=train_df,
            ensemble_hyperparameters=ensemble_hyperparameters,
            num_windows_per_layer=num_windows_per_layer,
            prediction_length=prediction_length,
            num_val_windows=num_val_windows,
        )
        validation_window_start = train_df.loc[train_df.item_ids[0]].index[-(prediction_length * num_val_windows)]

        data_per_window = trainer._get_validation_windows(train_df, None)
        model_names = trainer.get_model_names(layer=0)
        predictions_per_window = trainer._get_base_model_predictions(model_names)

        # mock the _fit_single_ensemble method with a method that captures the predictions per window and
        # data_per_window provided and adds them to captured_windows at every call. the method should be
        # called once per each ensemble specified in ensemble_hyperparameters
        captured_windows = []
        original_fit = ensemble_composer._fit_single_ensemble

        def capture_windows(*args, **kwargs):
            ppw = kwargs.get("predictions_per_window", {})
            labels = kwargs.get("data_per_window", {})
            captured_windows.append((ppw, labels))
            return original_fit(*args, **kwargs)

        with mock.patch.object(ensemble_composer, "_fit_single_ensemble", side_effect=capture_windows):
            ensemble_composer.fit(data_per_window=data_per_window, predictions_per_window=predictions_per_window)

            # assert the number of calls to _fit_single_ensemble is correct
            assert len(captured_windows) == len(expected_num_windows)

            for call_idx, (ppw, labels) in enumerate(captured_windows):
                assert len(labels) == expected_num_windows[call_idx]
                assert all(len(windows) == expected_num_windows[call_idx] for _, windows in ppw.items())
                for window_idx in range(len(labels)):
                    # for each window, assert that the data_per_window and predictions_per_window specify the same
                    # item and time indices
                    label = labels[window_idx].slice_by_timestep(-trainer.prediction_length, None)
                    inputs = [w[window_idx] for _, w in ppw.items()]

                    for input_ in inputs:
                        assert input_.index.tolist() == label.index.tolist()

                    # also assert, for each window, the start times of the windows are as expected
                    expected_offset = (expected_first_window_offset[call_idx] + window_idx) * trainer.prediction_length
                    expected_start = validation_window_start + pd.Timedelta(
                        expected_offset,
                        train_df.freq,  # type: ignore
                    )
                    actual_start = min(ts for _, ts in label.index)
                    assert actual_start == expected_start


def test_when_time_limit_exceeded_then_training_stops_early(tmp_path_factory, patch_models):
    """Test that ensemble training stops gracefully when time limit is exceeded."""
    path = str(tmp_path_factory.mktemp("agts_ensemble_time_limit_test"))
    trainer = TimeSeriesTrainer(path=path, prediction_length=3, num_val_windows=5)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"Naive": {}, "SeasonalNaive": {}},
    )

    ensemble_composer = EnsembleComposer(
        path=trainer.path,
        prediction_length=trainer.prediction_length,
        eval_metric=trainer.eval_metric,
        ensemble_hyperparameters=[
            {"GreedyEnsemble": {}, "SimpleAverageEnsemble": {}},
            {"GreedyEnsemble": {}},
        ],
        num_windows_per_layer=(3, 2),
        target=trainer.target,
        quantile_levels=trainer.quantile_levels,
        model_graph=trainer.model_graph,
    )

    data_per_window = trainer._get_validation_windows(DUMMY_TS_DATAFRAME, None)
    model_names = trainer.get_model_names(layer=0)
    predictions_per_window = trainer._get_base_model_predictions(model_names)

    # Set very short time limit to trigger timeout
    ensemble_composer.fit(
        data_per_window=data_per_window,
        predictions_per_window=predictions_per_window,
        time_limit=0.001,
    )

    # Should have trained fewer ensembles than requested due to timeout
    ensembles = list(ensemble_composer.iter_ensembles())
    assert len(ensembles) < 3, "Expected timeout to stop training before all ensembles completed"


class TestValidateEnsembleHyperparameters:
    def test_given_valid_hyperparameters_when_validate_called_then_does_not_raise(self):
        hyperparams = [{"GreedyEnsemble": {}, "PerformanceWeightedEnsemble": {"some_param": "value"}}]
        try:
            validate_ensemble_hyperparameters(hyperparams)
        except ValueError:
            pytest.fail("Unexpected ValueError raised")

    def test_given_invalid_ensemble_name_when_validate_called_then_error_raised(self):
        hyperparams = [{"InvalidEnsemble": {}}]
        with pytest.raises(ValueError, match="Unknown ensemble type: InvalidEnsemble"):
            validate_ensemble_hyperparameters(hyperparams)  # type: ignore

    @pytest.mark.parametrize("hyperparameters", ["invalid", {"invalid": {}}, {"GreedyEnsemble": {}}])
    def test_given_non_dict_input_when_validate_called_then_error_raised(self, hyperparameters):
        with pytest.raises(ValueError, match="ensemble_hyperparameters must be list"):
            validate_ensemble_hyperparameters(hyperparameters)


class TestEnsemblePredictTime:
    @pytest.fixture()
    def trainer(self, tmp_path_factory, patch_models):
        path = str(tmp_path_factory.mktemp("agts_predict_time_trainer"))
        trainer = TimeSeriesTrainer(path=path, prediction_length=3, num_val_windows=2)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
        )
        yield trainer

    @pytest.fixture(
        params=[
            [{"GreedyEnsemble": {"ensemble_size": 2}}],
            [{"SimpleAverageEnsemble": {}}],
            [{"GreedyEnsemble": [{"ensemble_size": 2}, {"ensemble_size": 3}]}],
            [{"GreedyEnsemble": {"ensemble_size": 2}}, {"SimpleAverageEnsemble": {}}],
        ]
    )
    def ensemble_composer(self, trainer, request):
        num_layers = len(request.param)
        num_windows_per_layer = (2,) if num_layers == 1 else (1, 1)
        ensemble_composer = EnsembleComposer(
            path=trainer.path,
            prediction_length=trainer.prediction_length,
            eval_metric=trainer.eval_metric,
            ensemble_hyperparameters=request.param,
            num_windows_per_layer=num_windows_per_layer,
            target=trainer.target,
            quantile_levels=trainer.quantile_levels,
            model_graph=trainer.model_graph,
        )
        data_per_window = trainer._get_validation_windows(DUMMY_TS_DATAFRAME, None)
        model_names = trainer.get_model_names(layer=0)
        predictions_per_window = trainer._get_base_model_predictions(model_names)
        ensemble_composer.fit(data_per_window=data_per_window, predictions_per_window=predictions_per_window)

        yield ensemble_composer

    def test_when_ensemble_trained_then_predict_time_marginal_set(self, ensemble_composer):
        ensembles = list(ensemble_composer.iter_ensembles())
        for _, ensemble, _ in ensembles:
            assert ensemble.predict_time_marginal is not None
            assert ensemble.predict_time_marginal > 0

    def test_when_ensemble_trained_then_predict_time_includes_base_models(self, ensemble_composer):
        ensembles = list(ensemble_composer.iter_ensembles())
        for _, ensemble, base_models in ensembles:
            ancestor_sum = 0
            for ancestor_name in nx.ancestors(ensemble_composer.model_graph, ensemble.name):
                ancestor_model = ensemble_composer._load_model(ancestor_name)
                # Use predict_time_marginal for ensembles, predict_time for base models
                if ancestor_model.predict_time_marginal is not None:
                    ancestor_sum += ancestor_model.predict_time_marginal
                else:
                    ancestor_sum += ancestor_model.predict_time

            assert ensemble.predict_time >= ensemble.predict_time_marginal
            assert ensemble.predict_time >= ancestor_sum
