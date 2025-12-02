import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import networkx as nx
import numpy as np
from typing_extensions import Self

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer
from autogluon.timeseries.models.ensemble import (
    AbstractTimeSeriesEnsembleModel,
    PerformanceWeightedEnsemble,
    get_ensemble_class,
)
from autogluon.timeseries.utils.timer import SplitTimer
from autogluon.timeseries.utils.warning_filters import warning_filter

from .utils import log_scores_and_times

logger = logging.getLogger("autogluon.timeseries.trainer")


class EnsembleComposer:
    """Helper class for TimeSeriesTrainer to build multi-layer stack ensembles.

    This class depends on the trainer to provide the necessary initialization parameters, training
    and validation data, as well as having fit the base (non-ensemble) models and persisted their
    out-of-fold predictions which will be used for ensemble training.

    Parameters
    ----------
    path
        Path of the calling TimeSeriesTrainer. EnsembleComposer finds the model objects and their
        out-of-fold prediction artifacts with respect to this path. EnsembleComposer only saves
        ensemble models and their out-of-fold predictions to this folder (i.e., does not pickle
        itself).
    prediction_length
        Number of time steps to forecast.
    eval_metric
        Metric used to evaluate ensemble performance.
    target
        Name of the target column in the time series data.
    num_windows_per_layer
        Number of windows used for training each ensemble layer. Length must match the number of layers
        in ensemble_hyperparameters. Example: (3, 2) means first layer uses 3 windows, second layer uses
        2 windows.

        Base models must have OOF predictions saved for all sum(num_windows_per_layer) windows, prior
        to this class being called.
    ensemble_hyperparameters
        Ensemble configuration. A list of dicts, one per layer. If an ensemble model should be fitted
        with multiple hyperparameter configurations, a list of dicts may be provided as the value.
        Each layer's dict maps ensemble names to either a single hyperparameter dict or a list of
        hyperparameter dicts.

        Examples:
        - ``[{"GreedyEnsemble": {}}, {"GreedyEnsemble": {}}]`` for 2 layers of greedy ensembles.
        - ``[{"GreedyEnsemble": [{"ensemble_size": 10}, {"ensemble_size": 20}]}]`` for a single layer of
          two greedy ensembles, with differing ensemble sizes.
    quantile_levels
        Quantile levels for probabilistic forecasting.
    model_graph
        Directed graph containing base models and their metadata (val_score, fit_time, etc.). Only
        base models (nodes without predecessors) are used for ensemble training.
    """

    def __init__(
        self,
        path: str,
        prediction_length: int,
        eval_metric: TimeSeriesScorer,
        target: str,
        num_windows_per_layer: tuple[int, ...],
        ensemble_hyperparameters: list[dict[str, dict | list[dict]]],
        quantile_levels: list[float],
        model_graph: nx.DiGraph,
    ):
        self.eval_metric = eval_metric
        self.path = path
        self.prediction_length = prediction_length
        self.target = target
        self.quantile_levels = quantile_levels

        self.num_windows_per_layer = num_windows_per_layer
        self.num_layers = len(num_windows_per_layer)

        if len(ensemble_hyperparameters) != self.num_layers:
            raise ValueError(
                "Number of ensemble_hyperparameters must match the number of layers. "
                f"Received {len(ensemble_hyperparameters)} ensemble_hyperparameters, "
                f"but {self.num_layers} layers."
            )
        self.ensemble_hyperparameters = ensemble_hyperparameters

        self.banned_model_names = list(model_graph.nodes)
        self.model_graph = self._get_base_model_graph(source_graph=model_graph)

    @staticmethod
    def _get_base_model_graph(source_graph: nx.DiGraph) -> nx.DiGraph:
        """Return a model graph by copying only base models (nodes without predecessors).

        This ensures we start fresh for training ensembles.
        """
        rootset = EnsembleComposer._get_rootset(source_graph)

        dst_graph = nx.DiGraph()
        for node in rootset:
            dst_graph.add_node(node, **source_graph.nodes[node])

        return dst_graph

    @staticmethod
    def _get_rootset(graph: nx.DiGraph) -> list[str]:
        return [n for n in graph.nodes if not list(graph.predecessors(n))]

    def _load_model(self, model_name: str) -> Any:
        """Load a model from the graph by name."""
        attrs = self.model_graph.nodes[model_name]
        model_path = os.path.join(self.path, *attrs["path"])
        return attrs["type"].load(path=model_path)

    def _iter_models(self, layer: int) -> Iterator[tuple[str, Any]]:
        """Iterate over models in a specific layer of the model graph.

        Parameters
        ----------
        layer
            Layer index (0 for base models, 1+ for ensemble layers)

        Yields
        ------
        model_name
            Name of the model
        model
            Loaded model instance
        """
        rootset = self._get_rootset(self.model_graph)
        layer_iter = nx.traversal.bfs_layers(self.model_graph, rootset)
        for layer_idx, layer_keys in enumerate(layer_iter):
            if layer_idx != layer:
                continue

            for model_name in layer_keys:
                model = self._load_model(model_name)
                yield model_name, model

    def iter_ensembles(self) -> Iterator[tuple[int, AbstractTimeSeriesEnsembleModel, list[str]]]:
        """Iterate over trained ensemble models, layer by layer. Used by the Trainer to copy the
        fitted models in EnsembleComposer's ``model_graph``.

        Yields
        ------
        layer_idx
            The layer index of the ensemble.
        model
            The ensemble model object
        base_model_names
            The names of the base models that are part of the ensemble.
        """
        for layer_idx in range(1, self.num_layers + 1):
            for model_name, model in self._iter_models(layer=layer_idx):
                yield (layer_idx, model, list(self.model_graph.predecessors(model_name)))

    def fit(
        self,
        data_per_window: list[TimeSeriesDataFrame],
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        time_limit: float | None = None,
    ) -> Self:
        base_model_names = [name for name, _ in self._iter_models(layer=0)]
        if not self._can_fit_ensemble(time_limit, len(base_model_names)):
            return self

        num_ensembles = sum(
            len(list(self.iter_layer_models_and_hps(layer))) for layer in range(1, self.num_layers + 1)
        )
        logger.info(f"Fitting {num_ensembles} ensemble(s), in {self.num_layers} layers.")

        assert len(data_per_window) == sum(self.num_windows_per_layer)

        def get_inputs_for_layer(layer_idx, model_names):
            """Retrieve predictions from previous layer models for current layer training."""
            if layer_idx == 1:
                # we need base models, so we use predictions_per_window provided by the trainer,
                # which contains base model predictions for all windows where ensembles will be
                # trained.
                num_windows = self.num_windows_per_layer[0]
                inputs = {name: predictions_per_window[name][:num_windows] for name in model_names}
            else:
                # if layer_idx > 1, we will be relying on predictions of previously trained ensembles
                window_start = -sum(self.num_windows_per_layer[layer_idx - 1 :])
                window_slice = slice(
                    window_start,
                    window_start + self.num_windows_per_layer[layer_idx - 1] if layer_idx < self.num_layers else None,
                )

                inputs = {}
                for model_name in model_names:
                    oof_predictions = self._get_model_oof_predictions(model_name)
                    inputs[model_name] = oof_predictions[window_slice]

            return inputs

        def get_ground_truth_for_layer(layer_idx):
            window_start = sum(self.num_windows_per_layer[: layer_idx - 1])
            window_end = window_start + self.num_windows_per_layer[layer_idx - 1]
            return data_per_window[window_start:window_end]

        main_loop_timer = SplitTimer(time_limit, rounds=num_ensembles).start()

        # main loop over layers of ensembles
        for layer_idx in range(1, self.num_layers + 1):
            layer_input_model_names = [name for name, _ in self._iter_models(layer=layer_idx - 1)]
            layer_input_model_scores = {
                name: self.model_graph.nodes[name]["val_score"] for name in layer_input_model_names
            }

            layer_predictions_per_window = get_inputs_for_layer(layer_idx, model_names=layer_input_model_names)
            layer_data_per_window = get_ground_truth_for_layer(layer_idx)

            for ensemble_name, ensemble_hp_dict in self.iter_layer_models_and_hps(layer_idx):
                try:
                    # train the ensemble model
                    time_start = time.monotonic()

                    ensemble = self._fit_single_ensemble(
                        model_name=ensemble_name,
                        hyperparameters=ensemble_hp_dict,
                        predictions_per_window=layer_predictions_per_window,
                        data_per_window=layer_data_per_window,
                        base_model_scores=layer_input_model_scores,
                        layer_idx=layer_idx,
                        time_limit=main_loop_timer.round_time_remaining(),
                    )
                    ensemble.fit_time = time.monotonic() - time_start

                    # for all windows of all layers starting from this layer, predict and save predictions
                    predictions = []
                    predict_time = 0
                    for pred_layer_idx in range(layer_idx, self.num_layers + 1):
                        predict_time_start = time.monotonic()

                        pred_base_predictions = get_inputs_for_layer(pred_layer_idx, ensemble.model_names)
                        for window_idx in range(self.num_windows_per_layer[pred_layer_idx - 1]):
                            prediction = ensemble.predict(
                                {n: pred_base_predictions[n][window_idx] for n in ensemble.model_names}
                            )
                            predictions.append(prediction)

                        predict_time = time.monotonic() - predict_time_start

                    # record marginal prediction time per window in the last layer's data
                    ensemble.predict_time_marginal = predict_time / self.num_windows_per_layer[-1]
                    ensemble.cache_oof_predictions(predictions)

                    # compute validation score using the last layer's validation windows
                    last_layer_oof_predictions = ensemble.get_oof_predictions()[-self.num_windows_per_layer[-1] :]
                    last_layer_ground_truth = get_ground_truth_for_layer(self.num_layers)
                    score_per_fold = [
                        self.eval_metric(data, prediction, target=self.target)
                        for prediction, data in zip(last_layer_oof_predictions, last_layer_ground_truth)
                    ]
                    ensemble.val_score = float(np.mean(score_per_fold, dtype=np.float64))

                    # add model to the graph, compute predict time, and save
                    self._add_model(ensemble, base_models=ensemble.model_names)
                    ensemble.predict_time = self._calculate_predict_time(ensemble)
                    self.model_graph.nodes[ensemble.name]["predict_time"] = ensemble.predict_time
                    ensemble.save()

                    # log performance
                    log_scores_and_times(
                        ensemble.val_score,
                        ensemble.fit_time,
                        ensemble.predict_time,
                        eval_metric_name=self.eval_metric.name_with_sign,
                    )

                    # check time and advance round
                    if main_loop_timer.timed_out():
                        logger.warning(
                            "Time limit exceeded during ensemble training, will stop training new ensembles."
                        )
                        return self

                except Exception as err:  # noqa
                    logger.error(
                        f"\tWarning: Exception caused {ensemble_name} to fail during training... Skipping this model."
                    )
                    logger.error(f"\t{err}")
                    logger.debug(traceback.format_exc())

                finally:
                    main_loop_timer.next_round()

        return self

    def iter_layer_models_and_hps(self, layer_idx: int):
        layer_hps = self.ensemble_hyperparameters[layer_idx - 1]

        for model_name, hps in layer_hps.items():
            if isinstance(hps, list):
                # If a list is provided, create one ensemble per hyperparameter dict
                for hp in hps:
                    yield model_name, hp
            else:
                yield model_name, hps

    def _fit_single_ensemble(
        self,
        model_name: str,
        hyperparameters: dict,
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        data_per_window: list[TimeSeriesDataFrame],
        base_model_scores: dict[str, float],
        layer_idx: int,
        time_limit: float | None = None,
    ) -> AbstractTimeSeriesEnsembleModel:
        ensemble_class = get_ensemble_class(model_name)

        # TODO: remove this after PerformanceWeightedEnsemble is removed. This is a temporary fix
        # to make sure PerformanceWeightedEnsemble is not fit on the validation scores of future
        # out-of-fold splits.
        if layer_idx < self.num_layers and ensemble_class is PerformanceWeightedEnsemble:
            raise RuntimeError(
                "PerformanceWeightedEnsemble is not supported for multilayer stack ensembles, except "
                "when it's used in the last layer of the ensemble."
            )

        ensemble: AbstractTimeSeriesEnsembleModel = ensemble_class(
            eval_metric=self.eval_metric,
            target=self.target,
            prediction_length=self.prediction_length,
            path=self.path,
            freq=data_per_window[0].freq,
            quantile_levels=self.quantile_levels,
            hyperparameters=hyperparameters,
        )

        # update name to prevent name collisions
        old_name = ensemble.name
        ensemble.name = self._get_ensemble_model_name(ensemble.name, layer_idx)
        if ensemble.name != old_name:
            path_obj = Path(ensemble.path)
            ensemble.path = str(path_obj.parent / ensemble.name)

        with warning_filter():
            ensemble.fit(
                predictions_per_window=predictions_per_window,
                data_per_window=data_per_window,
                model_scores=base_model_scores,
                time_limit=time_limit,
            )

        return ensemble

    def _get_model_oof_predictions(self, model_name: str) -> list[TimeSeriesDataFrame]:
        model_attrs = self.model_graph.nodes[model_name]
        model_path = os.path.join(self.path, *model_attrs["path"])
        return model_attrs["type"].load_oof_predictions(path=model_path)

    def _add_model(self, model, base_models: list[str]):
        self.model_graph.add_node(
            model.name,
            path=os.path.relpath(model.path, self.path).split(os.sep),
            type=type(model),
            fit_time=model.fit_time,
            predict_time=model.predict_time,
            val_score=model.val_score,
        )
        for base_model in base_models:
            self.model_graph.add_edge(base_model, model.name)
        self.banned_model_names.append(model.name)

    def _can_fit_ensemble(
        self,
        time_limit: float | None,
        num_models_available_for_ensemble: int,
    ) -> bool:
        if time_limit is not None and time_limit <= 0:
            logger.info(f"Not fitting ensemble due to lack of time remaining. Time left: {time_limit:.1f} seconds")
            return False

        if num_models_available_for_ensemble <= 1:
            logger.info(
                "Not fitting ensemble as "
                + (
                    "no models were successfully trained."
                    if not num_models_available_for_ensemble
                    else "only 1 model was trained."
                )
            )
            return False

        return True

    def _get_ensemble_model_name(self, name: str, layer_idx: int) -> str:
        """Revise name for an ensemble model, ensuring we don't have name collisions"""
        base_name = name
        layer_suffix = f"_L{layer_idx + 1}" if self.num_layers > 1 else ""
        name = f"{base_name}" + layer_suffix
        increment = 1
        while name in self.banned_model_names:
            increment += 1
            name = f"{base_name}_{increment}" + layer_suffix
        return name

    def _calculate_predict_time(self, model: AbstractTimeSeriesEnsembleModel) -> float:
        """Calculate ensemble predict time as sum of base model predict times."""
        assert model.predict_time_marginal is not None
        predict_time = model.predict_time_marginal
        for model_name in nx.ancestors(self.model_graph, model.name):
            ancestor = self._load_model(model_name)
            if isinstance(ancestor, AbstractTimeSeriesEnsembleModel):
                assert ancestor.predict_time_marginal is not None
                predict_time += ancestor.predict_time_marginal
            else:
                predict_time += ancestor.predict_time

        return predict_time


def validate_ensemble_hyperparameters(hyperparameters: list[dict[str, dict | list[dict]]]) -> None:
    if not isinstance(hyperparameters, list):
        raise ValueError(f"ensemble_hyperparameters must be list, got {type(hyperparameters)}")

    for layer_idx, layer_hp in enumerate(hyperparameters):
        if not isinstance(layer_hp, dict):
            raise ValueError(f"Layer {layer_idx} hyperparameters must be dict, got {type(layer_hp)}")
        for ensemble_name, ensemble_hp in layer_hp.items():
            get_ensemble_class(ensemble_name)  # Will raise if unknown
            hp_is_dict = isinstance(ensemble_hp, dict)
            hp_is_valid_list = isinstance(ensemble_hp, list) and all(isinstance(d, dict) for d in ensemble_hp)
            if not (hp_is_dict or hp_is_valid_list):
                raise ValueError(f"Hyperparameters for {ensemble_name} must be dict or list, got {type(ensemble_hp)}")
