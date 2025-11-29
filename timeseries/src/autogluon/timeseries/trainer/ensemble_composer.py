import logging
import os
import time
import traceback
from typing import Iterator

import networkx as nx
import numpy as np
from typing_extensions import Self

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer
from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel, get_ensemble_class
from autogluon.timeseries.utils.warning_filters import warning_filter

from .utils import log_scores_and_times

logger = logging.getLogger("autogluon.timeseries.trainer")


class EnsembleComposer:
    """Helper class for TimeSeriesTrainer to build multi-layer stack ensembles."""

    def __init__(
        self,
        path,
        prediction_length: int,
        eval_metric: TimeSeriesScorer,
        target: str,
        quantile_levels: list[float],
        model_graph: nx.DiGraph,
        ensemble_hyperparameters: dict,
    ):
        self.eval_metric = eval_metric
        self.path = path
        self.prediction_length = prediction_length
        self.target = target
        self.quantile_levels = quantile_levels

        self.ensemble_hyperparameters = ensemble_hyperparameters

        self.banned_model_names = list(model_graph.nodes)
        self.model_graph = self._get_base_model_graph(source_graph=model_graph)

    @staticmethod
    def _get_base_model_graph(source_graph: nx.DiGraph) -> nx.DiGraph:
        """Return a model graph by copying only base models (nodes without predecessors)
        This ensures we start fresh for ensemble building.
        """
        rootset = EnsembleComposer._get_rootset(source_graph)

        dst_graph = nx.DiGraph()
        for node in rootset:
            dst_graph.add_node(node, **source_graph.nodes[node])

        return dst_graph

    @staticmethod
    def _get_rootset(graph: nx.DiGraph) -> list[str]:
        return [n for n in graph.nodes if not list(graph.predecessors(n))]

    def iter_ensembles(self) -> Iterator[tuple[int, AbstractTimeSeriesEnsembleModel, list[str]]]:
        """Iterate over trained ensemble models, layer by layer.

        Yields
        ------
        layer_ix
            The layer index of the ensemble.
        model
            The ensemble model object
        base_model_names
            The names of the base models that are part of the ensemble.
        """
        rootset = self._get_rootset(self.model_graph)

        for layer_ix, layer in enumerate(nx.traversal.bfs_layers(self.model_graph, rootset)):
            if layer_ix == 0:  # we don't need base models
                continue

            for model_name in layer:
                attrs = self.model_graph.nodes[model_name]
                model_path = os.path.join(self.path, *attrs["path"])
                model = attrs["type"].load(path=model_path)

                yield (
                    layer_ix,
                    model,
                    list(self.model_graph.predecessors(model_name)),
                )

    def fit(
        self,
        data_per_window: list[TimeSeriesDataFrame],
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        time_limit: float | None = None,
    ) -> Self:
        base_model_scores = {k: self.model_graph.nodes[k]["val_score"] for k in self.model_graph.nodes}
        model_names = list(base_model_scores.keys())

        if not self._can_fit_ensemble(time_limit, len(model_names)):
            return self

        logger.info(f"Fitting {len(self.ensemble_hyperparameters)} ensemble(s).")

        for ensemble_name, ensemble_hp_dict in self.ensemble_hyperparameters.items():
            try:
                time_start = time.monotonic()
                ensemble_class = get_ensemble_class(ensemble_name)
                ensemble = ensemble_class(
                    eval_metric=self.eval_metric,
                    target=self.target,
                    prediction_length=self.prediction_length,
                    path=self.path,
                    freq=data_per_window[0].freq,
                    quantile_levels=self.quantile_levels,
                    hyperparameters=ensemble_hp_dict,
                )
                # update name to prevent name collisions
                ensemble.name = self._get_ensemble_model_name(ensemble.name)

                with warning_filter():
                    ensemble.fit(
                        predictions_per_window=predictions_per_window,
                        data_per_window=data_per_window,
                        model_scores=base_model_scores,
                        time_limit=time_limit,
                    )
                ensemble.fit_time = time.monotonic() - time_start

                ensemble_predictions_per_window = []
                score_per_window = []
                for window_idx, data in enumerate(data_per_window):
                    predictions = ensemble.predict(
                        {n: predictions_per_window[n][window_idx] for n in ensemble.model_names}
                    )
                    ensemble_predictions_per_window.append(predictions)
                    score_per_window.append(self.eval_metric.score(data, predictions, self.target))
                ensemble.cache_oof_predictions(ensemble_predictions_per_window)
                ensemble.val_score = float(np.mean(score_per_window, dtype=np.float64))

                # TODO: add ensemble's own time to predict_time
                ensemble.predict_time = self._calculate_base_models_predict_time(ensemble.model_names)

                log_scores_and_times(
                    ensemble.val_score,
                    ensemble.fit_time,
                    ensemble.predict_time,
                    eval_metric_name=self.eval_metric.name_with_sign,
                )

                self._add_model(ensemble, base_models=ensemble.model_names)

                # Save the ensemble model to disk
                ensemble.save()
            except Exception as err:  # noqa
                logger.error(
                    f"\tWarning: Exception caused {ensemble_name} to fail during training... Skipping this model."
                )
                logger.error(f"\t{err}")
                logger.debug(traceback.format_exc())

        return self

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

    def _get_ensemble_model_name(self, name: str) -> str:
        """Revise name for an ensemble model, ensuring we don't have name collisions"""
        base_name = name
        increment = 1
        while name in self.banned_model_names:
            increment += 1
            name = f"{base_name}_{increment}"
        return name

    def _calculate_base_models_predict_time(self, model_names: list[str]) -> float:
        """Calculate ensemble predict time as sum of base model predict times."""
        return sum(self.model_graph.nodes[name]["predict_time"] for name in model_names)


def validate_ensemble_hyperparameters(hyperparameters) -> dict:
    """Validate ensemble hyperparameters dict."""
    if not isinstance(hyperparameters, dict):
        raise ValueError(f"ensemble_hyperparameters must be dict, got {type(hyperparameters)}")

    # Validate all ensemble names are known
    for ensemble_name, ensemble_hyperparameters in hyperparameters.items():
        get_ensemble_class(ensemble_name)  # Will raise if unknown
        assert isinstance(ensemble_hyperparameters, dict)
    return hyperparameters
