import copy
import logging
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from autogluon.common.utils.utils import hash_pandas_df, seed_everything
from autogluon.core.trainer.abstract_trainer import AbstractTrainer
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer, check_get_evaluation_metric
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel, TimeSeriesModelBase
from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel, GreedyEnsemble
from autogluon.timeseries.models.multi_window import MultiWindowBacktestingModel
from autogluon.timeseries.models.presets import contains_searchspace, get_preset_models
from autogluon.timeseries.splitter import AbstractWindowSplitter, ExpandingWindowSplitter
from autogluon.timeseries.utils.features import (
    ConstantReplacementFeatureImportanceTransform,
    CovariateMetadata,
    PermutationFeatureImportanceTransform,
)
from autogluon.timeseries.utils.warning_filters import disable_tqdm, warning_filter

logger = logging.getLogger("autogluon.timeseries.trainer")


class TimeSeriesTrainer(AbstractTrainer[TimeSeriesModelBase]):
    _cached_predictions_filename = "cached_predictions.pkl"

    max_rel_importance_score: float = 1e5
    eps_abs_importance_score: float = 1e-5
    max_ensemble_time_limit: float = 600.0

    def __init__(
        self,
        path: str,
        prediction_length: int = 1,
        eval_metric: Union[str, TimeSeriesScorer, None] = None,
        save_data: bool = True,
        skip_model_selection: bool = False,
        enable_ensemble: bool = True,
        verbosity: int = 2,
        val_splitter: Optional[AbstractWindowSplitter] = None,
        refit_every_n_windows: Optional[int] = 1,
        # TODO: Set cache_predictions=False by default once all models in default presets have a reasonable inference speed
        cache_predictions: bool = True,
        ensemble_model_type: Optional[Type] = None,
        **kwargs,
    ):
        super().__init__(
            path=path,
            low_memory=True,
            save_data=save_data,
        )

        self.prediction_length = prediction_length
        self.quantile_levels = kwargs.get("quantile_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.target = kwargs.get("target", "target")
        self.covariate_metadata = kwargs.get("covariate_metadata", CovariateMetadata())
        self.is_data_saved = False
        self.skip_model_selection = skip_model_selection
        # Ensemble cannot be fit if val_scores are not computed
        self.enable_ensemble = enable_ensemble and not skip_model_selection
        if ensemble_model_type is None:
            ensemble_model_type = GreedyEnsemble
        else:
            logger.warning(
                "Using a custom `ensemble_model_type` is experimental functionality that may break in future versions."
            )
        self.ensemble_model_type: Type[AbstractTimeSeriesEnsembleModel] = ensemble_model_type

        self.verbosity = verbosity

        #: dict of normal model -> FULL model. FULL models are produced by
        #: self.refit_single_full() and self.refit_full().
        self.model_refit_map = {}

        self.eval_metric = check_get_evaluation_metric(eval_metric, prediction_length=prediction_length)
        if val_splitter is None:
            val_splitter = ExpandingWindowSplitter(prediction_length=self.prediction_length)
        assert isinstance(val_splitter, AbstractWindowSplitter), "val_splitter must be of type AbstractWindowSplitter"
        self.val_splitter = val_splitter
        self.refit_every_n_windows = refit_every_n_windows
        self.cache_predictions = cache_predictions
        self.hpo_results = {}

        if self._cached_predictions_path.exists():
            logger.debug(f"Removing existing cached predictions file {self._cached_predictions_path}")
            self._cached_predictions_path.unlink()

    @property
    def path_pkl(self) -> str:
        return os.path.join(self.path, self.trainer_file_name)

    def save_train_data(self, data: TimeSeriesDataFrame, verbose: bool = True) -> None:
        path = os.path.join(self.path_data, "train.pkl")
        save_pkl.save(path=path, object=data, verbose=verbose)

    def save_val_data(self, data: TimeSeriesDataFrame, verbose: bool = True) -> None:
        path = os.path.join(self.path_data, "val.pkl")
        save_pkl.save(path=path, object=data, verbose=verbose)

    def load_train_data(self) -> TimeSeriesDataFrame:
        path = os.path.join(self.path_data, "train.pkl")
        return load_pkl.load(path=path)

    def load_val_data(self) -> Optional[TimeSeriesDataFrame]:
        path = os.path.join(self.path_data, "val.pkl")
        if os.path.exists(path):
            return load_pkl.load(path=path)
        else:
            return None

    def load_data(self) -> tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        train_data = self.load_train_data()
        val_data = self.load_val_data()
        return train_data, val_data

    def save(self) -> None:
        models = self.models
        self.models = {}

        save_pkl.save(path=self.path_pkl, object=self)
        for model in self.models.values():
            model.save()

        self.models = models

    def _get_model_oof_predictions(self, model_name: str) -> list[TimeSeriesDataFrame]:
        model_path = os.path.join(self.path, self.get_model_attribute(model=model_name, attribute="path"))
        model_type = self.get_model_attribute(model=model_name, attribute="type")
        return model_type.load_oof_predictions(path=model_path)

    def _add_model(
        self,
        model: TimeSeriesModelBase,
        base_models: Optional[list[str]] = None,
    ):
        """Add a model to the model graph of the trainer. If the model is an ensemble, also add
        information about dependencies to the model graph (list of models specified via ``base_models``).

        Parameters
        ----------
        model
            The model to be added to the model graph.
        base_models
            If the model is an ensemble, the list of base model names that are included in the ensemble.
            Expected only when ``model`` is a ``AbstractTimeSeriesEnsembleModel``.

        Raises
        ------
        AssertionError
            If ``base_models`` are provided and ``model`` is not a ``AbstractTimeSeriesEnsembleModel``.
        """
        node_attrs = dict(
            path=os.path.relpath(model.path, self.path).split(os.sep),
            type=type(model),
            fit_time=model.fit_time,
            predict_time=model.predict_time,
            val_score=model.val_score,
        )
        self.model_graph.add_node(model.name, **node_attrs)

        if base_models:
            assert isinstance(model, AbstractTimeSeriesEnsembleModel)
            for base_model in base_models:
                self.model_graph.add_edge(base_model, model.name)

    def _get_model_levels(self) -> dict[str, int]:
        """Get a dictionary mapping each model to their level in the model graph"""

        # get nodes without a parent
        rootset = set(self.model_graph.nodes)
        for e in self.model_graph.edges():
            rootset.discard(e[1])

        # get shortest paths
        paths_from = defaultdict(dict)
        for source_node, paths_to in nx.shortest_path_length(self.model_graph):
            for dest_node in paths_to:
                paths_from[dest_node][source_node] = paths_to[dest_node]

        # determine levels
        levels = {}
        for n in paths_from:
            levels[n] = max(paths_from[n].get(src, 0) for src in rootset)

        return levels

    def get_models_attribute_dict(self, attribute: str, models: Optional[list[str]] = None) -> dict[str, Any]:
        """Get an attribute from the `model_graph` for each of the model names
        specified. If `models` is none, the attribute will be returned for all models"""
        results = {}
        if models is None:
            models = self.get_model_names()
        for model in models:
            results[model] = self.model_graph.nodes[model][attribute]
        return results

    def get_model_best(self) -> str:
        """Return the name of the best model by model performance on the validation set."""
        models = self.get_model_names()
        if not models:
            raise ValueError("Trainer has no fit models that can predict.")
        if len(models) == 1:
            return models[0]
        model_performances = self.get_models_attribute_dict(attribute="val_score")
        model_levels = self._get_model_levels()
        model_name_score_level_list = [
            (m, model_performances[m], model_levels.get(m, 0)) for m in models if model_performances[m] is not None
        ]

        if not model_name_score_level_list:
            raise ValueError("No fitted models have validation scores computed.")

        # rank models in terms of validation score. if two models have the same validation score,
        # rank them by their level in the model graph (lower level models are preferred).
        return max(
            model_name_score_level_list,
            key=lambda mns: (mns[1], -mns[2]),  # (score, -level)
        )[0]

    def get_model_names(self, level: Optional[int] = None) -> list[str]:
        """Get model names that are registered in the model graph"""
        if level is not None:
            return list(node for node, l in self._get_model_levels().items() if l == level)  # noqa: E741
        return list(self.model_graph.nodes)

    def get_info(self, include_model_info: bool = False) -> dict[str, Any]:
        num_models_trained = len(self.get_model_names())
        if self.model_best is not None:
            best_model = self.model_best
        else:
            try:
                best_model = self.get_model_best()
            except AssertionError:
                best_model = None
        if best_model is not None:
            best_model_score_val = self.get_model_attribute(model=best_model, attribute="val_score")
        else:
            best_model_score_val = None

        info = {
            "best_model": best_model,
            "best_model_score_val": best_model_score_val,
            "num_models_trained": num_models_trained,
        }

        if include_model_info:
            info["model_info"] = self.get_models_info()

        return info

    def _train_single(
        self,
        train_data: TimeSeriesDataFrame,
        model: AbstractTimeSeriesModel,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
    ) -> AbstractTimeSeriesModel:
        """Train the single model and return the model object that was fitted. This method
        does not save the resulting model."""
        model.fit(
            train_data=train_data,
            val_data=val_data,
            time_limit=time_limit,
            verbosity=self.verbosity,
            val_splitter=self.val_splitter,
            refit_every_n_windows=self.refit_every_n_windows,
        )
        return model

    def tune_model_hyperparameters(
        self,
        model: AbstractTimeSeriesModel,
        train_data: TimeSeriesDataFrame,
        time_limit: Optional[float] = None,
        val_data: Optional[TimeSeriesDataFrame] = None,
        hyperparameter_tune_kwargs: Union[str, dict] = "auto",
    ):
        default_num_trials = None
        if time_limit is None and (
            "num_samples" not in hyperparameter_tune_kwargs or isinstance(hyperparameter_tune_kwargs, str)
        ):
            default_num_trials = 10

        tuning_start_time = time.time()
        with disable_tqdm():
            hpo_models, _ = model.hyperparameter_tune(
                train_data=train_data,
                val_data=val_data,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                time_limit=time_limit,
                default_num_trials=default_num_trials,
                val_splitter=self.val_splitter,
                refit_every_n_windows=self.refit_every_n_windows,
            )
        total_tuning_time = time.time() - tuning_start_time

        self.hpo_results[model.name] = hpo_models
        model_names_trained = []
        # add each of the trained HPO configurations to the trained models
        for model_hpo_name, model_info in hpo_models.items():
            model_path = os.path.join(self.path, model_info["path"])
            # Only load model configurations that didn't fail
            if Path(model_path).exists():
                model_hpo = self.load_model(model_hpo_name, path=model_path, model_type=type(model))
                self._add_model(model_hpo)
                model_names_trained.append(model_hpo.name)

        logger.info(f"\tTrained {len(model_names_trained)} models while tuning {model.name}.")

        if len(model_names_trained) > 0:
            trained_model_results = [hpo_models[model_name] for model_name in model_names_trained]
            best_model_result = max(trained_model_results, key=lambda x: x["val_score"])

            logger.info(
                f"\t{best_model_result['val_score']:<7.4f}".ljust(15)
                + f"= Validation score ({self.eval_metric.name_with_sign})"
            )
            logger.info(f"\t{total_tuning_time:<7.2f} s".ljust(15) + "= Total tuning time")
            logger.debug(f"\tBest hyperparameter configuration: {best_model_result['hyperparameters']}")

        return model_names_trained

    def _train_and_save(
        self,
        train_data: TimeSeriesDataFrame,
        model: AbstractTimeSeriesModel,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
    ) -> list[str]:
        """Fit and save the given model on given training and validation data and save the trained model.

        Returns
        -------
        model_names_trained: the list of model names that were successfully trained
        """
        fit_start_time = time.time()
        model_names_trained = []
        try:
            if time_limit is not None:
                if time_limit <= 0:
                    logger.info(f"\tSkipping {model.name} due to lack of time remaining.")
                    return model_names_trained

            model = self._train_single(train_data, model, val_data=val_data, time_limit=time_limit)
            fit_end_time = time.time()
            model.fit_time = model.fit_time or (fit_end_time - fit_start_time)

            if time_limit is not None:
                time_limit = time_limit - (fit_end_time - fit_start_time)
            if val_data is not None and not self.skip_model_selection:
                model.score_and_cache_oof(
                    val_data, store_val_score=True, store_predict_time=True, time_limit=time_limit
                )

            self._log_scores_and_times(model.val_score, model.fit_time, model.predict_time)

            self.save_model(model=model)
        except TimeLimitExceeded:
            logger.error(f"\tTime limit exceeded... Skipping {model.name}.")
        except (Exception, MemoryError):
            logger.error(f"\tWarning: Exception caused {model.name} to fail during training... Skipping this model.")
            logger.error(traceback.format_exc())
        else:
            self._add_model(model=model)  # noqa: F821
            model_names_trained.append(model.name)  # noqa: F821
        finally:
            del model

        return model_names_trained

    def _log_scores_and_times(
        self,
        val_score: Optional[float] = None,
        fit_time: Optional[float] = None,
        predict_time: Optional[float] = None,
    ):
        if val_score is not None:
            logger.info(f"\t{val_score:<7.4f}".ljust(15) + f"= Validation score ({self.eval_metric.name_with_sign})")
        if fit_time is not None:
            logger.info(f"\t{fit_time:<7.2f} s".ljust(15) + "= Training runtime")
        if predict_time is not None:
            logger.info(f"\t{predict_time:<7.2f} s".ljust(15) + "= Validation (prediction) runtime")

    def _train_multi(
        self,
        train_data: TimeSeriesDataFrame,
        hyperparameters: Union[str, dict],
        val_data: Optional[TimeSeriesDataFrame] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, dict]] = None,
        excluded_model_types: Optional[list[str]] = None,
        time_limit: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> list[str]:
        logger.info(f"\nStarting training. Start time is {time.strftime('%Y-%m-%d %H:%M:%S')}")

        time_start = time.time()
        hyperparameters = copy.deepcopy(hyperparameters)

        if self.save_data and not self.is_data_saved:
            self.save_train_data(train_data)
            if val_data is not None:
                self.save_val_data(val_data)
            self.is_data_saved = True

        models = self.construct_model_templates(
            hyperparameters=hyperparameters,
            hyperparameter_tune=hyperparameter_tune_kwargs is not None,  # TODO: remove hyperparameter_tune
            freq=train_data.freq,
            multi_window=self.val_splitter.num_val_windows > 0,
            excluded_model_types=excluded_model_types,
        )

        logger.info(f"Models that will be trained: {list(m.name for m in models)}")

        if self.skip_model_selection:
            if len(models) > 1:
                raise ValueError(
                    "When `skip_model_selection=True`, only a single model must be provided via `hyperparameters` "
                    f"but {len(models)} models were given"
                )
            if contains_searchspace(models[0].get_hyperparameters()):
                raise ValueError(
                    "When `skip_model_selection=True`, model configuration should contain no search spaces."
                )

        num_base_models = len(models)
        model_names_trained = []
        for i, model in enumerate(models):
            assert isinstance(model, AbstractTimeSeriesModel)

            if time_limit is None:
                time_left = None
                time_left_for_model = None
            else:
                time_left = time_limit - (time.time() - time_start)
                if num_base_models > 1 and self.enable_ensemble:
                    time_reserved_for_ensemble = min(
                        self.max_ensemble_time_limit, time_left / (num_base_models - i + 1)
                    )
                    logger.debug(f"Reserving {time_reserved_for_ensemble:.1f}s for ensemble")
                else:
                    time_reserved_for_ensemble = 0.0
                time_left_for_model = (time_left - time_reserved_for_ensemble) / (num_base_models - i)
                if time_left <= 0:
                    logger.info(f"Stopping training due to lack of time remaining. Time left: {time_left:.1f} seconds")
                    break

            if random_seed is not None:
                seed_everything(random_seed + i)

            if contains_searchspace(model.get_hyperparameters()):
                fit_log_message = f"Hyperparameter tuning model {model.name}. "
                if time_left is not None:
                    fit_log_message += (
                        f"Tuning model for up to {time_left_for_model:.1f}s of the {time_left:.1f}s remaining."
                    )
                logger.info(fit_log_message)
                with tqdm.external_write_mode():
                    assert hyperparameter_tune_kwargs is not None, (
                        "`hyperparameter_tune_kwargs` must be provided if hyperparameters contain a search space"
                    )
                    model_names_trained += self.tune_model_hyperparameters(
                        model,
                        time_limit=time_left_for_model,
                        train_data=train_data,
                        val_data=val_data,
                        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                    )
            else:
                fit_log_message = f"Training timeseries model {model.name}. "
                if time_left is not None:
                    fit_log_message += (
                        f"Training for up to {time_left_for_model:.1f}s of the {time_left:.1f}s of remaining time."
                    )
                logger.info(fit_log_message)
                model_names_trained += self._train_and_save(
                    train_data, model=model, val_data=val_data, time_limit=time_left_for_model
                )

        if self.enable_ensemble:
            models_available_for_ensemble = self.get_model_names(level=0)

            time_left_for_ensemble = None
            if time_limit is not None:
                time_left_for_ensemble = time_limit - (time.time() - time_start)

            if time_left_for_ensemble is not None and time_left_for_ensemble <= 0:
                logger.info(
                    "Not fitting ensemble due to lack of time remaining. "
                    f"Time left: {time_left_for_ensemble:.1f} seconds"
                )
            elif len(models_available_for_ensemble) <= 1:
                logger.info(
                    "Not fitting ensemble as "
                    + (
                        "no models were successfully trained."
                        if not models_available_for_ensemble
                        else "only 1 model was trained."
                    )
                )
            else:
                try:
                    model_names_trained.append(
                        self.fit_ensemble(
                            data_per_window=self._get_ensemble_oof_data(train_data=train_data, val_data=val_data),
                            model_names=models_available_for_ensemble,
                            time_limit=time_left_for_ensemble,
                        )
                    )
                except Exception as err:  # noqa
                    logger.error(
                        "\tWarning: Exception caused ensemble to fail during training... Skipping this model."
                    )
                    logger.error(f"\t{err}")
                    logger.debug(traceback.format_exc())

        logger.info(f"Training complete. Models trained: {model_names_trained}")
        logger.info(f"Total runtime: {time.time() - time_start:.2f} s")
        try:
            best_model = self.get_model_best()
            logger.info(f"Best model: {best_model}")
            if not self.skip_model_selection:
                logger.info(f"Best model score: {self.get_model_attribute(best_model, 'val_score'):.4f}")
        except ValueError as e:
            logger.error(str(e))

        return model_names_trained

    def _get_ensemble_oof_data(
        self, train_data: TimeSeriesDataFrame, val_data: Optional[TimeSeriesDataFrame]
    ) -> list[TimeSeriesDataFrame]:
        if val_data is None:
            return [val_fold for _, val_fold in self.val_splitter.split(train_data)]
        else:
            return [val_data]

    def _get_ensemble_model_name(self) -> str:
        """Ensure we don't have name collisions in the ensemble model name"""
        ensemble_name = "WeightedEnsemble"
        increment = 1
        while ensemble_name in self._get_banned_model_names():
            increment += 1
            ensemble_name = f"WeightedEnsemble_{increment}"
        return ensemble_name

    def fit_ensemble(
        self,
        data_per_window: list[TimeSeriesDataFrame],
        model_names: list[str],
        time_limit: Optional[float] = None,
    ) -> str:
        logger.info("Fitting simple weighted ensemble.")

        predictions_per_window: dict[str, list[TimeSeriesDataFrame]] = {}
        base_model_scores = self.get_models_attribute_dict(attribute="val_score", models=self.get_model_names(0))

        for model_name in model_names:
            predictions_per_window[model_name] = self._get_model_oof_predictions(model_name=model_name)

        time_start = time.time()
        ensemble = self.ensemble_model_type(
            name=self._get_ensemble_model_name(),
            eval_metric=self.eval_metric,
            target=self.target,
            prediction_length=self.prediction_length,
            path=self.path,
            freq=data_per_window[0].freq,
            quantile_levels=self.quantile_levels,
            covariate_metadata=self.covariate_metadata,
        )
        with warning_filter():
            ensemble.fit(
                predictions_per_window=predictions_per_window,
                data_per_window=data_per_window,
                model_scores=base_model_scores,
                time_limit=time_limit,
            )
        ensemble.fit_time = time.time() - time_start

        predict_time = 0
        for m in ensemble.model_names:
            predict_time += self.get_model_attribute(model=m, attribute="predict_time")
        ensemble.predict_time = predict_time

        score_per_fold = []
        for window_idx, data in enumerate(data_per_window):
            predictions = ensemble.predict({n: predictions_per_window[n][window_idx] for n in ensemble.model_names})
            score_per_fold.append(self._score_with_predictions(data, predictions))
        ensemble.val_score = float(np.mean(score_per_fold, dtype=np.float64))

        self._log_scores_and_times(
            val_score=ensemble.val_score,
            fit_time=ensemble.fit_time,
            predict_time=ensemble.predict_time,
        )
        self._add_model(model=ensemble, base_models=ensemble.model_names)
        self.save_model(model=ensemble)
        return ensemble.name

    def leaderboard(
        self,
        data: Optional[TimeSeriesDataFrame] = None,
        extra_info: bool = False,
        extra_metrics: Optional[list[Union[str, TimeSeriesScorer]]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        logger.debug("Generating leaderboard for all models trained")

        model_names = self.get_model_names()
        if len(model_names) == 0:
            logger.warning("Warning: No models were trained during fit. Resulting leaderboard will be empty.")

        model_info = {}
        for ix, model_name in enumerate(model_names):
            model_info[model_name] = {
                "model": model_name,
                "fit_order": ix + 1,
                "score_val": self.get_model_attribute(model_name, "val_score"),
                "fit_time_marginal": self.get_model_attribute(model_name, "fit_time"),
                "pred_time_val": self.get_model_attribute(model_name, "predict_time"),
            }
            if extra_info:
                model = self.load_model(model_name=model_name)
                if isinstance(model, MultiWindowBacktestingModel):
                    model = model.most_recent_model
                    assert model is not None
                model_info[model_name]["hyperparameters"] = model.get_hyperparameters()

        if extra_metrics is None:
            extra_metrics = []

        if data is not None:
            past_data, known_covariates = data.get_model_inputs_for_scoring(
                prediction_length=self.prediction_length,
                known_covariates_names=self.covariate_metadata.known_covariates,
            )
            logger.info(
                "Additional data provided, testing on additional data. Resulting leaderboard "
                "will be sorted according to test score (`score_test`)."
            )
            model_predictions, pred_time_dict = self.get_model_pred_dict(
                model_names=model_names,
                data=past_data,
                known_covariates=known_covariates,
                raise_exception_if_failed=False,
                use_cache=use_cache,
            )

            for model_name in model_names:
                model_preds = model_predictions[model_name]
                if model_preds is None:
                    # Model failed at prediction time
                    model_info[model_name]["score_test"] = float("nan")
                    model_info[model_name]["pred_time_test"] = float("nan")
                else:
                    model_info[model_name]["score_test"] = self._score_with_predictions(data, model_preds)
                    model_info[model_name]["pred_time_test"] = pred_time_dict[model_name]

                for metric in extra_metrics:
                    if model_preds is None:
                        model_info[model_name][str(metric)] = float("nan")
                    else:
                        model_info[model_name][str(metric)] = self._score_with_predictions(
                            data, model_preds, metric=metric
                        )

        explicit_column_order = [
            "model",
            "score_test",
            "score_val",
            "pred_time_test",
            "pred_time_val",
            "fit_time_marginal",
            "fit_order",
        ]
        if extra_info:
            explicit_column_order += ["hyperparameters"]

        if data is None:
            explicit_column_order.remove("score_test")
            explicit_column_order.remove("pred_time_test")
            sort_column = "score_val"
        else:
            sort_column = "score_test"
            explicit_column_order += [str(metric) for metric in extra_metrics]

        df = pd.DataFrame(model_info.values(), columns=explicit_column_order)
        df.sort_values(by=[sort_column, "model"], ascending=[False, False], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df[explicit_column_order]

    def persist(
        self, model_names: Union[Literal["all", "best"], list[str]] = "all", with_ancestors: bool = False
    ) -> list[str]:
        if model_names == "all":
            model_names = self.get_model_names()
        elif model_names == "best":
            model_names = [self.get_model_best()]
        if not isinstance(model_names, list):
            raise ValueError(f"model_names must be a list of model names. Invalid value: {model_names}")

        if with_ancestors:
            models_with_ancestors = set()
            for model_name in model_names:
                models_with_ancestors = models_with_ancestors.union(self.get_minimum_model_set(model_name))
            model_names = list(models_with_ancestors)

        model_names_already_persisted = [model_name for model_name in model_names if model_name in self.models]
        model_names = [model_name for model_name in model_names if model_name not in model_names_already_persisted]

        for model_name in model_names:
            model = self.load_model(model_name)
            model.persist()
            self.models[model.name] = model

        return model_names

    def unpersist(self, model_names: Union[Literal["all"], list[str]] = "all") -> list[str]:
        if model_names == "all":
            model_names = list(self.models.keys())
        if not isinstance(model_names, list):
            raise ValueError(f"model_names must be a list of model names. Invalid value: {model_names}")
        unpersisted_models = []
        for model in model_names:
            if model in self.models:
                self.models.pop(model)
                unpersisted_models.append(model)
        return unpersisted_models

    def _get_model_for_prediction(
        self, model: Optional[Union[str, TimeSeriesModelBase]] = None, verbose: bool = True
    ) -> str:
        """Given an optional identifier or model object, return the name of the model with which to predict.

        If the model is not provided, this method will default to the best model according to the validation score.
        """
        if model is None:
            if self.model_best is None:
                best_model_name: str = self.get_model_best()
                self.model_best = best_model_name
            if verbose:
                logger.info(
                    f"Model not specified in predict, will default to the model with the "
                    f"best validation score: {self.model_best}",
                )
            return self.model_best
        else:
            if isinstance(model, TimeSeriesModelBase):
                return model.name
            else:
                if model not in self.get_model_names():
                    raise KeyError(f"Model '{model}' not found. Available models: {self.get_model_names()}")
                return model

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        model: Optional[Union[str, TimeSeriesModelBase]] = None,
        use_cache: bool = True,
        random_seed: Optional[int] = None,
    ) -> TimeSeriesDataFrame:
        model_name = self._get_model_for_prediction(model)
        model_pred_dict, _ = self.get_model_pred_dict(
            model_names=[model_name],
            data=data,
            known_covariates=known_covariates,
            use_cache=use_cache,
            random_seed=random_seed,
        )
        predictions = model_pred_dict[model_name]
        if predictions is None:
            raise ValueError(f"Model {model_name} failed to predict. Please check the model's logs.")
        return predictions

    def _get_eval_metric(self, metric: Union[str, TimeSeriesScorer, None]) -> TimeSeriesScorer:
        if metric is None:
            return self.eval_metric
        else:
            return check_get_evaluation_metric(
                metric,
                prediction_length=self.prediction_length,
                seasonal_period=self.eval_metric.seasonal_period,
                horizon_weight=self.eval_metric.horizon_weight,
            )

    def _score_with_predictions(
        self,
        data: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        metric: Union[str, TimeSeriesScorer, None] = None,
    ) -> float:
        """Compute the score measuring how well the predictions align with the data."""
        return self._get_eval_metric(metric).score(
            data=data,
            predictions=predictions,
            target=self.target,
        )

    def score(
        self,
        data: TimeSeriesDataFrame,
        model: Optional[Union[str, TimeSeriesModelBase]] = None,
        metric: Union[str, TimeSeriesScorer, None] = None,
        use_cache: bool = True,
    ) -> float:
        eval_metric = self._get_eval_metric(metric)
        scores_dict = self.evaluate(data=data, model=model, metrics=[eval_metric], use_cache=use_cache)
        return scores_dict[eval_metric.name]

    def evaluate(
        self,
        data: TimeSeriesDataFrame,
        model: Optional[Union[str, TimeSeriesModelBase]] = None,
        metrics: Optional[Union[str, TimeSeriesScorer, list[Union[str, TimeSeriesScorer]]]] = None,
        use_cache: bool = True,
    ) -> dict[str, float]:
        past_data, known_covariates = data.get_model_inputs_for_scoring(
            prediction_length=self.prediction_length, known_covariates_names=self.covariate_metadata.known_covariates
        )
        predictions = self.predict(data=past_data, known_covariates=known_covariates, model=model, use_cache=use_cache)

        metrics_ = [metrics] if not isinstance(metrics, list) else metrics
        scores_dict = {}
        for metric in metrics_:
            eval_metric = self._get_eval_metric(metric)
            scores_dict[eval_metric.name] = self._score_with_predictions(
                data=data, predictions=predictions, metric=eval_metric
            )
        return scores_dict

    def get_feature_importance(
        self,
        data: TimeSeriesDataFrame,
        features: list[str],
        model: Optional[Union[str, TimeSeriesModelBase]] = None,
        metric: Optional[Union[str, TimeSeriesScorer]] = None,
        time_limit: Optional[float] = None,
        method: Literal["naive", "permutation"] = "permutation",
        subsample_size: int = 50,
        num_iterations: Optional[int] = None,
        random_seed: Optional[int] = None,
        relative_scores: bool = False,
        include_confidence_band: bool = True,
        confidence_level: float = 0.99,
    ) -> pd.DataFrame:
        assert method in ["naive", "permutation"], f"Invalid feature importance method {method}."
        eval_metric = self._get_eval_metric(metric)

        logger.info("Computing feature importance")

        # seed everything if random_seed is provided
        if random_seed is not None:
            seed_everything(random_seed)

        # start timer and cap subsample size if it's greater than the number of items in the provided data set
        time_start = time.time()
        if subsample_size > data.num_items:
            logger.info(
                f"Subsample_size {subsample_size} is larger than the number of items in the data and will be ignored"
            )
            subsample_size = data.num_items

        # set default number of iterations and cap iterations if the number of items in the data is smaller
        # than the subsample size for the naive method
        num_iterations = num_iterations or (5 if method == "permutation" else 1)
        if method == "naive" and data.num_items <= subsample_size:
            num_iterations = 1

        # initialize the importance transform
        importance_transform_type = {
            "permutation": PermutationFeatureImportanceTransform,
            "naive": ConstantReplacementFeatureImportanceTransform,
        }.get(method)
        assert importance_transform_type is not None, (
            f"Invalid feature importance method {method}. Valid methods are 'permutation' and 'naive',"
        )

        importance_transform = importance_transform_type(
            covariate_metadata=self.covariate_metadata,
            prediction_length=self.prediction_length,
            random_seed=random_seed,
        )

        # if model is not provided, use the best model according to the validation score
        model = self._get_model_for_prediction(model, verbose=False)

        # persist trainer to speed up repeated inference
        persisted_models = self.persist(model_names=[model], with_ancestors=True)

        importance_samples = defaultdict(list)
        for n in range(num_iterations):
            if subsample_size < data.num_items:
                item_ids_sampled = data.item_ids.to_series().sample(subsample_size)  # noqa
                data_sample: TimeSeriesDataFrame = data.query("item_id in @item_ids_sampled")
            else:
                data_sample = data

            base_score = self.evaluate(data=data_sample, model=model, metrics=eval_metric, use_cache=False)[
                eval_metric.name
            ]

            for feature in features:
                # override importance for unused features
                if not self._model_uses_feature(model, feature):
                    continue
                else:
                    data_sample_replaced = importance_transform.transform(data_sample, feature_name=feature)
                    score = self.evaluate(
                        data=data_sample_replaced, model=model, metrics=eval_metric, use_cache=False
                    )[eval_metric.name]

                    importance = base_score - score
                    if relative_scores:
                        importance /= np.abs(base_score - self.eps_abs_importance_score)
                        importance = min(self.max_rel_importance_score, importance)

                    importance_samples[feature].append(importance)

            if time_limit is not None and time.time() - time_start > time_limit:
                logger.info(f"Time limit reached, stopping feature importance computation after {n} iterations")
                break

        self.unpersist(model_names=persisted_models)

        importance_df = (
            (
                pd.DataFrame(importance_samples)
                .agg(["mean", "std", "count"])
                .T.rename(columns={"mean": "importance", "std": "stdev", "count": "n"})
            )
            if len(importance_samples) > 0
            else pd.DataFrame(columns=["importance", "stdev", "n"])
        )

        if include_confidence_band:
            importance_df = self._add_ci_to_feature_importance(importance_df, confidence_level=confidence_level)

        return importance_df

    def _model_uses_feature(self, model: Union[str, TimeSeriesModelBase], feature: str) -> bool:
        """Check if the given model uses the given feature."""
        models_with_ancestors = set(self.get_minimum_model_set(model))

        if feature in self.covariate_metadata.static_features:
            return any(self.load_model(m).supports_static_features for m in models_with_ancestors)
        elif feature in self.covariate_metadata.known_covariates:
            return any(self.load_model(m).supports_known_covariates for m in models_with_ancestors)
        elif feature in self.covariate_metadata.past_covariates:
            return any(self.load_model(m).supports_past_covariates for m in models_with_ancestors)

        return False

    def _add_ci_to_feature_importance(
        self, importance_df: pd.DataFrame, confidence_level: float = 0.99
    ) -> pd.DataFrame:
        """Add confidence intervals to the feature importance."""
        import scipy.stats

        if confidence_level <= 0.5 or confidence_level >= 1.0:
            raise ValueError("confidence_level must lie between 0.5 and 1.0")
        ci_str = "{:.0f}".format(confidence_level * 100)

        alpha = 1 - confidence_level
        importance_df[f"p{ci_str}_low"] = np.nan
        importance_df[f"p{ci_str}_high"] = np.nan

        for i in importance_df.index:
            r = importance_df.loc[i]
            importance, stdev, n = r["importance"], r["stdev"], r["n"]
            if np.isnan(importance) or np.isnan(stdev) or np.isnan(n) or n <= 1:
                continue

            t_crit = scipy.stats.t.ppf(1 - alpha / 2, df=n - 1)

            importance_df.loc[i, f"p{ci_str}_low"] = importance - t_crit * stdev / np.sqrt(n)
            importance_df.loc[i, f"p{ci_str}_high"] = importance + t_crit * stdev / np.sqrt(n)

        return importance_df

    def _predict_model(
        self,
        model: Union[str, TimeSeriesModelBase],
        data: TimeSeriesDataFrame,
        model_pred_dict: dict[str, Optional[TimeSeriesDataFrame]],
        known_covariates: Optional[TimeSeriesDataFrame] = None,
    ) -> TimeSeriesDataFrame:
        """Generate predictions using the given model.

        This method assumes that model_pred_dict contains the predictions of all base models, if model is an ensemble.
        """
        if isinstance(model, str):
            model = self.load_model(model)
        model_inputs = self._get_inputs_to_model(model=model, data=data, model_pred_dict=model_pred_dict)
        return model.predict(model_inputs, known_covariates=known_covariates)

    def _get_inputs_to_model(
        self,
        model: Union[str, TimeSeriesModelBase],
        data: TimeSeriesDataFrame,
        model_pred_dict: dict[str, Optional[TimeSeriesDataFrame]],
    ) -> Union[TimeSeriesDataFrame, dict[str, Optional[TimeSeriesDataFrame]]]:
        """Get the first argument that should be passed to model.predict.

        This method assumes that model_pred_dict contains the predictions of all base models, if model is an ensemble.
        """
        model_set = self.get_minimum_model_set(model, include_self=False)
        if model_set:
            for m in model_set:
                if m not in model_pred_dict:
                    raise AssertionError(f"Prediction for base model {m} not found in model_pred_dict")
            return {m: model_pred_dict[m] for m in model_set}
        else:
            return data

    def get_model_pred_dict(
        self,
        model_names: list[str],
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        raise_exception_if_failed: bool = True,
        use_cache: bool = True,
        random_seed: Optional[int] = None,
    ) -> tuple[dict[str, Optional[TimeSeriesDataFrame]], dict[str, float]]:
        """Return a dictionary with predictions of all models for the given dataset.

        Parameters
        ----------
        model_names
            Names of the model for which the predictions should be produced.
        data
            Time series data to forecast with.
        known_covariates
            Future values of the known covariates.
        record_pred_time
            If True, will additionally return the total prediction times for all models (including the prediction time
            for base models). If False, will only return the model predictions.
        raise_exception_if_failed
            If True, the method will raise an exception if any model crashes during prediction.
            If False, error will be logged and predictions for failed models will contain None.
        use_cache
            If False, will ignore the cache even if it's available.
        """
        if self.cache_predictions and use_cache:
            dataset_hash = self._compute_dataset_hash(data=data, known_covariates=known_covariates)
            model_pred_dict, pred_time_dict_marginal = self._get_cached_pred_dicts(dataset_hash)
        else:
            model_pred_dict = {}
            pred_time_dict_marginal: dict[str, Any] = {}

        model_set = set()
        for model_name in model_names:
            model_set.update(self.get_minimum_model_set(model_name))
        if len(model_set) > 1:
            model_to_level = self._get_model_levels()
            model_set = sorted(model_set, key=model_to_level.get)  # type: ignore
        logger.debug(f"Prediction order: {model_set}")

        failed_models = []
        for model_name in model_set:
            if model_name not in model_pred_dict:
                if random_seed is not None:
                    seed_everything(random_seed)
                try:
                    predict_start_time = time.time()
                    model_pred_dict[model_name] = self._predict_model(
                        model=model_name,
                        data=data,
                        known_covariates=known_covariates,
                        model_pred_dict=model_pred_dict,
                    )
                    pred_time_dict_marginal[model_name] = time.time() - predict_start_time
                except Exception:
                    failed_models.append(model_name)
                    logger.error(f"Model {model_name} failed to predict with the following exception:")
                    logger.error(traceback.format_exc())
                    model_pred_dict[model_name] = None
                    pred_time_dict_marginal[model_name] = None

        if len(failed_models) > 0 and raise_exception_if_failed:
            raise RuntimeError(f"Following models failed to predict: {failed_models}")
        if self.cache_predictions and use_cache:
            self._save_cached_pred_dicts(
                dataset_hash,  # type: ignore
                model_pred_dict=model_pred_dict,
                pred_time_dict=pred_time_dict_marginal,
            )
        pred_time_dict_total = self._get_total_pred_time_from_marginal(pred_time_dict_marginal)

        final_model_pred_dict = {model_name: model_pred_dict[model_name] for model_name in model_names}
        final_pred_time_dict_total = {model_name: pred_time_dict_total[model_name] for model_name in model_names}

        return final_model_pred_dict, final_pred_time_dict_total

    def _get_total_pred_time_from_marginal(self, pred_time_dict_marginal: dict[str, float]) -> dict[str, float]:
        pred_time_dict_total = defaultdict(float)
        for model_name in pred_time_dict_marginal.keys():
            for base_model in self.get_minimum_model_set(model_name):
                if pred_time_dict_marginal[base_model] is not None:
                    pred_time_dict_total[model_name] += pred_time_dict_marginal[base_model]
        return dict(pred_time_dict_total)

    @property
    def _cached_predictions_path(self) -> Path:
        return Path(self.path) / self._cached_predictions_filename

    @staticmethod
    def _compute_dataset_hash(
        data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame] = None
    ) -> str:
        """Compute a unique string that identifies the time series dataset."""
        combined_hash = hash_pandas_df(data) + hash_pandas_df(known_covariates) + hash_pandas_df(data.static_features)
        return combined_hash

    def _load_cached_predictions(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Load cached predictions from disk. If loading fails, an empty dictionary is returned."""
        if self._cached_predictions_path.exists():
            try:
                cached_predictions = load_pkl.load(str(self._cached_predictions_path))
            except Exception:
                cached_predictions = {}
        else:
            cached_predictions = {}
        return cached_predictions

    def _get_cached_pred_dicts(
        self, dataset_hash: str
    ) -> tuple[dict[str, Optional[TimeSeriesDataFrame]], dict[str, float]]:
        """Load cached predictions for given dataset_hash from disk, if possible.

        If loading fails for any reason, empty dicts are returned.
        """
        cached_predictions = self._load_cached_predictions()
        if dataset_hash in cached_predictions:
            try:
                model_pred_dict = cached_predictions[dataset_hash]["model_pred_dict"]
                pred_time_dict = cached_predictions[dataset_hash]["pred_time_dict"]
                assert model_pred_dict.keys() == pred_time_dict.keys()
                return model_pred_dict, pred_time_dict
            except Exception:
                logger.warning("Cached predictions are corrupted. Predictions will be made from scratch.")
        return {}, {}

    def _save_cached_pred_dicts(
        self,
        dataset_hash: str,
        model_pred_dict: dict[str, Optional[TimeSeriesDataFrame]],
        pred_time_dict: dict[str, float],
    ) -> None:
        cached_predictions = self._load_cached_predictions()
        # Do not save results for models that failed
        cached_predictions[dataset_hash] = {
            "model_pred_dict": {k: v for k, v in model_pred_dict.items() if v is not None},
            "pred_time_dict": {k: v for k, v in pred_time_dict.items() if v is not None},
        }
        save_pkl.save(str(self._cached_predictions_path), object=cached_predictions)
        logger.debug(f"Cached predictions saved to {self._cached_predictions_path}")

    def _merge_refit_full_data(
        self, train_data: TimeSeriesDataFrame, val_data: Optional[TimeSeriesDataFrame]
    ) -> TimeSeriesDataFrame:
        if val_data is None:
            return train_data
        else:
            # TODO: Implement merging of arbitrary tuning_data with train_data
            raise NotImplementedError("refit_full is not supported if custom val_data is provided.")

    def refit_single_full(
        self,
        train_data: Optional[TimeSeriesDataFrame] = None,
        val_data: Optional[TimeSeriesDataFrame] = None,
        models: Optional[list[str]] = None,
    ) -> list[str]:
        train_data = train_data or self.load_train_data()
        val_data = val_data or self.load_val_data()
        refit_full_data = self._merge_refit_full_data(train_data, val_data)

        if models is None:
            models = self.get_model_names()

        model_to_level = self._get_model_levels()
        models_sorted_by_level = sorted(models, key=model_to_level.get)  # type: ignore

        model_refit_map = {}
        models_trained_full = []
        for model in models_sorted_by_level:
            model = self.load_model(model)
            model_name = model.name
            if model._get_tags()["can_refit_full"]:
                model_full = model.convert_to_refit_full_template()
                assert isinstance(model_full, AbstractTimeSeriesModel)
                logger.info(f"Fitting model: {model_full.name}")
                models_trained = self._train_and_save(
                    train_data=refit_full_data,
                    val_data=None,
                    model=model_full,
                )
            else:
                model_full = model.convert_to_refit_full_via_copy()
                logger.info(f"Fitting model: {model_full.name} | Skipping fit via cloning parent ...")
                models_trained = [model_full.name]
                if isinstance(model_full, AbstractTimeSeriesEnsembleModel):
                    model_full.remap_base_models(model_refit_map)
                    self._add_model(model_full, base_models=model_full.model_names)
                else:
                    self._add_model(model_full)
                self.save_model(model_full)

            if len(models_trained) == 1:
                model_refit_map[model_name] = models_trained[0]
            models_trained_full += models_trained

        self.model_refit_map.update(model_refit_map)
        self.save()
        return models_trained_full

    def refit_full(self, model: str = "all") -> dict[str, str]:
        time_start = time.time()
        existing_models = self.get_model_names()
        if model == "all":
            model_names = existing_models
        elif model == "best":
            model_names = self.get_minimum_model_set(self.get_model_best())
        else:
            model_names = self.get_minimum_model_set(model)

        valid_model_set = []
        for name in model_names:
            if name in self.model_refit_map and self.model_refit_map[name] in existing_models:
                logger.info(
                    f"Model '{name}' already has a refit _FULL model: "
                    f"'{self.model_refit_map[name]}', skipping refit..."
                )
            elif name in self.model_refit_map.values():
                logger.debug(f"Model '{name}' is a refit _FULL model, skipping refit...")
            else:
                valid_model_set.append(name)

        if valid_model_set:
            models_trained_full = self.refit_single_full(models=valid_model_set)
        else:
            models_trained_full = []

        self.save()
        logger.info(f"Refit complete. Models trained: {models_trained_full}")
        logger.info(f"Total runtime: {time.time() - time_start:.2f} s")
        return copy.deepcopy(self.model_refit_map)

    def construct_model_templates(
        self,
        hyperparameters: Union[str, dict[str, Any]],
        *,
        multi_window: bool = False,
        freq: Optional[str] = None,
        excluded_model_types: Optional[list[str]] = None,
        hyperparameter_tune: bool = False,
    ) -> list[TimeSeriesModelBase]:
        return get_preset_models(
            path=self.path,
            eval_metric=self.eval_metric,
            prediction_length=self.prediction_length,
            freq=freq,
            hyperparameters=hyperparameters,
            hyperparameter_tune=hyperparameter_tune,
            quantile_levels=self.quantile_levels,
            all_assigned_names=self._get_banned_model_names(),
            target=self.target,
            covariate_metadata=self.covariate_metadata,
            excluded_model_types=excluded_model_types,
            # if skip_model_selection = True, we skip backtesting
            multi_window=multi_window and not self.skip_model_selection,
        )

    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        hyperparameters: Union[str, dict[Any, dict]],
        val_data: Optional[TimeSeriesDataFrame] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, dict]] = None,
        excluded_model_types: Optional[list[str]] = None,
        time_limit: Optional[float] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Fit a set of timeseries models specified by the `hyperparameters`
        dictionary that maps model names to their specified hyperparameters.

        Parameters
        ----------
        train_data
            Training data for fitting time series timeseries models.
        hyperparameters
            A dictionary mapping selected model names, model classes or model factory to hyperparameter
            settings. Model names should be present in `trainer.presets.DEFAULT_MODEL_NAMES`. Optionally,
            the user may provide one of "default", "light" and "very_light" to specify presets.
        val_data
            Optional validation data set to report validation scores on.
        hyperparameter_tune_kwargs
            Args for hyperparameter tuning
        excluded_model_types
            Names of models that should not be trained, even if listed in `hyperparameters`.
        time_limit
            Time limit for training
        random_seed
            Random seed that will be set to each model during training
        """
        self._train_multi(
            train_data,
            val_data=val_data,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            excluded_model_types=excluded_model_types,
            time_limit=time_limit,
            random_seed=random_seed,
        )
