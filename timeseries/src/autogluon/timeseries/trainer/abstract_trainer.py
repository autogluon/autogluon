import copy
import logging
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from warnings import warn

import networkx as nx
import pandas as pd
from tqdm import tqdm

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.core.models import AbstractModel
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_json, save_pkl
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesEvaluator
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel, TimeSeriesGreedyEnsemble
from autogluon.timeseries.models.gluonts.abstract_gluonts import AbstractGluonTSModel
from autogluon.timeseries.models.presets import contains_searchspace
from autogluon.timeseries.utils.features import CovariateMetadata
from autogluon.timeseries.utils.warning_filters import disable_tqdm

logger = logging.getLogger("autogluon.timeseries.trainer")


# TODO: This class is meant to be moved to `core`, where it will likely
# TODO: be renamed `AbstractTrainer` and the current `AbstractTrainer`
# TODO: will inherit from this class.
# TODO: add documentation for abstract methods
class SimpleAbstractTrainer:
    trainer_file_name = "trainer.pkl"
    trainer_info_name = "info.pkl"
    trainer_info_json_name = "info.json"

    def __init__(self, path: str, low_memory: bool, save_data: bool, *args, **kwargs):
        self.path = path
        self.reset_paths = False

        self.low_memory = low_memory
        self.save_data = save_data

        self.models = {}
        self.model_graph = nx.DiGraph()
        self.model_best = None

        self._extra_banned_names = set()

    def get_model_names(self, **kwargs) -> List[str]:
        """Get all model names that are registered in the model graph"""
        return list(self.model_graph.nodes)

    def _get_banned_model_names(self) -> List[str]:
        """Gets all model names which would cause model files to be overwritten if a new model
        was trained with the name
        """
        return self.get_model_names() + list(self._extra_banned_names)

    def get_models_attribute_dict(self, attribute: str, models: List[str] = None) -> Dict[str, Any]:
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
        model_performances = self.get_models_attribute_dict(attribute="val_score")
        performances_list = [(m, model_performances[m]) for m in models if model_performances[m] is not None]

        if not performances_list:
            raise ValueError("No fitted models have validation scores computed.")

        return max(performances_list, key=lambda i: i[1])[0]

    def get_model_attribute(self, model: Union[str, AbstractModel], attribute: str):
        """Get a member attribute for given model from the `model_graph`."""
        if not isinstance(model, str):
            model = model.name
        return self.model_graph.nodes[model][attribute]

    def set_model_attribute(self, model: Union[str, AbstractModel], attribute: str, val):
        """Set a member attribute for given model in the `model_graph`."""
        if not isinstance(model, str):
            model = model.name
        self.model_graph.nodes[model][attribute] = val

    @property
    def path_root(self) -> str:
        return self.path.rsplit(os.path.sep, maxsplit=2)[0] + os.path.sep

    @property
    def path_utils(self) -> str:
        return self.path_root + "utils" + os.path.sep

    @property
    def path_data(self) -> str:
        return self.path_utils + "data" + os.path.sep

    @property
    def path_pkl(self) -> str:
        return self.path + self.trainer_file_name

    def set_contexts(self, path_context: str) -> None:
        self.path, model_paths = self.create_contexts(path_context)
        for model, path in model_paths.items():
            self.set_model_attribute(model=model, attribute="path", val=path)

    def create_contexts(self, path_context: str) -> Tuple[str, dict]:
        path = path_context
        # TODO: consider keeping track of model path suffixes in model_graph instead
        # TODO: of full paths
        model_paths = self.get_models_attribute_dict(attribute="path")
        for model, prev_path in model_paths.items():
            model_local_path = prev_path.split(self.path, 1)[1]
            new_path = path + model_local_path
            model_paths[model] = new_path

        return path, model_paths

    def save(self) -> None:
        # todo: remove / revise low_memory logic
        models = self.models
        if self.low_memory:
            self.models = {}
        try:
            save_pkl.save(path=self.path_pkl, object=self)
        except:  # noqa
            self.models = {}
            save_pkl.save(path=self.path_pkl, object=self)
        if not self.models:
            self.models = models

    @classmethod
    def load(cls, path: str, reset_paths: bool = False) -> "SimpleAbstractTrainer":
        load_path = path + cls.trainer_file_name
        if not reset_paths:
            return load_pkl.load(path=load_path)
        else:
            obj = load_pkl.load(path=load_path)
            obj.set_contexts(path)
            obj.reset_paths = reset_paths
            return obj

    def save_model(self, model: AbstractModel, **kwargs) -> None:  # noqa: F841
        model.save()
        if not self.low_memory:
            self.models[model.name] = model

    def load_model(
        self,
        model_name: Union[str, AbstractModel],
        path: Optional[str] = None,
        model_type: Optional[Type[AbstractModel]] = None,
    ):
        if isinstance(model_name, AbstractModel):
            return model_name
        if model_name in self.models.keys():
            return self.models[model_name]

        if path is None:
            path = self.get_model_attribute(model=model_name, attribute="path")
        if model_type is None:
            model_type = self.get_model_attribute(model=model_name, attribute="type")
        return model_type.load(path=path, reset_paths=self.reset_paths)

    def construct_model_templates(self, hyperparameters: Union[str, Dict[str, Any]], **kwargs):
        raise NotImplementedError

    # TODO: This is horribly inefficient beyond simple weighted ensembling.
    #  Refactor to Tabular's implementation if doing stacking / multiple ensembles
    def get_inputs_to_model(
        self, model, X, model_pred_proba_dict=None, known_covariates: Optional[TimeSeriesDataFrame] = None
    ):
        if model_pred_proba_dict is None:
            model_pred_proba_dict = {}
        model_set = self.get_minimum_model_set(model, include_self=False)
        if model_set:
            for m in model_set:
                if m not in model_pred_proba_dict:
                    model_pred_proba_dict[m] = self.predict(model=m, data=X, known_covariates=known_covariates)
            return model_pred_proba_dict
        else:
            return X

    # FIXME: Copy pasted from Tabular
    def get_minimum_model_set(self, model: Union[str, AbstractTimeSeriesModel], include_self: bool = True) -> list:
        """Gets the minimum set of models that the provided model depends on, including itself.
        Returns a list of model names"""
        if not isinstance(model, str):
            model = model.name
        minimum_model_set = list(nx.bfs_tree(self.model_graph, model, reverse=True))
        if not include_self:
            minimum_model_set = [m for m in minimum_model_set if m != model]
        return minimum_model_set

    def get_models_info(self, models: List[str] = None) -> Dict[str, Any]:
        if models is None:
            models = self.get_model_names()
        model_info_dict = dict()
        for model in models:
            if isinstance(model, str):
                if model in self.models.keys():
                    model = self.models[model]
            if isinstance(model, str):
                model_type = self.get_model_attribute(model=model, attribute="type")
                model_path = self.get_model_attribute(model=model, attribute="path")
                model_info_dict[model] = model_type.load_info(path=model_path)
            else:
                model_info_dict[model.name] = model.get_info()
        return model_info_dict

    @classmethod
    def load_info(cls, path, reset_paths=False, load_model_if_required=True) -> Dict[str, Any]:
        load_path = path + cls.trainer_info_name
        try:
            return load_pkl.load(path=load_path)
        except:  # noqa
            if load_model_if_required:
                trainer = cls.load(path=path, reset_paths=reset_paths)
                return trainer.get_info()
            else:
                raise

    def save_info(self, include_model_info: bool = False):
        info = self.get_info(include_model_info=include_model_info)

        save_pkl.save(path=self.path + self.trainer_info_name, object=info)
        save_json.save(path=self.path + self.trainer_info_json_name, obj=info)
        return info

    def get_info(self, include_model_info: bool = False) -> Dict[str, Any]:
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

    def predict(self, *args, **kwargs):
        raise NotImplementedError


class AbstractTimeSeriesTrainer(SimpleAbstractTrainer):
    def __init__(
        self,
        path: str,
        prediction_length: Optional[int] = 1,
        eval_metric: Optional[str] = None,
        save_data: bool = True,
        enable_ensemble: bool = True,
        verbosity: int = 2,
        **kwargs,
    ):
        super().__init__(path=path, save_data=save_data, low_memory=True, **kwargs)

        self.prediction_length = prediction_length
        self.quantile_levels = kwargs.get(
            "quantile_levels",
            kwargs.get("quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        )
        self.target = kwargs.get("target", "target")
        self.metadata = kwargs.get("metadata", CovariateMetadata())
        self.is_data_saved = False
        self.enable_ensemble = enable_ensemble
        self.ensemble_model_type = TimeSeriesGreedyEnsemble

        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)

        # Dict of normal model -> FULL model. FULL models are produced by
        # self.refit_single_full() and self.refit_ensemble_full().
        self.model_full_dict = {}

        # Dict of FULL model -> normal model validation score in case the normal model had been deleted.
        self._model_full_dict_val_score = {}
        self.eval_metric = TimeSeriesEvaluator.check_get_evaluation_metric(eval_metric)
        self.hpo_results = {}

    def save_train_data(self, data: TimeSeriesDataFrame, verbose: bool = True) -> None:
        path = self.path_data + "train.pkl"
        save_pkl.save(path=path, object=data, verbose=verbose)

    def save_val_data(self, data: TimeSeriesDataFrame, verbose: bool = True) -> None:
        path = self.path_data + "val.pkl"
        save_pkl.save(path=path, object=data, verbose=verbose)

    def load_train_data(self) -> TimeSeriesDataFrame:
        path = self.path_data + "train.pkl"
        return load_pkl.load(path=path)

    def load_val_data(self) -> TimeSeriesDataFrame:
        path = self.path_data + "val.pkl"
        return load_pkl.load(path=path)

    def load_data(self) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
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

    def _add_model(
        self,
        model: AbstractTimeSeriesModel,
        base_models: List[str] = None,
    ):
        """Add a model to the model graph of the trainer. If the model is an ensemble, also add
        information about dependencies to the model graph (list of models specified via ``base_models``).

        Parameters
        ----------
        model : AbstractTimeSeriesModel
            The model to be added to the model graph.
        base_models : List[str], optional, default None
            If the model is an ensemble, the list of base model names that are included in the ensemble.
            Expected only when ``model`` is a ``AbstractTimeSeriesEnsembleModel``.

        Raises
        ------
        AssertionError
            If ``base_models`` are provided and ``model`` is not a ``AbstractTimeSeriesEnsembleModel``.
        """
        node_attrs = dict(
            path=model.path,
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

    def _get_model_levels(self) -> Dict[str, int]:
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

    def get_model_names(self, level: Optional[int] = None, **kwargs) -> List[str]:
        """Get model names that are registered in the model graph"""
        if level is not None:
            return list(node for node, l in self._get_model_levels().items() if l == level)
        return list(self.model_graph.nodes)

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
            )
        total_tuning_time = time.time() - tuning_start_time

        self.hpo_results[model.name] = hpo_models
        model_names_trained = []
        # add each of the trained HPO configurations to the trained models
        for model_hpo_name, model_info in hpo_models.items():
            model_path = model_info["path"]
            # Only load model configurations that didn't fail
            if Path(model_path).exists():
                model_hpo = self.load_model(model_hpo_name, path=model_path, model_type=type(model))
                self._add_model(model_hpo)
                model_names_trained.append(model_hpo.name)

        logger.info(f"\tTrained {len(model_names_trained)} models while tuning {model.name}.")

        if len(model_names_trained) > 0:
            if TimeSeriesEvaluator.METRIC_COEFFICIENTS[self.eval_metric] == -1:
                sign_str = "-"
            else:
                sign_str = ""

            trained_model_results = [hpo_models[model_name] for model_name in model_names_trained]
            best_model_result = max(trained_model_results, key=lambda x: x["val_score"])

            logger.info(
                f"\t{best_model_result['val_score']:<7.4f}".ljust(15)
                + f"= Validation score ({sign_str}{self.eval_metric})"
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
    ) -> List[str]:
        """Fit and save the given model on given training and validation data and save the
        trained model.

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

            val_score = model.score(val_data) if val_data is not None else None
            model.val_score = val_score

            pred_end_time = time.time()

            model.fit_time = model.fit_time or (fit_end_time - fit_start_time)
            if model.predict_time is None:
                model.predict_time = None if val_score is None else (pred_end_time - fit_end_time)

            self._log_scores_and_times(val_score, model.fit_time, model.predict_time)

            self.save_model(model=model)
        except (Exception, MemoryError) as err:
            logger.error(f"\tWarning: Exception caused {model.name} to fail during training... Skipping this model.")
            logger.error(f"\t{err}")
            logger.debug(traceback.format_exc())
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
            if TimeSeriesEvaluator.METRIC_COEFFICIENTS[self.eval_metric] == -1:
                sign_str = "-"
            else:
                sign_str = ""
            logger.info(f"\t{val_score:<7.4f}".ljust(15) + f"= Validation score ({sign_str}{self.eval_metric})")
        if fit_time is not None:
            logger.info(f"\t{fit_time:<7.2f} s".ljust(15) + "= Training runtime")
        if predict_time is not None:
            logger.info(f"\t{predict_time:<7.2f} s".ljust(15) + "= Validation (prediction) runtime")

    def _train_multi(
        self,
        train_data: TimeSeriesDataFrame,
        hyperparameters: Optional[Union[str, Dict]] = None,
        models: Optional[List[AbstractTimeSeriesModel]] = None,
        val_data: Optional[TimeSeriesDataFrame] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, dict]] = None,
        time_limit: Optional[float] = None,
    ) -> List[str]:

        logger.info(f"\nStarting training. Start time is {time.strftime('%Y-%m-%d %H:%M:%S')}")

        time_start = time.time()
        if hyperparameters is not None:
            hyperparameters = copy.deepcopy(hyperparameters)
        else:
            if models is None:
                raise ValueError("Either models or hyperparameters should be provided")

        if self.save_data and not self.is_data_saved:
            self.save_train_data(train_data)
            if val_data is not None:
                self.save_val_data(val_data)
            self.is_data_saved = True

        if models is None:
            models = self.construct_model_templates(
                hyperparameters=hyperparameters,
                hyperparameter_tune=hyperparameter_tune_kwargs is not None,  # TODO: remove hyperparameter_tune
                freq=train_data.freq,
            )

        logger.info(f"Models that will be trained: {list(m.name for m in models)}")

        time_limit_model_split = time_limit
        if time_limit is not None and len(models) > 0:
            time_limit_model_split /= len(models)

        model_names_trained = []
        for i, model in enumerate(models):
            if hyperparameter_tune_kwargs is not None:
                time_left = time_limit_model_split

                fit_log_message = f"Hyperparameter tuning model: {model.name}. "
                if time_limit is not None and time_limit_model_split is not None:
                    fit_log_message += (
                        f"Tuning model for up to {time_limit_model_split:.2f}s " f"of the {time_limit:.2f}s remaining."
                    )
                logger.info(fit_log_message)

                if contains_searchspace(model.get_user_params()):
                    with tqdm.external_write_mode():
                        model_names_trained += self.tune_model_hyperparameters(
                            model,
                            time_limit=time_left,
                            train_data=train_data,
                            val_data=val_data,
                            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        )
                else:
                    model_names_trained += self._train_and_save(
                        train_data, model=model, val_data=val_data, time_limit=time_left
                    )
            else:
                time_left = None
                fit_log_message = f"Training timeseries model {model.name}. "
                if time_limit is not None:
                    time_start_model = time.time()
                    time_left = time_limit - (time_start_model - time_start)
                    if time_left <= 0:
                        logger.info(
                            f"Stopping training due to lack of time remaining. Time left: {time_left:.2f} seconds"
                        )
                        break

                    fit_log_message += (
                        f"Training for up to {time_left:.2f}s of " f"the {time_left:.2f}s of remaining time."
                    )

                logger.info(fit_log_message)
                model_names_trained += self._train_and_save(
                    train_data, model=model, val_data=val_data, time_limit=time_left
                )

        if self.enable_ensemble:
            models_available_for_ensemble = self.get_model_names(level=0)

            time_left_for_ensemble = None
            if time_limit is not None:
                time_left_for_ensemble = time_limit - (time.time() - time_start)

            if time_left_for_ensemble is not None and time_left_for_ensemble <= 0:
                logger.info(
                    "Not fitting ensemble due to lack of time remaining. "
                    f"Time left: {time_left_for_ensemble:.2f} seconds"
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
                            val_data=val_data,
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
            logger.info(f"Best model score: {self.get_model_attribute(best_model, 'val_score'):.4f}")
        except ValueError as e:
            logger.error(str(e))

        return model_names_trained

    def _get_ensemble_model_name(self) -> str:
        """Ensure we don't have name collisions in the ensemble model name"""
        ensemble_name = "WeightedEnsemble"
        increment = 1
        while ensemble_name in self._get_banned_model_names():
            increment += 1
            ensemble_name = f"WeightedEnsemble_{increment}"
        return ensemble_name

    def fit_ensemble(
        self, val_data: TimeSeriesDataFrame, model_names: List[str], time_limit: Optional[float] = None
    ) -> str:
        logger.info("Fitting simple weighted ensemble.")

        model_preds = {}
        for model_name in model_names:
            model: AbstractGluonTSModel = self.load_model(model_name=model_name)

            # FIXME: This differs from predictions made to calc val_score for the models. Try to align.
            #  Can either seed for deterministic results or cache the pred during val_score calc and reuse.
            model_preds[model_name] = model.predict_for_scoring(data=val_data, quantile_levels=self.quantile_levels)

        time_start = time.time()
        ensemble = self.ensemble_model_type(
            name=self._get_ensemble_model_name(),
            eval_metric=self.eval_metric,
            target=self.target,
            prediction_length=self.prediction_length,
            path=self.path,
            freq=val_data.freq,
            quantile_levels=self.quantile_levels,
            metadata=self.metadata,
        )
        ensemble.fit_ensemble(predictions=model_preds, data=val_data, time_limit=time_limit)
        time_end = time.time()
        ensemble.fit_time = time_end - time_start

        evaluator = TimeSeriesEvaluator(
            eval_metric=self.eval_metric,
            prediction_length=self.prediction_length,
            target_column=self.target,
        )
        forecasts = ensemble.predict({n: model_preds[n] for n in ensemble.model_names})
        ensemble.val_score = evaluator(val_data, forecasts) * evaluator.coefficient

        predict_time = 0
        for m in ensemble.model_names:
            predict_time += self.get_model_attribute(model=m, attribute="predict_time")
        ensemble.predict_time = predict_time

        self._log_scores_and_times(
            val_score=ensemble.val_score,
            fit_time=ensemble.fit_time,
            predict_time=ensemble.predict_time,
        )

        self._add_model(model=ensemble, base_models=ensemble.model_names)
        self.save_model(model=ensemble)
        return ensemble.name

    def leaderboard(self, data: Optional[TimeSeriesDataFrame] = None) -> pd.DataFrame:
        logger.debug("Generating leaderboard for all models trained")

        model_names = self.get_model_names()
        model_info = {}
        for ix, model_name in enumerate(model_names):
            model_info[model_name] = {
                "model": model_name,
                "fit_order": ix + 1,
                "score_val": self.get_model_attribute(model_name, "val_score"),
                "fit_time_marginal": self.get_model_attribute(model_name, "fit_time"),
                "pred_time_val": self.get_model_attribute(model_name, "predict_time"),
            }

        if data is not None:
            logger.info(
                "Additional data provided, testing on additional data. Resulting leaderboard "
                "will be sorted according to test score (`score_test`)."
            )
            for model_name in model_names:
                try:
                    # TODO: time only prediction and not score for pred_time_val and pred_time_test
                    time_start_test_score = time.time()
                    model_info[model_name]["score_test"] = self.score(data, model_name)
                    model_info[model_name]["pred_time_test"] = time.time() - time_start_test_score
                except Exception as e:  # noqa
                    logger.error(f"Cannot score with model {model_name}. An error occurred: {str(e)}")
                    logger.debug(traceback.format_exc())
                    model_info[model_name]["score_test"] = float("nan")
                    model_info[model_name]["pred_time_test"] = float("nan")

        df = pd.DataFrame(model_info.values())

        sort_column = "score_test" if "score_test" in df.columns else "score_val"
        df.sort_values(by=[sort_column, "model"], ascending=[False, False], inplace=True)
        df.reset_index(drop=True, inplace=True)

        explicit_column_order = [
            "model",
            "score_test",
            "score_val",
            "pred_time_test",
            "pred_time_val",
            "fit_time_marginal",
            "fit_order",
        ]
        explicit_column_order = [c for c in explicit_column_order if c in df.columns] + [
            c for c in df.columns if c not in explicit_column_order
        ]

        return df[explicit_column_order]

    def _get_model_for_prediction(
        self, model: Optional[Union[str, AbstractTimeSeriesModel]] = None
    ) -> AbstractTimeSeriesModel:
        """Given an optional identifier or model object, return the model to perform predictions
        with. If the model is not provided, this method will default to the best model according to
        the validation score."""

        if model is None:
            if self.model_best is None:
                best_model_name: str = self.get_model_best()
                self.model_best = best_model_name
            logger.info(
                f"Model not specified in predict, will default to the model with the "
                f"best validation score: {self.model_best}",
            )
            return self.load_model(self.model_best)
        else:
            if isinstance(model, str):
                return self.load_model(model)
            return model

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        model: Optional[Union[str, AbstractTimeSeriesModel]] = None,
        **kwargs,
    ) -> Union[TimeSeriesDataFrame, None]:
        model_was_selected_automatically = model is None
        model = self._get_model_for_prediction(model)
        try:
            return self._predict_model(data, model, known_covariates=known_covariates, **kwargs)
        except Exception as err:
            logger.error(f"Warning: Model {model.name} failed during prediction with exception: {err}")
            logger.debug(traceback.format_exc())
            other_models = [m for m in self.get_model_names() if m != model.name]
            if len(other_models) > 0 and model_was_selected_automatically:
                logger.info(f"\tYou can call predict(data, model) with one of other available models: {other_models}")
            return None

    def score(
        self,
        data: TimeSeriesDataFrame,
        model: Optional[Union[str, AbstractTimeSeriesModel]] = None,
        metric: Optional[str] = None,
    ) -> float:
        model = self._get_model_for_prediction(model)
        eval_metric = self.eval_metric if metric is None else metric

        if isinstance(model, AbstractTimeSeriesEnsembleModel):
            evaluator = TimeSeriesEvaluator(
                eval_metric=eval_metric,
                prediction_length=self.prediction_length,
                target_column=self.target,
            )
            model_preds = {}
            base_models = self.get_minimum_model_set(model, include_self=False)
            for base_model in base_models:
                try:
                    base_model_loaded = self._get_model_for_prediction(base_model)
                    model_preds[base_model] = base_model_loaded.predict_for_scoring(
                        data, quantile_levels=self.quantile_levels
                    )
                except Exception:
                    model_preds[base_model] = None
            forecasts = model.predict(model_preds)

            model_score = evaluator(data, forecasts) * evaluator.coefficient
            return model_score

        return model.score(data, metric=eval_metric)

    def _predict_model(
        self,
        data: TimeSeriesDataFrame,
        model: Union[str, AbstractTimeSeriesModel],
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if isinstance(model, str):
            model = self.load_model(model)
        data = self.get_inputs_to_model(model=model, X=data, known_covariates=known_covariates)
        return model.predict(data, known_covariates=known_covariates, **kwargs)

    # TODO: experimental
    def refit_single_full(self, train_data=None, val_data=None, models=None):
        warn("Refitting logic is experimental for autogluon.timeseries.")
        models_trained_full = []
        model_full_dict = {}

        train_data = train_data or self.load_train_data()  # noqa
        val_data = val_data or self.load_val_data()

        # FIXME: implement the append operation below in datasets
        # refit_full_data = train_data + val_data
        # for now we assume validation data includes train data (as in GluonTS)
        refit_full_data = val_data

        if models is None:
            self.get_model_names()

        for model in models:
            model: AbstractTimeSeriesModel = self.load_model(model)
            model_name = model.name
            model_full = model.convert_to_refit_full_template()  # FIXME: not available
            models_trained = self._train_multi(
                train_data=refit_full_data,
                val_data=None,
                hyperparameters=None,
                hyperparameter_tune_kwargs=None,
                models=[model_full],
            )

            if len(models_trained) == 1:
                model_full_dict[model_name] = models_trained[0]
            for model_trained in models_trained:
                self._model_full_dict_val_score[model_trained] = self.get_model_attribute(model_name, "val_score")
            models_trained_full += models_trained

        self.model_full_dict.update(model_full_dict)
        self.save()
        return models_trained_full

    # TODO: experimental
    def refit_full(self, models="all"):
        warn("Refitting logic is experimental for autogluon.timeseries.")
        if isinstance(models, str):
            if models == "all":
                models = self.get_model_names()
            elif models == "best":
                models = [self.get_model_best()]
            else:
                models = self.load_model(models)
        existing_models = self.get_model_names()
        valid_model_set = []
        for model in models:
            if model in self.model_full_dict and self.model_full_dict[model] in existing_models:
                logger.log(
                    20,
                    f"Model '{model}' already has a refit _FULL model: "
                    f"'{self.model_full_dict[model]}', skipping refit...",
                )
            else:
                valid_model_set.append(model)

        if valid_model_set:
            self.refit_single_full(models=valid_model_set)

        self.save()
        return copy.deepcopy(self.model_full_dict)

    def construct_model_templates(
        self, hyperparameters: Union[str, Dict[str, Any]], **kwargs
    ) -> List[AbstractTimeSeriesModel]:
        """Constructs a list of unfit models based on the hyperparameters dict."""
        raise NotImplementedError

    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        hyperparameters: Dict[str, Any],
        val_data: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    # TODO: def _filter_base_models_via_infer_limit

    # TODO: persist and unpersist models

    # TODO: generate weighted ensemble
