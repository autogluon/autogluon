import copy
import logging
import os
import time
from typing import Optional, Tuple, List, Any, Dict, Union, Type
from warnings import warn

import networkx as nx
import pandas as pd

from autogluon.core.models import AbstractModel
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.core.utils.savers import save_pkl, save_json
from autogluon.core.utils.loaders import load_pkl

from ..dataset import TimeSeriesDataFrame
from ..models.abstract import AbstractTimeSeriesModel
from ..models.gluonts.abstract_gluonts import AbstractGluonTSModel
from ..utils.metric_utils import check_get_evaluation_metric

logger = logging.getLogger(__name__)


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

    def get_model_names(self) -> List[str]:
        """Get all model names that are registered in the model graph"""
        return list(self.model_graph.nodes)

    def _get_banned_model_names(self) -> List[str]:
        """Gets all model names which would cause model files to be overwritten if a new model
        was trained with the name
        """
        return self.get_model_names() + list(self._extra_banned_names)

    def get_models_attribute_dict(
        self, attribute: str, models: List[str] = None
    ) -> Dict[str, Any]:
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
            raise AssertionError("Trainer has no fit models that can infer.")
        model_performances = self.get_models_attribute_dict(attribute="val_score")
        performances_list = [
            (m, model_performances[m])
            for m in models
            if model_performances[m] is not None
        ]
        return max(performances_list, key=lambda i: i[1])[0]

    def get_model_attribute(self, model: Union[str, AbstractModel], attribute: str):
        """Get a member attribute for given model from the `model_graph`."""
        if not isinstance(model, str):
            model = model.name
        return self.model_graph.nodes[model][attribute]

    def set_model_attribute(
        self, model: Union[str, AbstractModel], attribute: str, val
    ):
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
        else:
            if path is None:
                path = self.get_model_attribute(model=model_name, attribute="path")
            if model_type is None:
                model_type = self.get_model_attribute(
                    model=model_name, attribute="type"
                )
            return model_type.load(path=path, reset_paths=self.reset_paths)

    def construct_model_templates(
        self, hyperparameters: Union[str, Dict[str, Any]], **kwargs
    ):
        raise NotImplementedError

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
    def load_info(
        cls, path, reset_paths=False, load_model_if_required=True
    ) -> Dict[str, Any]:
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
            best_model_score_val = self.get_model_attribute(
                model=best_model, attribute="val_score"
            )
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


class AbstractTimeSeriesTrainer(SimpleAbstractTrainer):
    def __init__(
        self,
        path: str,
        prediction_length: Optional[int] = 1,
        eval_metric: Optional[str] = None,
        save_data: bool = True,
        **kwargs,
    ):
        super().__init__(
            path=path, save_data=save_data, low_memory=True, **kwargs
        )

        self.prediction_length = prediction_length
        self.quantile_levels = kwargs.get(
            "quantile_levels",
            kwargs.get("quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        )
        self.target = kwargs.get("target", "target")
        self.is_data_saved = False

        # Dict of normal model -> FULL model. FULL models are produced by
        # self.refit_single_full() and self.refit_ensemble_full().
        self.model_full_dict = {}

        # Dict of FULL model -> normal model validation score in case the normal model had been deleted.
        self._model_full_dict_val_score = {}
        self.eval_metric = check_get_evaluation_metric(eval_metric)
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

    def _add_model(self, model: AbstractTimeSeriesModel):
        # TODO: also register predict time
        node_attrs = dict(
            path=model.path,
            type=type(model),
            fit_time=model.fit_time,
            val_score=model.val_score,
        )
        self.model_graph.add_node(model.name, **node_attrs)

    def _train_single(
        self,
        train_data: TimeSeriesDataFrame,
        model: AbstractTimeSeriesModel,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
    ) -> AbstractTimeSeriesModel:
        """Train the single model and return the model object that was fitted. This method
        does not save the resulting model."""
        model.fit(train_data=train_data, val_data=val_data, time_limit=time_limit)
        return model

    def tune_model_hyperparameters(
        self,
        model: AbstractTimeSeriesModel,
        train_data: TimeSeriesDataFrame,
        time_limit: Optional[float] = None,
        val_data: Optional[TimeSeriesDataFrame] = None,
        hyperparameter_tune_kwargs: Union[str, dict] = "auto",
    ):
        scheduler_cls, scheduler_options = scheduler_factory(
            hyperparameter_tune_kwargs, time_out=time_limit
        )
        if all(scheduler_options.get(s) is None for s in ["num_trials", "time_out"]):
            scheduler_options["num_trials"] = 10
        hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(
            train_data=train_data,
            val_data=val_data,
            scheduler_options=(scheduler_cls, scheduler_options),
            time_limit=time_limit,
        )
        hpo_results.pop("search_strategy", None)
        self.hpo_results[model.name] = hpo_results
        model_names_trained = []
        # add each of the trained HPO configurations to the trained models
        for model_hpo_name, model_path in hpo_models.items():
            model_hpo = self.load_model(
                model_hpo_name, path=model_path, model_type=type(model)
            )
            self._add_model(model_hpo)
            model_names_trained.append(model_hpo.name)
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
                    logging.log(
                        15, f"Skipping {model.name} due to lack of time remaining."
                    )
                    return model_names_trained
            else:
                logging.log(20, f"Fitting model: {model.name} ...")
            model = self._train_single(
                train_data, model, val_data=val_data, time_limit=time_limit
            )
            fit_end_time = time.time()

            val_score = model.score(val_data) if val_data is not None else None
            model.val_score = val_score

            pred_end_time = time.time()

            model.fit_time = model.fit_time or (fit_end_time - fit_start_time)
            if model.predict_time is None:
                model.predict_time = (
                    None if val_score is None else (pred_end_time - fit_end_time)
                )

            self.save_model(model=model)
        except Exception as err:
            logger.error(
                f"\tWarning: Exception caused {model.name} to fail during training... Skipping this model."
            )
            logger.error(f"\t\t{err}")
            logger.exception("Detailed Traceback:")
        else:
            self._add_model(model=model)  # noqa: F821
            model_names_trained.append(model.name)  # noqa: F821
        finally:
            del model

        return model_names_trained

    def _train_multi(
        self,
        train_data: TimeSeriesDataFrame,
        hyperparameters: Optional[Union[str, Dict]] = None,
        models: Optional[List[AbstractTimeSeriesModel]] = None,
        val_data: Optional[TimeSeriesDataFrame] = None,
        hyperparameter_tune: bool = False,
        time_limit: Optional[float] = None,
    ) -> List[str]:
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
                hyperparameter_tune=hyperparameter_tune,
                freq=train_data.freq,
            )

        time_limit_model_split = time_limit
        if time_limit is not None and len(models) > 0:
            time_limit_model_split /= len(models)

        model_names_trained = []
        for i, model in enumerate(models):
            if hyperparameter_tune:
                time_left = time_limit_model_split
                model_names_trained += self.tune_model_hyperparameters(
                    model,
                    time_limit=time_left,
                    train_data=train_data,
                    val_data=val_data,
                )
            else:
                time_left = None
                if time_limit is not None:
                    time_start_model = time.time()
                    time_left = time_limit - (time_start_model - time_start)
                    if time_left <= 0:
                        logger.log(
                            30, "Stopping training due to lack of time remaining"
                        )
                        break

                model_names_trained += self._train_and_save(
                    train_data, model=model, val_data=val_data, time_limit=time_left
                )

        return model_names_trained

    def leaderboard(self, data: Optional[TimeSeriesDataFrame] = None) -> pd.DataFrame:
        logger.log(30, "Generating leaderboard for all models trained...")
        model_names = self.get_model_names()
        score_val = []
        fit_order = list(range(1, len(model_names) + 1))
        score_dict = self.get_models_attribute_dict("val_score")

        fit_time_marginal = []
        fit_time_marginal_dict = self.get_models_attribute_dict("fit_time")

        for model_name in model_names:
            score_val.append(score_dict[model_name])
            fit_time_marginal.append(fit_time_marginal_dict[model_name])

        test_score = []
        if data is not None:
            logger.log(
                30, "Additional data provided, testing on the additional data..."
            )
            for model_name in model_names:
                test_score.append(self.score(data, model_name))
        df = pd.DataFrame(
            data={
                "model": model_names,
                "score_val": score_val,
                "fit_time_marginal": fit_time_marginal,
                "fit_order": fit_order,
            }
        )
        if test_score:
            df["test_score"] = test_score

        df_sorted = df.sort_values(
            by=["score_val", "model"], ascending=[False, False]
        ).reset_index(drop=True)

        df_columns_lst = df_sorted.columns.tolist()
        explicit_order = ["model", "score_val", "fit_time_marginal", "fit_order"]
        explicit_order = [
            column for column in explicit_order if column in df_columns_lst
        ]
        df_columns_other = [
            column for column in df_columns_lst if column not in explicit_order
        ]
        df_columns_new = explicit_order + df_columns_other
        df_sorted = df_sorted[df_columns_new]

        return df_sorted

    def _get_model_for_prediction(
        self, model: Optional[Union[str, AbstractTimeSeriesModel]] = None
    ) -> AbstractTimeSeriesModel:
        """Given an optional identifier or model object, return the model to perform predictions
        with. If the model is not provided, this method will default to the best model according to
        the validation score."""

        if model is None:
            logger.log(
                30,
                "Model not specified, will default to the model with the best validation score",
            )
            if self.model_best is None:
                best_model_name: str = self.get_model_best()
                self.model_best = best_model_name
            return self.load_model(self.model_best)
        else:
            if isinstance(model, str):
                return self.load_model(model)
            return model

    def predict(
        self,
        data: TimeSeriesDataFrame,
        model: Optional[AbstractTimeSeriesModel] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        model = self._get_model_for_prediction(model)
        return self._predict_model(data, model, **kwargs)

    def score(
        self,
        data: TimeSeriesDataFrame,
        model: Optional[Union[str, AbstractModel]] = None,
        metric: Optional[str] = None
    ) -> float:
        model = self._get_model_for_prediction(model)

        # FIXME: this method should be able to score on all data sets regardless of
        # FIXME: whether the implementation is in GluonTS
        if not isinstance(model, AbstractGluonTSModel):
            raise ValueError("Model must be a GluonTS model to score")

        # FIXME: when ensembling is implemented, score logic will have to be revised
        # FIXME: in order to enable prior model predictions in the ensemble
        eval_metric = self.eval_metric if metric is None else metric
        return model.score(data, metric=eval_metric)

    def _predict_model(
        self,
        data: TimeSeriesDataFrame,
        model: Union[str, AbstractTimeSeriesModel],
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if isinstance(model, str):
            model = self.load_model(model)
        return model.predict(data, **kwargs)

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
                hyperparameter_tune=False,
                models=[model_full],
            )

            if len(models_trained) == 1:
                model_full_dict[model_name] = models_trained[0]
            for model_trained in models_trained:
                self._model_full_dict_val_score[
                    model_trained
                ] = self.get_model_attribute(model_name, "val_score")
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
            if (
                model in self.model_full_dict
                and self.model_full_dict[model] in existing_models
            ):
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
