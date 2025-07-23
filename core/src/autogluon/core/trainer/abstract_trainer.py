from __future__ import annotations

import os
from typing import Any, Generic, Type, TypeVar

import networkx as nx
from typing_extensions import Self

from autogluon.core.models import ModelBase 
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_json, save_pkl

ModelTypeT = TypeVar("ModelTypeT", bound=ModelBase)


class AbstractTrainer(Generic[ModelTypeT]):
    trainer_file_name = "trainer.pkl"
    trainer_info_name = "info.pkl"
    trainer_info_json_name = "info.json"

    def __init__(self, path: str, *, low_memory: bool, save_data: bool):
        self.path = path
        self.reset_paths = False

        self.low_memory: bool = low_memory
        self.save_data: bool = save_data

        #: dict of model name -> model object. A key, value pair only exists if a model is persisted in memory.
        self.models: dict[str, Any] = {}

        #: Directed Acyclic Graph (DAG) of model interactions. Describes how certain models depend on the predictions of certain
        #: other models. Contains numerous metadata regarding each model.
        self.model_graph = nx.DiGraph()
        self.model_best: str | None = None

        #: Names which are banned but are not used by a trained model.
        self._extra_banned_names: set[str] = set()

    def _get_banned_model_names(self) -> list[str]:
        """Gets all model names which would cause model files to be overwritten if a new model
        was trained with the name
        """
        return self.get_model_names() + list(self._extra_banned_names)

    @property
    def path_root(self) -> str:
        """directory containing learner.pkl"""
        return os.path.dirname(self.path)

    @property
    def path_utils(self) -> str:
        return os.path.join(self.path_root, "utils")

    @property
    def path_data(self) -> str:
        return os.path.join(self.path_utils, "data")

    def set_contexts(self, path_context: str) -> None:
        self.path = self.create_contexts(path_context)

    def create_contexts(self, path_context: str) -> str:
        path = path_context
        return path

    def save_model(self, model: ModelTypeT) -> None:
        model.save()
        if not self.low_memory:
            self.models[model.name] = model

    def get_models_attribute_dict(self, attribute: str, models: list[str] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def get_model_attribute(self, model: str | ModelTypeT, attribute: str, **kwargs) -> Any:
        """Return model attribute value.
        If `default` is specified, return default value if attribute does not exist.
        If `default` is not specified, raise ValueError if attribute does not exist.
        """
        if not isinstance(model, str):
            model = model.name
        if model not in self.model_graph.nodes:
            raise ValueError(f"Model does not exist: (model={model})")
        if attribute not in self.model_graph.nodes[model]:
            if "default" in kwargs:
                return kwargs["default"]
            else:
                raise ValueError(f"Model does not contain attribute: (model={model}, attribute={attribute})")
        if attribute == "path":
            return os.path.join(*self.model_graph.nodes[model][attribute])
        return self.model_graph.nodes[model][attribute]

    def set_model_attribute(self, model: str | ModelTypeT, attribute: str, val: Any) -> None:
        if not isinstance(model, str):
            model = model.name
        self.model_graph.nodes[model][attribute] = val

    def get_minimum_model_set(self, model: str | ModelTypeT, include_self: bool = True) -> list:
        """Gets the minimum set of models that the provided model depends on, including itself
        Returns a list of model names
        """
        if not isinstance(model, str):
            model = model.name
        minimum_model_set = list(nx.bfs_tree(self.model_graph, model, reverse=True))
        if not include_self:
            minimum_model_set = [m for m in minimum_model_set if m != model]
        return minimum_model_set

    def get_model_info(self, model: str | ModelTypeT) -> dict[str, Any]:
        if isinstance(model, str):
            if model in self.models.keys():
                model = self.models[model]
        if isinstance(model, str):
            model_type = self.get_model_attribute(model=model, attribute="type")
            model_path = self.get_model_attribute(model=model, attribute="path")
            model_info = model_type.load_info(path=os.path.join(self.path, model_path))
        else:
            model_info = model.get_info()
        return model_info

    def get_model_names(self) -> list[str]:
        """Get all model names that are registered in the model graph, in no particular order."""
        return list(self.model_graph.nodes)

    def get_models_info(self, models: list[str | ModelTypeT] | None = None) -> dict[str, dict[str, Any]]:
        models_ = self.get_model_names() if models is None else models
        model_info_dict = dict()
        for model in models_:
            model_name = model if isinstance(model, str) else model.name
            model_info_dict[model_name] = self.get_model_info(model=model)
        return model_info_dict

    # TODO: model_name change to model in params
    def load_model(
        self, model_name: str | ModelTypeT, path: str | None = None, model_type: Type[ModelTypeT] | None = None
    ) -> ModelTypeT:
        if isinstance(model_name, ModelBase):
            return model_name
        if model_name in self.models.keys():
            return self.models[model_name]
        else:
            if path is None:
                path = self.get_model_attribute(
                    model=model_name, attribute="path"
                )  # get relative location of the model to the trainer
                assert path is not None
            if model_type is None:
                model_type = self.get_model_attribute(model=model_name, attribute="type")
                assert model_type is not None
            return model_type.load(path=os.path.join(self.path, path), reset_paths=self.reset_paths)

    @classmethod
    def load_info(cls, path: str, reset_paths: bool = False, load_model_if_required: bool = True) -> dict[str, Any]:
        load_path = os.path.join(path, cls.trainer_info_name)
        try:
            return load_pkl.load(path=load_path)
        except:
            if load_model_if_required:
                trainer = cls.load(path=path, reset_paths=reset_paths)
                return trainer.get_info()
            else:
                raise

    def save_info(self, include_model_info: bool = False) -> dict[str, Any]:
        info = self.get_info(include_model_info=include_model_info)

        save_pkl.save(path=os.path.join(self.path, self.trainer_info_name), object=info)
        save_json.save(path=os.path.join(self.path, self.trainer_info_json_name), obj=info)
        return info

    def construct_model_templates(
        self, hyperparameters: dict[str, Any]
    ) -> tuple[list[ModelTypeT], dict] | list[ModelTypeT]:
        raise NotImplementedError

    def get_model_best(self) -> str:
        raise NotImplementedError

    def get_info(self, include_model_info: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, reset_paths: bool = False) -> Self:
        load_path = os.path.join(path, cls.trainer_file_name)
        if not reset_paths:
            return load_pkl.load(path=load_path)
        else:
            obj = load_pkl.load(path=load_path)
            obj.set_contexts(path)
            obj.reset_paths = reset_paths
            return obj

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError
