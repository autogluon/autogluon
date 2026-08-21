from __future__ import annotations

from typing import Type

import pandas as pd

from autogluon.core.models import AbstractModel


# TODO: Move to core? Maybe TimeSeries can reuse?
# TODO: Use this / refer to this in the custom model tutorial
# TODO: Add to documentation website
# TODO: Test register logic in AG
class ModelRegistry:
    """
    ModelRegistry keeps track of all known model classes to AutoGluon.
    It can provide information such as:
        What model classes and keys are valid to specify in an AutoGluon predictor fit call.
        What a model's name is.
        What a model's key is (such as the key specified by the user in `hyperparameters` to refer to a specific model type).
        What a model's priority is (aka which order to fit a list of models).

    Additionally, users can register custom models to AutoGluon so the key is recognized in `hyperparameters` and is treated with the proper priority and name.
    They can register new models via `ModelRegistry.add(model_cls)`.

    Therefore, if a user creates a custom model `MyCustomModel` that inherits from `AbstractModel`, they can set the class attributes in `MyCustomModel`:
        ag_key: The string key that can be specified in `hyperparameters`. Example: "GBM" for LGBModel
        ag_name: The string name that is used in logging and accessing the model. Example: "LightGBM" for LGBModel
        ag_priority: The int priority that is used to order the fitting of models. Higher values will be fit before lower values. Default 0. Example: 90 for LGBModel
        ag_priority_to_problem_type: A dictionary of problem_type to priority that overrides `ag_priority` if specified for a given problem_type. Optional.

    Then they can say `ag_model_registry.add(MyCustomModel)`.
    Assuming MyCustomModel.ag_key = "MY_MODEL", they can now do:
    ```
    predictor.fit(..., hyperparameters={"MY_MODEL": ...})
    ```
    """

    def __init__(self, model_cls_list: list[Type[AbstractModel]] | None = None):
        if model_cls_list is None:
            model_cls_list = []
        assert isinstance(model_cls_list, list)
        self._model_cls_list = []
        self._key_to_cls_map = dict()
        for model_cls in model_cls_list:
            self.add(model_cls)

    def exists(self, model_cls: Type[AbstractModel]) -> bool:
        return model_cls in self._model_cls_list

    def add(self, model_cls: Type[AbstractModel]):
        """
        Adds `model_cls` to the model registry
        """
        assert not self.exists(model_cls), f"Cannot add model_cls that is already registered: {model_cls}"
        if model_cls.ag_key is None:
            raise AssertionError(
                f"Cannot add model_cls with `ag_key=None`. "
                f"Ensure you set class attribute `ag_key` to a string for your model_cls: {model_cls}"
                f'\n\tFor example, LightGBModel sets `ag_key = "GBM"`'
            )
        if model_cls.ag_name is None:
            raise AssertionError(
                f"Cannot add model_cls with `ag_name=None`. "
                f"Ensure you set class attribute `ag_name` to a string for your model_cls: {model_cls}"
                f'\n\tFor example, LightGBModel sets `ag_name = "LightGBM"`'
            )
        assert isinstance(model_cls.ag_key, str)
        assert isinstance(model_cls.ag_name, str)
        assert isinstance(model_cls.ag_priority, int)
        if model_cls.ag_key in self._key_to_cls_map:
            raise AssertionError(
                f"Cannot register a model class that shares a model key with an already registered model class."
                f"\n`model_cls.ag_key` must be unique among registered models:"
                f"\n\t        New  Class: {model_cls}"
                f"\n\tConflicting  Class: {self._key_to_cls_map[model_cls.ag_key]}"
                f"\n\tConflicting ag_key: {model_cls.ag_key}"
            )
        self._model_cls_list.append(model_cls)
        self._key_to_cls_map[model_cls.ag_key] = model_cls

    def remove(self, model_cls: Type[AbstractModel]):
        """
        Removes `model_cls` from the model registry
        """
        assert self.exists(model_cls), f"Cannot remove model_cls that isn't registered: {model_cls}"
        self._model_cls_list = [m for m in self._model_cls_list if m != model_cls]
        self._key_to_cls_map.pop(model_cls.ag_key)

    @property
    def model_cls_list(self) -> list[Type[AbstractModel]]:
        return self._model_cls_list

    @property
    def keys(self) -> list[str]:
        return [self.key(model_cls) for model_cls in self.model_cls_list]

    def key_to_cls_map(self) -> dict[str, Type[AbstractModel]]:
        return self._key_to_cls_map

    def key_to_cls(self, key: str) -> Type[AbstractModel]:
        if key not in self._key_to_cls_map:
            raise ValueError(
                f"No registered model exists with provided key: {key}"
                f"\n\tValid keys: {list(self.key_to_cls_map().keys())}"
            )
        return self.key_to_cls_map()[key]

    def priority_map(self, problem_type: str | None = None) -> dict[Type[AbstractModel], int]:
        return {model_cls: self.priority(model_cls, problem_type=problem_type) for model_cls in self._model_cls_list}

    def key(self, model_cls: Type[AbstractModel]) -> str:
        assert self.exists(model_cls), f"Model class must be registered: {model_cls}"
        return model_cls.ag_key

    def name_map(self) -> dict[Type[AbstractModel], str]:
        return {model_cls: model_cls.ag_name for model_cls in self._model_cls_list}

    def name(self, model_cls: Type[AbstractModel]) -> str:
        assert self.exists(model_cls), f"Model class must be registered: {model_cls}"
        return model_cls.ag_name

    def priority(self, model_cls: Type[AbstractModel], problem_type: str | None = None) -> int:
        assert self.exists(model_cls), f"Model class must be registered: {model_cls}"
        return model_cls.get_ag_priority(problem_type=problem_type)

    def docstring(self, model_cls: Type[AbstractModel]) -> str:
        assert self.exists(model_cls), f"Model class must be registered: {model_cls}"
        return model_cls.__doc__

    # TODO: Could add a lot of information here to track which features are supported for each model:
    #  ag.early_stop support
    #  refit_full support
    #  GPU support
    #  etc.
    def to_frame(self) -> pd.DataFrame:
        model_classes = self.model_cls_list
        cls_dict = {}
        for model_cls in model_classes:
            cls_dict[self.key(model_cls)] = {
                "model_cls": model_cls.__name__,
                "ag_name": self.name(model_cls),
                "ag_priority": self.priority(model_cls),
            }
        df = pd.DataFrame(cls_dict).T
        df.index.name = "ag_key"
        return df
