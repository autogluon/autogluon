from __future__ import annotations

from typing import Type

import pandas as pd

from autogluon.features.generators.abstract import AbstractFeatureGenerator


class FeatureGeneratorRegistry:
    def __init__(self, cls_list: list[Type[AbstractFeatureGenerator]] | None = None):
        if cls_list is None:
            cls_list = []
        assert isinstance(cls_list, list)
        self._cls_list = []
        self._key_to_cls_map = dict()
        for fe_cls in cls_list:
            self.add(fe_cls)

    def exists(self, fe_cls: Type[AbstractFeatureGenerator]) -> bool:
        return fe_cls in self._cls_list

    def add(self, fe_cls: Type[AbstractFeatureGenerator]):
        """
        Adds `fe_cls` to the model registry
        """
        assert not self.exists(fe_cls), f"Cannot add fe_cls that is already registered: {fe_cls}"
        if fe_cls.__name__ is None:
            raise AssertionError(
                f"Cannot add fe_cls with `ag_key=None`. "
                f"Ensure you set class attribute `ag_key` to a string for your fe_cls: {fe_cls}"
                f'\n\tFor example, LightGBModel sets `ag_key = "GBM"`'
            )
        assert isinstance(fe_cls.__name__, str)
        if fe_cls.__name__ in self._key_to_cls_map:
            raise AssertionError(
                f"Cannot register a model class that shares a model key with an already registered model class."
                f"\n`fe_cls.ag_key` must be unique among registered models:"
                f"\n\t        New  Class: {fe_cls}"
                f"\n\tConflicting  Class: {self._key_to_cls_map[fe_cls.__name__]}"
                f"\n\tConflicting ag_key: {fe_cls.__name__}"
            )
        self._cls_list.append(fe_cls)
        self._key_to_cls_map[fe_cls.__name__] = fe_cls

    def remove(self, fe_cls: Type[AbstractFeatureGenerator]):
        """
        Removes `fe_cls` from the model registry
        """
        assert self.exists(fe_cls), f"Cannot remove fe_cls that isn't registered: {fe_cls}"
        self._cls_list = [m for m in self._cls_list if m != fe_cls]
        self._key_to_cls_map.pop(fe_cls.__name__)

    @property
    def cls_list(self) -> list[Type[AbstractFeatureGenerator]]:
        return self._cls_list

    @property
    def keys(self) -> list[str]:
        return [self.key(fe_cls) for fe_cls in self.cls_list]

    def key_to_cls_map(self) -> dict[str, Type[AbstractFeatureGenerator]]:
        return self._key_to_cls_map

    def key_to_cls(self, key: str) -> Type[AbstractFeatureGenerator]:
        if key not in self._key_to_cls_map:
            raise ValueError(
                f"No registered model exists with provided key: {key}"
                f"\n\tValid keys: {list(self.key_to_cls_map().keys())}"
            )
        return self.key_to_cls_map()[key]

    def key(self, fe_cls: Type[AbstractFeatureGenerator]) -> str:
        assert self.exists(fe_cls), f"Model class must be registered: {fe_cls}"
        return fe_cls.__name__

    def docstring(self, fe_cls: Type[AbstractFeatureGenerator]) -> str:
        assert self.exists(fe_cls), f"Model class must be registered: {fe_cls}"
        return fe_cls.__doc__

    # TODO: Could add a lot of information here to track which features are supported for each model:
    #  ag.early_stop support
    #  refit_full support
    #  GPU support
    #  etc.
    def to_frame(self) -> pd.DataFrame:
        model_classes = self.cls_list
        cls_dict = {}
        for fe_cls in model_classes:
            cls_dict[self.key(fe_cls)] = {
                "fe_cls": fe_cls.__name__,
            }
        df = pd.DataFrame(cls_dict).T
        df.index.name = "name"
        return df
