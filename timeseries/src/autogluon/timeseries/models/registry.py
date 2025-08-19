from abc import ABCMeta
from dataclasses import dataclass
from inspect import isabstract
from typing import Dict, List, Union


@dataclass
class ModelRecord:
    model_class: type
    ag_priority: int


class ModelRegistry(ABCMeta):
    """Registry metaclass for time series models. Ensures that TimeSeriesModel classes
    which implement this metaclass are automatically registered, in order to centralize
    access to model types.

    See, https://github.com/faif/python-patterns.
    """

    REGISTRY: Dict[str, ModelRecord] = {}

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)

        if name is not None and not isabstract(new_cls):
            record = ModelRecord(
                model_class=new_cls,
                ag_priority=getattr(new_cls, "ag_priority", 0),
            )
            cls._add(name.removesuffix("Model"), record)

            # if the class provides additional aliases, register them too
            if aliases := attrs.get("_aliases"):
                for alias in aliases:
                    cls._add(alias, record)

        return new_cls

    @classmethod
    def _add(cls, alias: str, record: ModelRecord) -> None:
        if alias in cls.REGISTRY:
            raise ValueError(f"You are trying to define a new model with {alias}, but this model already exists.")
        cls.REGISTRY[alias] = record

    @classmethod
    def _get_model_record(cls, alias: Union[str, type]) -> ModelRecord:
        if isinstance(alias, type):
            alias = alias.__name__
        alias = alias.removesuffix("Model")
        if alias not in cls.REGISTRY:
            raise ValueError(f"Unknown model: {alias}, available models are: {cls.available_aliases()}")
        return cls.REGISTRY[alias]

    @classmethod
    def get_model_class(cls, alias: Union[str, type]) -> type:
        return cls._get_model_record(alias).model_class

    @classmethod
    def get_model_priority(cls, alias: Union[str, type]) -> int:
        return cls._get_model_record(alias).ag_priority

    @classmethod
    def available_aliases(cls) -> List[str]:
        return sorted(cls.REGISTRY.keys())
