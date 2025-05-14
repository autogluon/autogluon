from abc import ABCMeta
from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelRecord:
    model_class: type
    default_priority: int


class ModelRegistry(type):
    REGISTRY: Dict[str, ModelRecord] = {}

    def __new__(cls, name, bases, attrs):
        """See, https://github.com/faif/python-patterns."""
        new_cls = type.__new__(cls, name, bases, attrs)

        record = ModelRecord(
            model_class=new_cls,
            default_priority=getattr(new_cls, "default_priority", 0),
        )

        if name is not None:
            cls._add(name, record)

            # additionally, if the name is like XXModel, alias it as "XX"
            if name.endswith("Model") and len(name) > 5:
                cls._add(name[:-5], record)

        # if the class provides additional aliases, register them too
        if aliases := attrs.get("_aliases"):
            for alias in aliases:
                cls._add(alias, record)

        return new_cls

    @classmethod
    def _add(cls, alias: str, record: ModelRecord):
        if alias in cls.REGISTRY:
            raise ValueError(f"You are trying to define a new model with {alias}, but this model already exists.")
        cls.REGISTRY[alias] = record

    @classmethod
    def get_model_class(cls, alias: str):
        return cls.REGISTRY[alias].model_class

    @classmethod
    def get_model_priority(cls, alias: str):
        return cls.REGISTRY[alias].default_priority

    @classmethod
    def available_aliases(cls):
        return list(cls.REGISTRY.keys())


class RegisteredABCMeta(ModelRegistry, ABCMeta):
    pass


class RegisteredABC(metaclass=RegisteredABCMeta):
    pass
