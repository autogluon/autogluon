from abc import ABC, abstractmethod
from typing import Any, Dict


class RenderingBackend(ABC):

    @abstractmethod
    def render(self, model: Dict[str, Any]):
        raise NotImplemented


class EstimatorsBackend(ABC):

    # TODO: maybe restrict only to main use cases

    @abstractmethod
    def fit(self, **kwargs):
        raise NotImplemented

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplemented
