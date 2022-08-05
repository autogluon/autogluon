from abc import ABC, abstractmethod

from pandas import DataFrame


class RenderingBackend(ABC):

    @abstractmethod
    def render_text(self, text: str, text_type=None):
        raise NotImplemented

    @abstractmethod
    def render_table(self, df: DataFrame):
        raise NotImplemented

    @abstractmethod
    def render_histogram(self, df: DataFrame, **kwargs):
        raise NotImplemented


class EstimatorsBackend(ABC):

    # TODO: maybe restrict only to main use cases

    @abstractmethod
    def fit(self, **kwargs):
        raise NotImplemented

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplemented
