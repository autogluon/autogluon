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
