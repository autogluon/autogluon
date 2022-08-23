from abc import ABC, abstractmethod
from typing import List

from autogluon.eda import AnalysisState


class AbstractVisualization(ABC):

    def __init__(self, namespace: str = None, **kwargs) -> None:
        super().__init__()
        self.namespace = namespace
        self._kwargs = kwargs
        if self.namespace is not None:
            self.namespace: List[str] = self.namespace.split('.')

    def get_namespace_state(self, state):
        if self.namespace is not None:
            for k in self.namespace:
                state = state[k]
        return state

    @abstractmethod
    def can_handle(self, state: AnalysisState) -> bool:
        raise NotImplemented

    @abstractmethod
    def _render(self, state: AnalysisState) -> None:
        raise NotImplemented

    def render(self, state: AnalysisState) -> None:
        state = self.get_namespace_state(state)
        if self.can_handle(state):
            self._render(state)
