from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Union

from .. import AnalysisState

logger = logging.getLogger(__name__)


class AbstractAnalysis(ABC):

    def __init__(self,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:

        self.parent = parent
        self.children: List[AbstractAnalysis] = children
        self.state: AnalysisState = None
        for c in self.children:
            c.parent = self
            c.state = self.state
        self.args = kwargs

    def _gather_args(self) -> AnalysisState:
        chain = [self]
        while chain[0].parent is not None:
            chain.insert(0, chain[0].parent)
        args = {}
        for node in chain:
            args = AnalysisState({**args, **node.args})
        return args

    def _get_state_from_root(self) -> AnalysisState:
        pointer = self
        while pointer.parent is not None:
            pointer = pointer.parent
        state = pointer.state
        if state is None:
            state = AnalysisState()
        return state

    def _get_state_from_parent(self) -> AnalysisState:
        state = self.state
        if state is None:
            if self.parent is None:
                state = AnalysisState()
            else:
                state = self.parent.state
        return state

    @abstractmethod
    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        """
        Fit composing EDA primitives.
        """
        raise NotImplemented

    def fit(self, **kwargs) -> AnalysisState:
        self.state = self._get_state_from_parent()
        self._fit(self.state, self._gather_args(), **kwargs)
        for c in self.children:
            c.fit(**kwargs)
        return self.state


class NamespaceAbstractAnalysis(AbstractAnalysis, ABC):
    def __init__(self,
                 namespace: str = None,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.namespace = namespace

    def _get_state_from_parent(self) -> AnalysisState:
        state = super()._get_state_from_parent()
        state[self.namespace] = {}
        return state[self.namespace]


class Namespace(NamespaceAbstractAnalysis):

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        pass
