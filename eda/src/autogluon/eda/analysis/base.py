from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Union, Tuple

from pandas import DataFrame

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

    def available_datasets(self, args: AnalysisState) -> Tuple[str, DataFrame]:
        """
        Generator which iterates only through the datasets provided in arguments

        Parameters
        ----------
        args: AnalysisState
            arguments passed into the call. These are different from `self.args` in a way that it's arguments assembled from the
            parents and shadowed via children (allows to isolate reused parameters in upper arguments declarations.

        Returns
        -------
            tuple of dataset name (train_data, test_data or tuning_data) and dataset itself

        """
        for ds in ['train_data', 'test_data', 'tuning_data', 'val_data']:
            if ds in args and args[ds] is not None:
                df: DataFrame = args[ds]
                yield ds, df

    def _get_state_from_parent(self) -> AnalysisState:
        state = self.state
        if state is None:
            if self.parent is None:
                state = AnalysisState()
            else:
                state = self.parent.state
        return state

    @abstractmethod
    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        """
        @override
        Component-specific internal processing.
        This method is designed to be overridden by the component developer

        Parameters
        ----------
        state: AnalysisState
            state to be updated by this fit function
        args: AnalysisState
            analysis properties assembled from root of analysis hierarchy to this component (with lower levels shadowing upper level args).
        fit_kwargs
            arguments passed into fit call
        """
        raise NotImplemented

    def fit(self, **kwargs) -> AnalysisState:
        """
        Fit the analysis tree.

        Parameters
        ----------
        kwargs
            fit arguments

        Returns
        -------
            state produced by fit

        """
        self.state = self._get_state_from_parent()
        self._fit(self.state, self._gather_args(), **kwargs)
        for c in self.children:
            c.fit(**kwargs)
        return self.state


class Namespace(AbstractAnalysis, ABC):
    def __init__(self,
                 namespace: str = None,
                 parent: Union[None, AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.namespace = namespace

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        pass

    def _get_state_from_parent(self) -> AnalysisState:
        state = super()._get_state_from_parent()
        state[self.namespace] = {}
        return state[self.namespace]
