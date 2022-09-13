import logging
from abc import ABC, abstractmethod
from typing import List

from .. import AnalysisState

logger = logging.getLogger(__name__)


class AbstractVisualization(ABC):

    def __init__(self, namespace: str = None, **kwargs) -> None:
        """
        Parameters
        ----------
        namespace: str
            namespace to use; can be nested like `ns_a.ns_b.ns_c`
        kwargs
        """
        super().__init__()
        self.namespace = namespace
        self._kwargs = kwargs
        if self.namespace is not None:
            self.namespace: List[str] = self.namespace.split('.')

    def _get_namespace_state(self, state):
        if self.namespace is not None:
            for k in self.namespace:
                state = state[k]
        return state

    @abstractmethod
    def can_handle(self, state: AnalysisState) -> bool:
        """
        Checks if state has all the required parameters for visualization.
        See also :func:`_at_least_one_key_must_be_present` and :func:`_all_keys_must_be_present` helpers
        to construct more complex logic.

        Parameters
        ----------
        state: AnalysisState
            fitted state

        Returns
        -------
            `True` if all the pre-requisites for rendering are present

        """
        raise NotImplemented

    def _at_least_one_key_must_be_present(self, state: AnalysisState, keys: List[str]):
        for k in keys:
            if k in state:
                return True
        logger.warning(f'{self.__class__.__name__}: at least one of the following keys must be present: {keys}')
        return False

    def _all_keys_must_be_present(self, state: AnalysisState, keys: List[str]):
        keys_not_present = [k for k in keys if k not in state.keys()]
        can_handle = len(keys_not_present) == 0
        if not can_handle:
            logger.warning(f'{self.__class__.__name__}: all of the following keys must be present: {keys}. The following keys are missing: {keys_not_present}')
        return can_handle

    @abstractmethod
    def _render(self, state: AnalysisState) -> None:
        """
        @override
        Component-specific rendering logic.
        This method is designed to be overridden by the component developer.

        Parameters
        ----------
        state

        Returns
        -------

        """
        raise NotImplemented

    def render(self, state: AnalysisState) -> None:
        """
        Render component.

        Parameters
        ----------
        state: AnalysisState
            state to render

        """
        state = self._get_namespace_state(state)
        if self.can_handle(state):
            self._render(state)
