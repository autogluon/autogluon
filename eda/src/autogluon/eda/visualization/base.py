import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from ..state import AnalysisState, StateCheckMixin

logger = logging.getLogger(__name__)


class AbstractVisualization(ABC, StateCheckMixin):
    """
    Base class for visualization functionality.

    Provides basic functionality for namespace management and helper method to access frequently-used methods.

    Specifying namespace would narrow visibility scope to specific subspace of `state`. Namespaces
    can be specified in a nested form: `ns_a.ns_b.ns_c`. Please see :py:class:`~autogluon.eda.analysis.base.Namespace`
    wrapper on how to create namespaces.

    The main entry method of analysis is `render` function. When called, the execution flow is the following:
    - narrow `state` scope to specified `namespace`
    - call `_render` function for each component that returned `True` from `can_handle` call

    Parameters
    ----------
    namespace: str
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    kwargs

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.base.Namespace`
    """

    def __init__(self, namespace: Optional[str] = None, **kwargs) -> None:
        super().__init__()
        self.namespace: List[str] = []
        self._kwargs = kwargs
        if namespace is not None:
            self.namespace = namespace.split(".")

    def _get_namespace_state(self, state):
        if self.namespace is not None:
            for k in self.namespace:
                state = state[k]
        return state

    @abstractmethod
    def can_handle(self, state: AnalysisState) -> bool:
        """
        Checks if state has all the required parameters for visualization.
        See also :func:`at_least_one_key_must_be_present` and :func:`all_keys_must_be_present` helpers
        to construct more complex logic.

        Parameters
        ----------
        state: AnalysisState
            fitted state

        Returns
        -------
            `True` if all the pre-requisites for rendering are present

        """
        raise NotImplementedError

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
        raise NotImplementedError

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
