from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Tuple

from pandas import DataFrame

from ..state import AnalysisState, StateCheckMixin

logger = logging.getLogger(__name__)


class AbstractAnalysis(ABC, StateCheckMixin):
    def __init__(
        self,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        state: Optional[AnalysisState] = None,
        **kwargs,
    ) -> None:
        self.parent = parent
        self.children: List[AbstractAnalysis] = [] if children is None else children
        self.state: Optional[AnalysisState] = state
        for c in self.children:
            c.parent = self
            c.state = self.state
        self.args = kwargs

    def _gather_args(self) -> AnalysisState:
        chain = [self]
        while chain[0].parent is not None:
            chain.insert(0, chain[0].parent)
        args = AnalysisState()
        for node in chain:
            args = AnalysisState({**args, **node.args})
        return args

    @staticmethod
    def available_datasets(args: AnalysisState) -> Generator[Tuple[str, DataFrame], None, None]:
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
        for ds in ["train_data", "test_data", "tuning_data", "val_data"]:
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
        return state  # type: ignore

    @abstractmethod
    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        """
        Checks if state and args has all the required parameters for fitting.
        See also :func:`at_least_one_key_must_be_present` and :func:`all_keys_must_be_present` helpers
        to construct more complex logic.

        Parameters
        ----------
        state: AnalysisState
            state to be updated by this fit function
        args: AnalysisState
            analysis properties assembled from root of analysis hierarchy to this component (with lower levels shadowing upper level args).

        Returns
        -------
            `True` if all the pre-requisites for fitting are present

        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        if self.parent is not None:
            assert (
                self.state is not None
            ), "Inner analysis fit() is called while parent has no state. Please call top-level analysis fit instead"
        _args = self._gather_args()
        if self.can_handle(self.state, _args):
            self._fit(self.state, _args, **kwargs)
            for c in self.children:
                c.fit(**kwargs)
        return self.state


class BaseAnalysis(AbstractAnalysis):
    def __init__(
        self, parent: Optional[AbstractAnalysis] = None, children: Optional[List[AbstractAnalysis]] = None, **kwargs
    ) -> None:
        super().__init__(parent, children, **kwargs)

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        pass


class Namespace(AbstractAnalysis):
    """
    Creates a nested namespace in state. All the components within `children` will have relative root of the state moved into this subspace.
    To instruct visualization facets to use a specific subspace, please use `namespace` argument (see the example).

    Parameters
    ----------
    namespace: Optional[str], default = None
        namespace to use; use root if not specified
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call
    kwargs

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>>
    >>> auto.analyze(
    >>>     train_data=..., label=...,
    >>>     anlz_facets=[
    >>>         # Puts output into the root namespace
    >>>         eda.interaction.Correlation(),
    >>>         # Puts output into the focus namespace
    >>>         eda.Namespace(namespace='focus', children=[
    >>>             eda.interaction.Correlation(focus_field='Fare', focus_field_threshold=0.3),
    >>>         ])
    >>>     ],
    >>>     viz_facets=[
    >>>         # Renders correlations from the root namespace
    >>>         viz.interaction.CorrelationVisualization(),
    >>>         # Renders correlations from the focus namespace
    >>>         viz.interaction.CorrelationVisualization(namespace='focus'),
    >>>     ]
    >>> )

    """

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def __init__(
        self,
        namespace: Optional[str] = None,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)
        self.namespace = namespace

    def fit(self, **kwargs) -> AnalysisState:
        assert (
            self.parent is not None
        ), "Namespace must be wrapped into other analysis. You can use BaseAnalysis of one is needed"
        return super().fit(**kwargs)

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        pass

    def _get_state_from_parent(self) -> AnalysisState:
        state = super()._get_state_from_parent()
        if self.namespace not in state:
            state[self.namespace] = {}
        return state[self.namespace]
