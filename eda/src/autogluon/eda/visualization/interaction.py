from abc import ABC
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from .base import AbstractVisualization
from .jupyter import JupyterMixin
from ..state import AnalysisState

__all__ = ["CorrelationVisualization", "CorrelationSignificanceVisualization"]


class _AbstractCorrelationChart(AbstractVisualization, JupyterMixin, ABC):
    def __init__(
        self,
        headers: bool = False,
        namespace: Optional[str] = None,
        fig_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        if fig_args is None:
            fig_args = {}
        self.fig_args = fig_args

    def _render_internal(self, state: AnalysisState, render_key: str, header: str, chart_args: Dict[str, Any]) -> None:
        for ds, corr in state[render_key].items():
            # Don't render single cell
            if len(state.correlations[ds]) <= 1:
                continue

            if state.correlations_focus_field is not None:
                focus_field_header = f"; focus: absolute correlation for {state.correlations_focus_field} >= {state.correlations_focus_field_threshold}"
            else:
                focus_field_header = ""
            self.render_header_if_needed(state, f"{ds} - {state.correlations_method} {header}{focus_field_header}")

            fig, ax = plt.subplots(**self.fig_args)
            sns.heatmap(
                corr,
                annot=True,
                ax=ax,
                linewidths=0.9,
                linecolor="white",
                fmt=".2f",
                square=True,
                cbar_kws={"shrink": 0.5},
                **chart_args,
            )
            plt.yticks(rotation=0)
            plt.show(fig)


class CorrelationVisualization(_AbstractCorrelationChart):
    """
    Display feature correlations matrix.

    This report renders correlations between variable in a form of heatmap.
    The details of the report to be rendered depend on the configuration of
    :py:class:`~autogluon.eda.analysis.interaction.Correlation`

    Parameters
    ----------
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into chart figure

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.interaction.Correlation`
    """

    def can_handle(self, state: AnalysisState) -> bool:
        return "correlations" in state

    def _render(self, state: AnalysisState) -> None:
        args = {"vmin": 0 if state.correlations_method == "phik" else -1, "vmax": 1, "center": 0, "cmap": "Spectral"}
        self._render_internal(state, "correlations", "correlation matrix", args)


class CorrelationSignificanceVisualization(_AbstractCorrelationChart):
    """
    Display feature correlations significance matrix.

    This report renders correlations significance matrix in a form of heatmap.
    The details of the report to be rendered depend on the configuration of
    :py:class:`~autogluon.eda.analysis.interaction.Correlation` and
    :py:class:`~autogluon.eda.analysis.interaction.CorrelationSignificance` analyses.

    Parameters
    ----------
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    fig_args: Optional[Dict[str, Any]] = None,
        kwargs to pass into chart figure

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.interaction.Correlation`
    :py:class:`~autogluon.eda.analysis.interaction.CorrelationSignificance`
    """

    def can_handle(self, state: AnalysisState) -> bool:
        return "significance_matrix" in state

    def _render(self, state: AnalysisState) -> None:
        args = {"center": 3, "vmax": 5, "cmap": "Spectral", "robust": True}
        self._render_internal(state, "significance_matrix", "correlation significance matrix", args)
